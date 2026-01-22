import os
import sys
import pandas as pd
import numpy as np
import pytz
import json
import warnings
from datetime import datetime, timedelta
from collections import deque

# --- 1. System Path & Imports ---
sys.path.append(r'C:\WFM_Scripting\Automation')
from scripthelper import Config, Logger, BigQueryManager, EmailManager, GeneralFuncs

# Third party ML/Stat imports
try:
    import statsmodels.api as sm
    from xgboost import XGBRegressor
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from scipy.stats import pearsonr
except ImportError as e:
    print(f"Warning: ML libraries missing ({e}). Script may fail.")

# --- 2. Global Initialization ---
def initialize_globals():
    global config, logger, bigquery_manager, email_manager, general_funcs
    global rpt_id, is_test
    
    rpt_id = 288
    config = Config(rpt_id=rpt_id)
    logger = config.logger
    bigquery_manager = BigQueryManager(config)
    email_manager = EmailManager(config)
    general_funcs = GeneralFuncs(config)
    
    # Force Test Mode
    is_test = True 

    # --- Constants & SQL ---
    global GBQ_MAIN_QRY, GBQ_FRED_HIST_QRY, GBQ_FRED_FX_QRY, GBQ_CALENDAR_FILE
    global LOCAL_CSV, AUDIT_CSV, STATS_CSV, XGB_VIZ_CSV, MODELING_DATA_CSV, SUMMARIES_FILE
    global FORECAST_HORIZON, TEST_LEN, LAGS, STAMP, FX_ID_PREFIX
    global FRED_COLS, CALENDAR_COLS, ALL_EXOG_COLS
    global BACKCAST_MODE, BACKCAST_START_DATE
    
    # --- BACKCAST SETTINGS ---
    BACKCAST_MODE       = False 
    BACKCAST_START_DATE = '2025-01-01'

    # Data Queries
    GBQ_MAIN_QRY = """
    SELECT
      MonthOfOrder AS date,
      client_id,
      COALESCE(product, 'Total') AS product,
      TotalOrders  AS target_volume
    FROM `tax_clnt_svcs.view_cx_nontax_platforms_workload_drivers`
    ORDER BY MonthOfOrder
    """
    
    GBQ_FRED_HIST_QRY = """
    SELECT DISTINCT
        Date, UNRATE, HSN1F, FEDFUNDS, MORTGAGE30US
    FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.fred`
    ORDER BY Date
    """
    
    GBQ_FRED_FX_QRY   = """
    SELECT 
        Date, UNRATE, HSN1F, FEDFUNDS, MORTGAGE30US, Forecast_Date 
    FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.fred_fx`
    QUALIFY ROW_NUMBER() OVER(PARTITION BY Date ORDER BY Forecast_Date DESC) = 1
    ORDER BY Date
    """
    
    # Calendar SQL File Path
    GBQ_CALENDAR_FILE = r"C:\WFM_Scripting\Forecasting\GBQ - Calendar days.sql"

    # --- Local Outputs ---
    suffix = "_BACKCAST" if BACKCAST_MODE else ""
    LOCAL_CSV         = fr"C:\WFM_Scripting\forecast_results{suffix}.csv"
    AUDIT_CSV         = fr"C:\WFM_Scripting\model_eval_debug{suffix}.csv"
    STATS_CSV         = fr"C:\WFM_Scripting\statistical_tests{suffix}.csv"
    XGB_VIZ_CSV       = fr"C:\WFM_Scripting\xgb_feature_importance{suffix}.csv"
    MODELING_DATA_CSV = fr"C:\WFM_Scripting\modeling_data{suffix}.csv"
    SUMMARIES_FILE    = fr"C:\WFM_Scripting\model_summaries{suffix}.txt"

    FORECAST_HORIZON = 15
    TEST_LEN         = 6
    LAGS             = (3, 6, 12)
    
    # Feature Columns
    FRED_COLS = ["UNRATE", "HSN1F", "FEDFUNDS", "MORTGAGE30US"] 
    CALENDAR_COLS = ["total_days", "weekday_count", "weekend_day_count", "holiday_count", "business_day_count"]
    
    ALL_EXOG_COLS = FRED_COLS + CALENDAR_COLS

    STAMP_TZ = pytz.timezone("America/Chicago")
    STAMP    = datetime.now(STAMP_TZ)

    FX_ID_PREFIX = {
        "SeasonalNaive":           "snaive_workload",
        "SeasonalNaiveGR":         "snaive_gr_workload",
        "Native3m":                "native3m_workload",
        "MLR":                     "mlr_workload",
        "XGBoost":                 "xgb_workload",
        "SARIMA":                  "sarima_workload"
    }

    warnings.filterwarnings("ignore")

# ------------------- Metrics & Helpers ----------------------

def smape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    denom = (np.abs(a) + np.abs(f)) / 2.0
    out = np.where(denom == 0.0, 0.0, np.abs(a - f) / denom)
    return float(np.mean(out) * 100.0)

def mape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    return float(np.mean(np.abs((a - f) / np.where(a == 0, 1, a))) * 100.0)

def rmse(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    return float(np.sqrt(np.mean((a - f) ** 2)))

def perform_statistical_tests(df, cid, prod, target_col='y'):
    """ Runs ADF for stationarity and Pearson Correlation for feature relevance """
    results = {'client_id': cid, 'product': prod}
    
    # 1. ADF Stationarity
    series = df[target_col].dropna()
    if len(series) > 8:
        try:
            adf_res = adfuller(series)
            results['adf_p_value'] = adf_res[1]
            results['is_stationary'] = adf_res[1] < 0.05
        except:
            results['adf_p_value'] = np.nan
    else:
        results['adf_p_value'] = np.nan

    # 2. Correlations
    correlations = {}
    potential_cols = [c for c in df.columns if "lag" in c or c in ALL_EXOG_COLS]
    
    for col in potential_cols:
        if col not in df.columns: continue
        mask = df[target_col].notna() & df[col].notna()
        if mask.sum() > 4:
            try:
                corr, p_val = pearsonr(df.loc[mask, target_col], df.loc[mask, col])
                if not np.isnan(corr):
                    correlations[col] = {'r': corr, 'p': p_val}
            except: pass
            
    top_corr = sorted(correlations.items(), key=lambda x: abs(x[1]['r']), reverse=True)[:5]
    results['top_correlations'] = json.dumps({k: round(v['r'], 3) for k, v in top_corr})
    
    return results

# ------------------- Data Prep -----------------------------

def prepare_exog_data(df_hist, df_fx, df_cal):
    """ Merge historical/forecast FRED data AND Calendar data """
    # 1. Clean FRED
    # FIX: Rename only target Date to 'date'. Do NOT rename Forecast_Date to 'date'.
    df_hist = df_hist.rename(columns={"Date": "date"})
    df_fx = df_fx.rename(columns={"Date": "date"}) 
    
    df_hist['date'] = pd.to_datetime(df_hist['date'])
    df_fx['date']   = pd.to_datetime(df_fx['date'])
    
    df_hist = df_hist.set_index('date').sort_index()
    df_fx   = df_fx.set_index('date').sort_index()
    
    # Combine FRED: Hist + FX
    fred_full = df_hist.combine_first(df_fx)
    
    # 2. Clean Calendar
    df_cal = df_cal.rename(columns={"month_start": "date"})
    df_cal['date'] = pd.to_datetime(df_cal['date'])
    df_cal = df_cal.set_index('date').sort_index()
    
    # 3. Merge All
    exog_master = fred_full.join(df_cal, how='outer')
    
    # Fill FRED holes (if any)
    for c in FRED_COLS:
        if c in exog_master.columns:
            exog_master[c] = pd.to_numeric(exog_master[c], errors='coerce').ffill().bfill()
        else:
            exog_master[c] = 0.0
            
    # Fill Calendar holes
    for c in CALENDAR_COLS:
        if c in exog_master.columns:
            exog_master[c] = pd.to_numeric(exog_master[c], errors='coerce').ffill()
        else:
            exog_master[c] = 0.0
            
    return exog_master[ALL_EXOG_COLS]

def add_lags(df, target_col="y", lag_cols=[]) -> pd.DataFrame:
    """ 
    Create Lags for Target and specified External Vars (FRED).
    Note: Calendar vars are usually NOT lagged (used concurrently).
    """
    out = df.copy()
    # Lag Target
    for L in LAGS:
        out[f"lag_{L}"] = out[target_col].shift(L)
    
    # Lag specified columns (FRED only)
    for col in lag_cols:
        for L in LAGS:
            if col in out.columns:
                out[f"{col}_lag_{L}"] = out[col].shift(L)
    return out

# ------------------- Model Logic ----------------------------

def native_3m_growth_series(ts, idx):
    out, s12 = [], ts.shift(12)
    for d in idx:
        base = s12.get(d, np.nan)
        if not np.isfinite(base) or d not in ts.index:
            out.append(np.nan); continue
        
        pos = ts.index.get_loc(d)
        if pos < 18: 
            out.append(base); continue
            
        cur_3m = ts.iloc[pos-3:pos].mean()
        py_3m  = ts.iloc[pos-15:pos-12].mean() 
        
        growth_factor = (cur_3m / py_3m) if py_3m > 0 else 1.0
        out.append(max(0.0, base * growth_factor))
    return pd.Series(out, index=idx, dtype=float)

def seasonal_naive_gr_series(ts, idx):
    out, s12 = [], ts.shift(12)
    for d in idx:
        base = s12.get(d, np.nan)
        if not np.isfinite(base) or d not in ts.index:
            out.append(np.nan); continue
        pos = ts.index.get_loc(d)
        if pos < 6: r = 0.0
        else:
            prev = ts.iloc[pos-6:pos-3].mean()
            curr = ts.iloc[pos-3:pos].mean()
            r = (curr / prev - 1.0) if prev > 0 else 0.0
        out.append(max(0.0, base * (1.0 + r)))
    return pd.Series(out, index=idx, dtype=float)

# ------------------- Core Process ---------------------------

def run_forecast_logic():
    mode_msg = f"BACKCAST MODE (Cutoff: {BACKCAST_START_DATE})" if BACKCAST_MODE else "PRODUCTION MODE"
    logger.info(f"Starting Workload Forecasting - {mode_msg}")

    # 1. Fetch Data
    logger.info("Fetching Data (Main, FRED, Calendar)...")
    df_main    = bigquery_manager.run_gbq_sql(GBQ_MAIN_QRY, return_dataframe=True)
    df_fred_h  = bigquery_manager.run_gbq_sql(GBQ_FRED_HIST_QRY, return_dataframe=True)
    df_fred_f  = bigquery_manager.run_gbq_sql(GBQ_FRED_FX_QRY, return_dataframe=True)
    
    try:
        with open(GBQ_CALENDAR_FILE, 'r') as f:
            cal_sql = f.read()
        df_calendar = bigquery_manager.run_gbq_sql(cal_sql, return_dataframe=True)
    except Exception as e:
        logger.warning(f"Could not read local SQL file, passing path: {e}")
        df_calendar = bigquery_manager.run_gbq_sql(GBQ_CALENDAR_FILE, return_dataframe=True)

    if df_main.empty: raise ValueError("Main dataset empty.")
    if df_calendar.empty: raise ValueError("Calendar dataset empty.")
    
    # Prepare Master Exog (FRED + Calendar)
    exog_master = prepare_exog_data(df_fred_h, df_fred_f, df_calendar)
    
    df_main["date"] = pd.to_datetime(df_main["date"])
    
    # --- APPLY BACKCAST FILTERING ---
    if BACKCAST_MODE:
        cutoff = pd.to_datetime(BACKCAST_START_DATE)
        orig_len = len(df_main)
        df_main = df_main[df_main["date"] < cutoff]
        logger.info(f"Backcast Truncation: Reduced data from {orig_len} to {len(df_main)} rows (Data < {cutoff})")
    
    df_main = df_main.sort_values(["client_id", "product", "date"])

    forecasts, audit_rows, stat_test_results = [], [], []
    xgb_imp_rows, all_modeling_data, model_summaries = [], [], []

    # 2. Group Loop
    for (cid, prod), g in df_main.groupby(["client_id", "product"]):
        
        ts = g.set_index("date")["target_volume"].asfreq("MS").fillna(0)
        
        # Merge Exog (Left join TS with Master)
        df_merged = pd.DataFrame({"y": ts}).join(exog_master, how="left").ffill().bfill()
        
        # Add Lags (Only for FRED cols, preserve Calendar cols as is)
        feat_df = add_lags(df_merged, target_col="y", lag_cols=FRED_COLS)
        feat_df = feat_df.dropna()
        
        if feat_df.shape[0] <= (TEST_LEN + 6):
            logger.info(f"Skipping {cid}-{prod}: Insufficient history.")
            continue

        # --- A. Statistical Tests ---
        stat_res = perform_statistical_tests(feat_df, cid, prod)
        stat_test_results.append(stat_res)
        
        # --- B. Save Modeling Data ---
        dump_df = feat_df.copy()
        dump_df['client_id'] = cid
        dump_df['product'] = prod
        all_modeling_data.append(dump_df)

        # Split
        val_idx = feat_df.index[-TEST_LEN:]
        train_idx = feat_df.index[:-TEST_LEN]
        df_train = feat_df.loc[train_idx]
        df_val   = feat_df.loc[val_idx]
        ts_full  = df_merged["y"]

        # Define Features: Lags + Calendar Columns (Concurrent)
        lag_feats = [c for c in feat_df.columns if "lag" in c]
        cal_feats = [c for c in CALENDAR_COLS if c in feat_df.columns]
        feature_cols = lag_feats + cal_feats

        preds, models_cache = {}, {}

        # --- C. Naive Models ---
        preds["SeasonalNaive"]   = ts_full.shift(12).reindex(val_idx)
        preds["SeasonalNaiveGR"] = seasonal_naive_gr_series(ts_full, val_idx)
        preds["Native3m"]        = native_3m_growth_series(ts_full, val_idx)

        # --- D. ML Models (XGBoost) ---
        try:
            xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
            xgb.fit(df_train[feature_cols], df_train['y'])
            p_xgb = xgb.predict(df_val[feature_cols])
            preds["XGBoost"] = pd.Series(np.maximum(0, p_xgb), index=val_idx)
            models_cache["XGBoost"] = xgb
            
            gain = xgb.get_booster().get_score(importance_type='gain')
            imp_sorted = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)
            xgb_imp_rows.append({"client_id": cid, "product": prod, "importance_json": json.dumps(imp_sorted)})
        except Exception as e:
            logger.warning(f"XGB failed {cid}: {e}")

        # --- E. ML Models (MLR - Statsmodels OLS) ---
        X_tr_ols = sm.add_constant(df_train[feature_cols])
        X_val_ols = sm.add_constant(df_val[feature_cols], has_constant='add')
        
        try:
            ols = sm.OLS(df_train['y'], X_tr_ols).fit()
            p_mlr = ols.predict(X_val_ols)
            preds["MLR"] = pd.Series(np.maximum(0, p_mlr), index=val_idx)
            models_cache["MLR"] = ols
            
            summ_text = f"\n--- Client: {cid} | Prod: {prod} | Model: MLR (OLS) ---\n{ols.summary().as_text()}\n"
            model_summaries.append(summ_text)
        except Exception as e:
            logger.warning(f"MLR Failed for {cid}-{prod}: {e}")

        # --- F. SARIMA ---
        try:
            sar = SARIMAX(ts_full.loc[train_idx], order=(1,1,1), seasonal_order=(0,1,0,12), 
                          enforce_stationarity=False, enforce_invertibility=False)
            sar_res = sar.fit(disp=False)
            p_sar = sar_res.forecast(steps=len(val_idx))
            preds["SARIMA"] = pd.Series(np.maximum(0, p_sar.values), index=val_idx)
            models_cache["SARIMA"] = sar_res
        except: pass

        # --- G. Evaluate & Select Winner ---
        smapes = {}
        for m, pser in preds.items():
            if pser.isna().all(): continue
            y_true = ts_full.loc[val_idx]
            s = smape(y_true, pser)
            smapes[m] = s
            audit_rows.append({
                "client_id": cid, "product": prod, "model": m,
                "SMAPE": s, "MAPE": mape(y_true, pser), "RMSE": rmse(y_true, pser)
            })

        if not smapes: continue
        
        priority = ["Native3m", "MLR", "XGBoost", "SeasonalNaiveGR", "SeasonalNaive"]
        best_model = min(smapes, key=lambda k: (smapes[k], priority.index(k) if k in priority else 99))
        logger.info(f"· {cid} [{prod}] Winner: {best_model} ({smapes[best_model]:.2f}%)")

        # --- H. Recursive Forecast (15 mo) ---
        last_date = ts_full.index.max()
        future_dates = pd.date_range(last_date, periods=FORECAST_HORIZON+1, freq="MS")[1:]
        
        history_df = df_merged.copy() 

        for fd in future_dates:
            new_row = pd.DataFrame(index=[fd], columns=history_df.columns)
            
            # 1. Fill ALL Exog (FRED + Calendar) from Master
            if fd in exog_master.index:
                for c in ALL_EXOG_COLS: 
                    new_row.at[fd, c] = exog_master.at[fd, c]
            else:
                for c in ALL_EXOG_COLS: 
                    new_row.at[fd, c] = history_df.iloc[-1][c]
            
            history_df = pd.concat([history_df, new_row])
            
            # 2. Recalc Lags for prediction row
            tail_df = history_df.iloc[-(max(LAGS)+5):].copy()
            feat_tail = add_lags(tail_df, target_col="y", lag_cols=FRED_COLS)
            
            # Extract features for this specific date
            X_pred = feat_tail.iloc[[-1]][feature_cols]
            
            # 3. Predict
            pred_val = 0.0
            
            if best_model == "MLR" and "MLR" in models_cache:
                X_pred_ols = sm.add_constant(X_pred, has_constant='add')
                pred_val = float(models_cache["MLR"].predict(X_pred_ols)[0])
                
            elif best_model == "XGBoost" and "XGBoost" in models_cache:
                pred_val = float(models_cache["XGBoost"].predict(X_pred)[0])
                
            elif best_model == "Native3m":
                if len(history_df) >= 16:
                    base = history_df.iloc[-13]["y"]
                    curr_3m = history_df.iloc[-4:-1]["y"].mean()
                    prior_3m = history_df.iloc[-16:-13]["y"].mean()
                    gf = (curr_3m / prior_3m) if prior_3m > 0 else 1.0
                    pred_val = base * gf
                else: pred_val = history_df.iloc[-2]["y"]
            
            elif best_model == "SeasonalNaiveGR":
                 if len(history_df) >= 13:
                    base = history_df.iloc[-13]["y"]
                    curr_3m = history_df.iloc[-4:-1]["y"].mean()
                    prev_3m = history_df.iloc[-7:-4]["y"].mean()
                    r = (curr_3m / prev_3m - 1) if prev_3m > 0 else 0
                    pred_val = base * (1 + r)
                 else: pred_val = history_df.iloc[-2]["y"]
            
            else: 
                pred_val = history_df.iloc[-13]["y"] if len(history_df) >= 13 else 0

            pred_val = max(0.0, pred_val)
            history_df.at[fd, "y"] = pred_val
            
            fx_tag = f"{FX_ID_PREFIX.get(best_model, 'other')}_{STAMP:%Y%m%d}"
            forecasts.append({
                "fx_date": fd.strftime("%Y-%m-01"),
                "client_id": cid, "product": prod,
                "fx_vol": int(round(pred_val)),
                "fx_id": fx_tag, "fx_status": "A",
                "load_ts": STAMP.strftime("%Y-%m-%d %H:%M:%S")
            })
            
        audit_rows.append({"client_id": cid, "product": prod, "model": f"{best_model}_WINNER", 
                           "SMAPE": smapes[best_model], "MAPE": 0, "RMSE": 0})

    # 3. Save All Outputs Locally
    if forecasts:
        fx_df = pd.DataFrame(forecasts)
        logger.info(f"Writing {len(fx_df)} forecast rows locally to {LOCAL_CSV}...")
        
        fx_df.to_csv(LOCAL_CSV, index=False)
        
        if audit_rows: pd.DataFrame(audit_rows).to_csv(AUDIT_CSV, index=False)
        if stat_test_results: pd.DataFrame(stat_test_results).to_csv(STATS_CSV, index=False)
        if xgb_imp_rows: pd.DataFrame(xgb_imp_rows).to_csv(XGB_VIZ_CSV, index=False)
        if all_modeling_data: pd.concat(all_modeling_data).to_csv(MODELING_DATA_CSV, index=True)
        if model_summaries:
            with open(SUMMARIES_FILE, 'w') as f:
                f.write("\n".join(model_summaries))

        logger.info(f"✓ All files saved to C:\\WFM_Scripting\\")
    else:
        logger.warning("No forecasts generated.")

# ------------------- Main Execution -------------------------
if __name__ == '__main__':
    initialize_globals()
    try:
        run_forecast_logic()
    except Exception as exc:
        logger.error(f"Script failed: {exc}")
        if 'email_manager' in globals():
            email_manager.handle_error("Forecast Script Failed", exc, is_test=is_test) --- (venv_Master) PS C:\WFM_Scripting\Forecasting> & C:/Scripting/Python_envs/venv_Master/Scripts/python.exe c:/WFM_Scripting/Forecasting/Rpt_288_File.py
Traceback (most recent call last):
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\frame.py", line 4339, in _set_value
    self._mgr.column_setitem(icol, iindex, value, inplace_only=True)
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\internals\managers.py", line 1306, in column_setitem
    col_mgr.setitem_inplace(idx, value)
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\internals\managers.py", line 1991, in setitem_inplace
    super().setitem_inplace(indexer, value)
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\internals\base.py", line 332, in setitem_inplace
    arr[indexer] = value
    ~~~^^^^^^^^^
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\arrays\masked.py", line 292, in __setitem__
    value = self._validate_setitem_value(value)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\arrays\masked.py", line 283, in _validate_setitem_value
    raise TypeError(f"Invalid value '{str(value)}' for dtype {self.dtype}")
TypeError: Invalid value '59286.28261951567' for dtype Int64

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "c:\WFM_Scripting\Forecasting\Rpt_288_File.py", line 499, in <module>
    email_manager.handle_error("Forecast Script Failed", exc, is_test=is_test)
  File "C:\WFM_Scripting\Automation\scripthelper.py", line 1742, in handle_error
    raise exception
  File "c:\WFM_Scripting\Forecasting\Rpt_288_File.py", line 495, in <module>
    run_forecast_logic()
  File "c:\WFM_Scripting\Forecasting\Rpt_288_File.py", line 458, in run_forecast_logic
    history_df.at[fd, "y"] = pred_val
    ~~~~~~~~~~~~~^^^^^^^^^
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\indexing.py", line 2499, in __setitem__
    return super().__setitem__(key, value)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\indexing.py", line 2455, in __setitem__
    self.obj._set_value(*key, value=value, takeable=self._takeable)
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\frame.py", line 4351, in _set_value
    self.loc[index, col] = value
    ~~~~~~~~^^^^^^^^^^^^
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\indexing.py", line 885, in __setitem__
    iloc._setitem_with_indexer(indexer, value, self.name)
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\indexing.py", line 1893, in _setitem_with_indexer
    self._setitem_with_indexer_split_path(indexer, value, name)
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\indexing.py", line 1986, in _setitem_with_indexer_split_path
    self._setitem_single_column(loc, value, pi)
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\indexing.py", line 2095, in _setitem_single_column
    self.obj._mgr.column_setitem(loc, plane_indexer, value)
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\internals\managers.py", line 1308, in column_setitem
    new_mgr = col_mgr.setitem((idx,), value)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\internals\managers.py", line 399, in setitem
    return self.apply("setitem", indexer=indexer, value=value)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\internals\managers.py", line 354, in apply
    applied = getattr(b, f)(**kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\internals\blocks.py", line 1758, in setitem
    values[indexer] = value
    ~~~~~~^^^^^^^^^
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\arrays\masked.py", line 292, in __setitem__
    value = self._validate_setitem_value(value)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\arrays\masked.py", line 283, in _validate_setitem_value
    raise TypeError(f"Invalid value '{str(value)}' for dtype {self.dtype}")
TypeError: Invalid value '59286.28261951567' for dtype Int64
