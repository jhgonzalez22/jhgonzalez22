"""
Credit Tickets - Monthly Volume Forecast — v1.17 (Final Hotfix)

This version includes a hotfix to resolve a NameError by re-introducing the 'mape'
metric calculation within the safe_reg_metrics helper function.
"""

# ------------------- standard imports -----------------------
import os
import sys
import warnings
import pytz
import json
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# ------------------- helper-package path --------------------
sys.path.append(r"C:\WFM_Scripting\Automation")
from scripthelper import Config, Logger, BigQueryManager, EmailManager

# ------------------- initialise scripthelper ---------------
config           = Config(rpt_id=198)
logger           = Logger(config)
email_manager    = EmailManager(config)
bigquery_manager = BigQueryManager(config)

# ------------------- file / table paths --------------------
LOCAL_CSV          = r"C:\WFM_Scripting\credit_ticket_forecast_results.csv"
AUDIT_CSV          = r"C:\WFM_Scripting\credit_ticket_model_eval_debug.csv"
MODELING_DATA_CSV  = r"C:\WFM_Scripting\credit_ticket_modeling_data.csv"
STATS_CSV          = r"C:\WFM_Scripting\credit_ticket_statistical_tests.csv"
SUMMARIES_FILE     = r"C:\WFM_Scripting\credit_ticket_model_summaries.txt"
XGB_VIZ_CSV        = r"C:\WFM_Scripting\credit_ticket_xgb_feature_importance.csv"

# ------------------- forecast parameters --------------------
FORECAST_HORIZON   = 15
MAX_LAGS           = 12
VAL_LEN_6M         = 6
STAMP              = datetime.now(pytz.timezone("America/Chicago"))

FX_ID_PREFIX = {
    "XGBoost": "xgb_ticket",
    "OLS_Auto_Market": "ols_auto_mkt_ticket",
    "SeasonalNaive3mGR": "seasonal_naive3mgr_ticket",
    "SeasonalNaive3mDiffGR": "seasonal_naive3mdiffgr_ticket"
}

# ------------------- metric & modeling helpers -------------------------
def smape(actual, forecast):
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    denom = (np.abs(actual) + np.abs(forecast)) / 2.0
    denom[denom == 0] = 1.0
    return np.mean(np.abs(actual - forecast) / denom) * 100

def safe_round(x):
    x = 0.0 if (x is None or not np.isfinite(x)) else x
    return int(max(0, round(x)))

def safe_expm1(x, lo=-20.0, hi=20.0):
    arr = np.asarray(x, dtype=float)
    return np.expm1(np.clip(arr, lo, hi))

# --- FIX: The safe_reg_metrics function now correctly includes the mape() definition ---
def safe_reg_metrics(y_true, y_pred):
    def mape(actual, forecast):
        actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
        denom = np.where(actual == 0, 1.0, np.abs(actual))
        return np.mean(np.abs((actual - forecast) / denom)) * 100
    if y_pred is None: return dict(smape=np.nan, mape=np.nan, rmse=np.nan)
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any(): return dict(smape=np.nan, mape=np.nan, rmse=np.nan)
    return dict(smape=smape(y[mask], p[mask]), mape=mape(y[mask], p[mask]), rmse=float(np.sqrt(mean_squared_error(y[mask], p[mask]))))

def recent_3m_growth(ts: pd.Series, pos: int) -> float:
    if pos < 6: return 0.0
    cur3, prev3 = ts.iloc[pos-3:pos].values, ts.iloc[pos-6:pos-3].values
    prev_mean = np.mean(prev3)
    return (np.mean(cur3) / prev_mean - 1.0) if prev_mean > 0 else 0.0

def prior_year_3m_growth(ts: pd.Series, pos: int) -> float:
    if pos < 18: return 0.0
    win_py, prev_py = ts.iloc[pos-15:pos-12].values, ts.iloc[pos-18:pos-15].values
    prev_mean = np.mean(prev_py)
    return (np.mean(win_py) / prev_mean - 1.0) if prev_mean > 0 else 0.0

def perform_statistical_tests(group_data: pd.DataFrame, client_id: str):
    results = {'client_id': client_id}
    target_col = 'total_cases_opened'
    adf_series = group_data[target_col].dropna()
    if len(adf_series) > 3:
        adf_p = adfuller(adf_series)[1]
        results['adf_p_value'], results['is_stationary'] = adf_p, adf_p < 0.05
    return results

# ------------------- data fetching -------------------------
def fetch_market_data(bq: BigQueryManager) -> pd.DataFrame:
    sql = """
    SELECT DATE_TRUNC(Date, MONTH) as date, AVG(UNRATE) as UNRATE, AVG(HSN1F) as HSN1F, AVG(FEDFUNDS) as FEDFUNDS, AVG(MORTGAGE30US) as MORTGAGE30US
    FROM tax_clnt_svcs.fred 
    WHERE Date >= '2023-01-01'
    GROUP BY 1 ORDER BY 1
    """
    df = bq.run_gbq_sql(sql, return_dataframe=True)
    if df.empty: return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').ffill()
    return df

def fetch_credit_workload_drivers(bq: BigQueryManager) -> pd.DataFrame:
    sql = "SELECT MonthOfOrder AS date, TotalOrders FROM tax_clnt_svcs.view_cx_nontax_platforms_workload_drivers WHERE client_id = 'Credit'"
    df = bq.run_gbq_sql(sql, return_dataframe=True)
    if df.empty: return pd.DataFrame(columns=["date", "TotalOrders"])
    df['date'] = pd.to_datetime(df['date'])
    return df.groupby('date', as_index=False)['TotalOrders'].sum()

def fetch_credit_driver_forecasts(bq: BigQueryManager) -> pd.DataFrame:
    sql = "SELECT fx_date AS date, fx_vol AS TotalOrders_fx FROM tax_clnt_svcs.cx_nontax_platforms_workload_drivers_fx WHERE fx_status = 'A' AND client_id = 'Credit'"
    df = bq.run_gbq_sql(sql, return_dataframe=True)
    if df.empty: return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
    return df

# ------------------- MAIN EXECUTION -------------------------
try:
    logger.info("Starting Credit Ticket Forecasting Script (v1.17)...")

    TICKET_QUERY = """
    SELECT DATE_TRUNC(dt, MONTH) AS date, client_id, 'tickets' AS vol_type, SUM(cases_opened) AS total_cases_opened
    FROM tax_clnt_svcs.view_nontax_agg
    WHERE client_id IN ('Credit - Customer Support', 'Credit - Tech Support')
    GROUP BY 1, 2, 3 ORDER BY 2, 1;
    """
    df = bigquery_manager.run_gbq_sql(TICKET_QUERY, return_dataframe=True)
    if df.empty: raise RuntimeError("BQ returned zero rows for tickets")
    df['date'] = pd.to_datetime(df['date'])

    market_df = fetch_market_data(bigquery_manager)
    drv_df = fetch_credit_workload_drivers(bigquery_manager)
    drv_fx_df = fetch_credit_driver_forecasts(bigquery_manager)
    
    if not market_df.empty: df = pd.merge(df, market_df, on='date', how='left')
    if not drv_df.empty: df = pd.merge(df, drv_df, on='date', how='left')
    df = df.sort_values(['client_id', 'date']).ffill()

    df['month'] = df['date'].dt.month
    for lag in range(1, MAX_LAGS + 1):
        df[f'lag_{lag}'] = df.groupby('client_id')['total_cases_opened'].shift(lag)
        if 'TotalOrders' in df.columns:
            df[f'drv_lag_{lag}'] = df.groupby('client_id')['TotalOrders'].shift(lag)

    forecasts, audit_rows, modeling_data, stat_test_results, all_model_summaries, xgb_imp_rows = [], [], [], [], [], []

    for cid, client_df_orig in df.groupby('client_id'):
        logger.info(f"\n{'='*36}\nProcessing Client: {cid}\n{'='*36}")

        stat_test_results.append(perform_statistical_tests(client_df_orig, cid))

        client_df = client_df_orig.copy()
        client_df['trend'] = np.arange(len(client_df))

        ts = client_df.set_index('date')['total_cases_opened'].asfreq('MS')
        feat_all = pd.DataFrame({'y': ts}).join(client_df.set_index('date'))
        
        cols_to_clean = [c for c in feat_all.columns if c not in ['client_id', 'vol_type', 'date']]
        for col in cols_to_clean:
            feat_all[col] = pd.to_numeric(feat_all[col], errors='coerce')
        feat_all.dropna(inplace=True)

        if len(feat_all) < VAL_LEN_6M + 2: 
            logger.warning(f"Skipping {cid} - insufficient data for validation.")
            continue
        
        modeling_data.append(feat_all.assign(client_id=cid))

        train_df, val_df = feat_all.iloc[:-VAL_LEN_6M], feat_all.iloc[-VAL_LEN_6M:]
        y_tr_log = np.log1p(train_df['y'])
        
        model_preds_val, model_meta = {}, {}
        
        candidate_cols = [c for c in feat_all.columns if c not in ['y', 'client_id', 'vol_type', 'date']]
        X_tr, X_val = train_df[candidate_cols].astype(float), val_df[candidate_cols].astype(float)

        # --- Model Competition ---
        # 1. XGBoost
        xgb = XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42).fit(X_tr, y_tr_log)
        model_preds_val['XGBoost'] = safe_expm1(xgb.predict(X_val))
        model_meta['XGBoost'] = {'model': xgb}
        gain = xgb.get_booster().get_score(importance_type='gain')
        xgb_imp_rows.append({'client_id':cid, 'feature_importance': json.dumps(sorted(gain.items(), key=lambda i:i[1], reverse=True))})

        # 2. OLS_Auto_Market (Lasso Selection)
        market_vars = ['UNRATE', 'HSN1F', 'FEDFUNDS', 'MORTGAGE30US']
        scaler = StandardScaler().fit(X_tr)
        lasso = LassoCV(cv=3, random_state=42).fit(scaler.transform(X_tr), y_tr_log)
        selected_feats = [candidate_cols[i] for i in np.where(np.abs(lasso.coef_) > 1e-6)[0]]
        
        if selected_feats and all(f in market_vars for f in selected_feats):
            logger.info(f"Only static market vars selected. Forcing 'trend' into model for {cid}.")
            if 'trend' not in selected_feats: selected_feats.append('trend')

        if selected_feats:
            ols_auto = sm.OLS(y_tr_log, sm.add_constant(train_df[selected_feats].astype(float))).fit()
            all_model_summaries.append(f"--- Client:{cid}, Model:OLS_Auto_Market ---\n{str(ols_auto.summary())}\n")
            model_preds_val['OLS_Auto_Market'] = safe_expm1(ols_auto.predict(sm.add_constant(val_df[selected_feats].astype(float))))
            model_meta['OLS_Auto_Market'] = {'model': ols_auto, 'features': selected_feats}

        # 3. SeasonalNaive3mGR and 4. SeasonalNaive3mDiffGR
        snaive_gr_preds, snaive_diffgr_preds = [], []
        full_ts_for_pos = df[df['client_id'] == cid].set_index('date')['total_cases_opened'].asfreq('MS')
        for d in val_df.index:
            base = ts.get(d - pd.DateOffset(months=12), np.nan)
            pos = full_ts_for_pos.index.get_loc(d)
            r = recent_3m_growth(full_ts_for_pos, pos)
            rp = prior_year_3m_growth(full_ts_for_pos, pos)
            snaive_gr_preds.append(base * (1 + r) if pd.notna(base) else np.nan)
            snaive_diffgr_preds.append(base * (1 + (r - rp)) if pd.notna(base) else np.nan)
        model_preds_val['SeasonalNaive3mGR'] = snaive_gr_preds
        model_preds_val['SeasonalNaive3mDiffGR'] = snaive_diffgr_preds
        model_meta['SeasonalNaive3mGR'], model_meta['SeasonalNaive3mDiffGR'] = {}, {}
        
        # --- Select Winner ---
        scores = {name: safe_reg_metrics(val_df['y'].values, pred)['smape'] for name, pred in model_preds_val.items()}
        winner = min(scores, key=scores.get)
        logger.info(f"✓ WINNER for {cid}: {winner} (SMAPE: {scores[winner]:.2f})")
        for name in model_preds_val: audit_rows.append({'client_id':cid, 'model':name, 'winner_model':winner, 'val_smape_6m':scores.get(name)})

        # --- Recursive Forecasting ---
        hist = deque(ts.tolist(), maxlen=24)
        drv_hist = deque(client_df['TotalOrders'].dropna().tolist(), maxlen=24)
        market_hist_df = client_df[market_vars].dropna()
        last_known_market = market_hist_df.iloc[-1].to_dict() if not market_hist_df.empty else {}
        last_trend = client_df['trend'].iloc[-1]

        driver_fx_lookup = {pd.Timestamp(d):v for d,v in drv_fx_df[['date','TotalOrders_fx']].values} if not drv_fx_df.empty else {}
        
        future_idx = pd.date_range(ts.index[-1], periods=FORECAST_HORIZON+1, freq='MS')[1:]
        
        for i, d in enumerate(future_idx, 1):
            pred = np.nan
            
            if winner in ['XGBoost', 'OLS_Auto_Market']:
                row = {f'lag_{k}':(hist[-k] if len(hist)>=k else hist[-1]) for k in range(1,MAX_LAGS+1)}
                row['month'] = d.month
                row['TotalOrders'] = driver_fx_lookup.get(d, drv_hist[-1] if drv_hist else 0)
                row['trend'] = last_trend + i
                row.update(last_known_market)
                
                x_row_df = pd.DataFrame([row])
                if winner == 'XGBoost':
                    pred = safe_expm1(model_meta['XGBoost']['model'].predict(x_row_df[X_tr.columns])[0])
                else: # OLS_Auto_Market
                    features = model_meta['OLS_Auto_Market']['features']
                    raw_pred = model_meta['OLS_Auto_Market']['model'].predict(sm.add_constant(x_row_df[features]))[0]
                    pred = np.expm1(raw_pred)
            
            elif winner in ['SeasonalNaive3mGR', 'SeasonalNaive3mDiffGR']:
                base = hist[-12] if len(hist)>=12 else hist[-1]
                temp_series = pd.Series(list(hist))
                pos = len(temp_series)
                r = recent_3m_growth(temp_series, pos)
                if winner == 'SeasonalNaive3mGR':
                    pred = base * (1 + r)
                else: # SeasonalNaive3mDiffGR
                    rp = prior_year_3m_growth(temp_series, pos)
                    pred = base * (1 + (r - rp))

            hist.append(pred)
            # Need to update drv_hist for drv_lag features in the next iteration
            if 'TotalOrders' in row: drv_hist.append(row['TotalOrders'])
            
            forecasts.append({'fx_date':d,'client_id':cid,'vol_type':'tickets','fx_vol':safe_round(pred),'fx_id':f"{FX_ID_PREFIX[winner]}_{STAMP:%Y%m%d}",'fx_status':"A",'load_ts':STAMP})

    if forecasts:
        fx_df = pd.DataFrame(forecasts).sort_values(['client_id','fx_date'])
        
        fx_df.to_csv(LOCAL_CSV, index=False)
        pd.DataFrame(audit_rows).sort_values(['client_id']).to_csv(AUDIT_CSV, index=False)
        if all_model_summaries:
            with open(SUMMARIES_FILE, 'w') as f: f.write("\n\n".join(all_model_summaries))
        if stat_test_results:
            pd.DataFrame(stat_test_results).to_csv(STATS_CSV, index=False)
        if modeling_data:
            pd.concat(modeling_data).to_csv(MODELING_DATA_CSV, index=False)
        if xgb_imp_rows:
            pd.DataFrame(xgb_imp_rows).to_csv(XGB_VIZ_CSV, index=False)
            
        logger.info(f"✓ All local audit files saved.")
        logger.info("✓ Credit ticket forecasting completed successfully (local only).")
    else:
        logger.warning("No forecasts were generated.")

except Exception as exc:
    email_manager.handle_error("Credit Ticket Forecasting Script Failure", exc, is_test=True)
