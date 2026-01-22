"""
Workload Forecasting Script (Rpt 288) - Corrected + Heavily Commented

Root cause of your crash:
- `history_df["y"]` ended up as pandas nullable integer dtype `Int64`
- During recursive forecasting you assign a float prediction (e.g., 59286.28)
- Pandas raises: TypeError: Invalid value '...' for dtype Int64

Fix:
- Force ALL target series / columns that will receive predictions to be float early
- Ensure the forecast “new_row” includes y as float NaN (not an integer)
- Avoid any operation that accidentally casts y back to Int64

Notes:
- I kept your overall logic intact (models, evaluation, recursion, outputs)
- Added detailed in-code comments where it matters most
"""

import os
import sys
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytz
from scipy.stats import pearsonr

# --- 1. System Path & Imports -------------------------------------------------
sys.path.append(r"C:\WFM_Scripting\Automation")
from scripthelper import Config, BigQueryManager, EmailManager, GeneralFuncs  # Logger comes from config.logger

# Third-party ML/Stat imports
try:
    import statsmodels.api as sm
    from xgboost import XGBRegressor
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError as e:
    print(f"Warning: ML libraries missing ({e}). Script may fail.")


# --- 2. Global Initialization -------------------------------------------------
def initialize_globals():
    """
    Initialize shared, script-wide objects and constants.

    Keep everything centralized so the script behaves consistently when executed
    by scheduler / task runner, and so we can swap between test/prod modes.
    """
    global config, logger, bigquery_manager, email_manager, general_funcs
    global rpt_id, is_test

    rpt_id = 288
    config = Config(rpt_id=rpt_id)
    logger = config.logger  # provided by your scripthelper Config
    bigquery_manager = BigQueryManager(config)
    email_manager = EmailManager(config)
    general_funcs = GeneralFuncs(config)

    # Force Test Mode (scripthelper EmailManager will usually respect this flag)
    is_test = True

    # --- Constants & SQL ------------------------------------------------------
    global GBQ_MAIN_QRY, GBQ_FRED_HIST_QRY, GBQ_FRED_FX_QRY, GBQ_CALENDAR_FILE
    global LOCAL_CSV, AUDIT_CSV, STATS_CSV, XGB_VIZ_CSV, MODELING_DATA_CSV, SUMMARIES_FILE
    global FORECAST_HORIZON, TEST_LEN, LAGS, STAMP, FX_ID_PREFIX
    global FRED_COLS, CALENDAR_COLS, ALL_EXOG_COLS
    global BACKCAST_MODE, BACKCAST_START_DATE

    # --- BACKCAST SETTINGS ----------------------------------------------------
    BACKCAST_MODE = False
    BACKCAST_START_DATE = "2025-01-01"

    # --- Queries --------------------------------------------------------------
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

    GBQ_FRED_FX_QRY = """
    SELECT
        Date, UNRATE, HSN1F, FEDFUNDS, MORTGAGE30US, Forecast_Date
    FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.fred_fx`
    QUALIFY ROW_NUMBER() OVER(PARTITION BY Date ORDER BY Forecast_Date DESC) = 1
    ORDER BY Date
    """

    # Calendar SQL File Path (local file containing SQL)
    GBQ_CALENDAR_FILE = r"C:\WFM_Scripting\Forecasting\GBQ - Calendar days.sql"

    # --- Local Outputs --------------------------------------------------------
    suffix = "_BACKCAST" if BACKCAST_MODE else ""
    LOCAL_CSV = fr"C:\WFM_Scripting\forecast_results{suffix}.csv"
    AUDIT_CSV = fr"C:\WFM_Scripting\model_eval_debug{suffix}.csv"
    STATS_CSV = fr"C:\WFM_Scripting\statistical_tests{suffix}.csv"
    XGB_VIZ_CSV = fr"C:\WFM_Scripting\xgb_feature_importance{suffix}.csv"
    MODELING_DATA_CSV = fr"C:\WFM_Scripting\modeling_data{suffix}.csv"
    SUMMARIES_FILE = fr"C:\WFM_Scripting\model_summaries{suffix}.txt"

    # Forecast parameters
    FORECAST_HORIZON = 15
    TEST_LEN = 6
    LAGS = (3, 6, 12)

    # Feature Columns
    FRED_COLS = ["UNRATE", "HSN1F", "FEDFUNDS", "MORTGAGE30US"]
    CALENDAR_COLS = ["total_days", "weekday_count", "weekend_day_count", "holiday_count", "business_day_count"]
    ALL_EXOG_COLS = FRED_COLS + CALENDAR_COLS

    # Timestamp / TZ
    STAMP_TZ = pytz.timezone("America/Chicago")
    STAMP = datetime.now(STAMP_TZ)

    # Forecast ID prefixes (used to create fx_id tags)
    FX_ID_PREFIX = {
        "SeasonalNaive": "snaive_workload",
        "SeasonalNaiveGR": "snaive_gr_workload",
        "Native3m": "native3m_workload",
        "MLR": "mlr_workload",
        "XGBoost": "xgb_workload",
        "SARIMA": "sarima_workload",
    }

    warnings.filterwarnings("ignore")


# ------------------- Metrics & Helpers ---------------------------------------
def smape(a, f) -> float:
    """Symmetric MAPE (percent). Handles zeros safely."""
    a, f = np.asarray(a, float), np.asarray(f, float)
    denom = (np.abs(a) + np.abs(f)) / 2.0
    out = np.where(denom == 0.0, 0.0, np.abs(a - f) / denom)
    return float(np.mean(out) * 100.0)


def mape(a, f) -> float:
    """MAPE (percent). Avoid divide-by-zero by substituting 1 when a==0."""
    a, f = np.asarray(a, float), np.asarray(f, float)
    return float(np.mean(np.abs((a - f) / np.where(a == 0, 1, a))) * 100.0)


def rmse(a, f) -> float:
    """RMSE (same units as target)."""
    a, f = np.asarray(a, float), np.asarray(f, float)
    return float(np.sqrt(np.mean((a - f) ** 2)))


def perform_statistical_tests(df: pd.DataFrame, cid, prod, target_col: str = "y") -> Dict:
    """
    Runs:
      1) ADF test (stationarity) on target series
      2) Pearson correlations between target and candidate features (lags + exog)

    Returns a single dict for convenient appending into a list then writing CSV.
    """
    results = {"client_id": cid, "product": prod}

    # 1) ADF Stationarity ------------------------------------------------------
    series = df[target_col].dropna()
    if len(series) > 8:
        try:
            adf_res = adfuller(series)
            results["adf_p_value"] = adf_res[1]
            results["is_stationary"] = adf_res[1] < 0.05
        except Exception:
            results["adf_p_value"] = np.nan
            results["is_stationary"] = np.nan
    else:
        results["adf_p_value"] = np.nan
        results["is_stationary"] = np.nan

    # 2) Correlations ----------------------------------------------------------
    correlations = {}
    potential_cols = [c for c in df.columns if "lag" in c or c in ALL_EXOG_COLS]

    for col in potential_cols:
        if col not in df.columns:
            continue

        mask = df[target_col].notna() & df[col].notna()
        if mask.sum() > 4:
            try:
                corr, p_val = pearsonr(df.loc[mask, target_col], df.loc[mask, col])
                if not np.isnan(corr):
                    correlations[col] = {"r": corr, "p": p_val}
            except Exception:
                pass

    top_corr = sorted(correlations.items(), key=lambda x: abs(x[1]["r"]), reverse=True)[:5]
    results["top_correlations"] = json.dumps({k: round(v["r"], 3) for k, v in top_corr})
    return results


# ------------------- Data Prep ------------------------------------------------
def prepare_exog_data(df_hist: pd.DataFrame, df_fx: pd.DataFrame, df_cal: pd.DataFrame) -> pd.DataFrame:
    """
    Build a master monthly exogenous dataset:
      - FRED: historical + forecasted
      - Calendar: counts per month (business days, holidays, etc.)
    Returns a DataFrame indexed by month-start ("MS") with ALL_EXOG_COLS.

    Important corrections:
      - Only rename Date -> date (do NOT clobber Forecast_Date)
      - Ensure values are numeric and filled appropriately
    """
    # --- 1) Clean FRED --------------------------------------------------------
    df_hist = df_hist.rename(columns={"Date": "date"})
    df_fx = df_fx.rename(columns={"Date": "date"})

    df_hist["date"] = pd.to_datetime(df_hist["date"])
    df_fx["date"] = pd.to_datetime(df_fx["date"])

    # Index for time joins
    df_hist = df_hist.set_index("date").sort_index()
    df_fx = df_fx.set_index("date").sort_index()

    # Combine:
    # - keep historical where present
    # - fill holes/future from forecast table
    fred_full = df_hist.combine_first(df_fx)

    # --- 2) Clean Calendar ----------------------------------------------------
    df_cal = df_cal.rename(columns={"month_start": "date"})
    df_cal["date"] = pd.to_datetime(df_cal["date"])
    df_cal = df_cal.set_index("date").sort_index()

    # --- 3) Merge All ---------------------------------------------------------
    exog_master = fred_full.join(df_cal, how="outer")

    # Fill FRED holes:
    # - FRED should be continuous month-to-month for forecasting stability
    for c in FRED_COLS:
        if c in exog_master.columns:
            exog_master[c] = pd.to_numeric(exog_master[c], errors="coerce").ffill().bfill()
        else:
            exog_master[c] = 0.0

    # Fill Calendar holes:
    # - calendar days should not need backfill generally, but protect anyway
    for c in CALENDAR_COLS:
        if c in exog_master.columns:
            exog_master[c] = pd.to_numeric(exog_master[c], errors="coerce").ffill()
        else:
            exog_master[c] = 0.0

    # Ensure numeric float output (important for downstream ML / stats)
    out = exog_master[ALL_EXOG_COLS].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out


def add_lags(df: pd.DataFrame, target_col: str = "y", lag_cols: List[str] = None) -> pd.DataFrame:
    """
    Create lag features.

    - Always creates target lags: lag_3, lag_6, lag_12
    - Optionally creates lagged versions of columns in `lag_cols` (FRED only)

    NOTE:
      Calendar vars are typically used "concurrently" (same month), so they are NOT lagged here.
    """
    if lag_cols is None:
        lag_cols = []

    out = df.copy()

    # --- Target lags ----------------------------------------------------------
    for L in LAGS:
        out[f"lag_{L}"] = out[target_col].shift(L)

    # --- Exog lags (FRED) -----------------------------------------------------
    for col in lag_cols:
        if col not in out.columns:
            continue
        for L in LAGS:
            out[f"{col}_lag_{L}"] = out[col].shift(L)

    return out


# ------------------- Naive Model Helpers -------------------------------------
def native_3m_growth_series(ts: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """
    A seasonal (YoY) baseline with a 3-month growth adjustment.

    For each date d:
      base = y(d-12)
      growth_factor = mean(last 3 months) / mean(last-year same 3 months)
      prediction = base * growth_factor
    """
    out = []
    s12 = ts.shift(12)

    for d in idx:
        base = s12.get(d, np.nan)
        if not np.isfinite(base) or d not in ts.index:
            out.append(np.nan)
            continue

        pos = ts.index.get_loc(d)

        # Need enough history to compute both 3m windows safely
        if pos < 18:
            out.append(base)
            continue

        cur_3m = ts.iloc[pos - 3 : pos].mean()
        py_3m = ts.iloc[pos - 15 : pos - 12].mean()
        growth_factor = (cur_3m / py_3m) if py_3m > 0 else 1.0
        out.append(max(0.0, base * growth_factor))

    return pd.Series(out, index=idx, dtype="float64")


def seasonal_naive_gr_series(ts: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """
    Seasonal naive (YoY) + short-term growth rate.

    For each date d:
      base = y(d-12)
      r = (mean(last 3 months) / mean(prev 3 months)) - 1
      prediction = base * (1 + r)
    """
    out = []
    s12 = ts.shift(12)

    for d in idx:
        base = s12.get(d, np.nan)
        if not np.isfinite(base) or d not in ts.index:
            out.append(np.nan)
            continue

        pos = ts.index.get_loc(d)
        if pos < 6:
            r = 0.0
        else:
            prev = ts.iloc[pos - 6 : pos - 3].mean()
            curr = ts.iloc[pos - 3 : pos].mean()
            r = (curr / prev - 1.0) if prev > 0 else 0.0

        out.append(max(0.0, base * (1.0 + r)))

    return pd.Series(out, index=idx, dtype="float64")


# ------------------- Core Process --------------------------------------------
def run_forecast_logic():
    mode_msg = f"BACKCAST MODE (Cutoff: {BACKCAST_START_DATE})" if BACKCAST_MODE else "PRODUCTION MODE"
    logger.info(f"Starting Workload Forecasting - {mode_msg}")

    # 1) Fetch Data ------------------------------------------------------------
    logger.info("Fetching Data (Main, FRED, Calendar)...")
    df_main = bigquery_manager.run_gbq_sql(GBQ_MAIN_QRY, return_dataframe=True)
    df_fred_h = bigquery_manager.run_gbq_sql(GBQ_FRED_HIST_QRY, return_dataframe=True)
    df_fred_f = bigquery_manager.run_gbq_sql(GBQ_FRED_FX_QRY, return_dataframe=True)

    # Calendar query is stored as a local .sql file containing SQL text.
    # Some GBQ wrappers accept the file path, some need the SQL string.
    try:
        with open(GBQ_CALENDAR_FILE, "r", encoding="utf-8") as f:
            cal_sql = f.read()
        df_calendar = bigquery_manager.run_gbq_sql(cal_sql, return_dataframe=True)
    except Exception as e:
        logger.warning(f"Could not read local SQL file contents; passing path to runner. Details: {e}")
        df_calendar = bigquery_manager.run_gbq_sql(GBQ_CALENDAR_FILE, return_dataframe=True)

    if df_main.empty:
        raise ValueError("Main dataset empty.")
    if df_calendar.empty:
        raise ValueError("Calendar dataset empty.")

    # 2) Build Exog Master -----------------------------------------------------
    exog_master = prepare_exog_data(df_fred_h, df_fred_f, df_calendar)

    # Ensure main date typed
    df_main["date"] = pd.to_datetime(df_main["date"])

    # Backcast: truncate history at cutoff
    if BACKCAST_MODE:
        cutoff = pd.to_datetime(BACKCAST_START_DATE)
        orig_len = len(df_main)
        df_main = df_main[df_main["date"] < cutoff]
        logger.info(f"Backcast Truncation: {orig_len} -> {len(df_main)} rows (Data < {cutoff.date()})")

    df_main = df_main.sort_values(["client_id", "product", "date"])

    # Containers for outputs
    forecasts = []
    audit_rows = []
    stat_test_results = []
    xgb_imp_rows = []
    all_modeling_data = []
    model_summaries = []

    # 3) Group Loop ------------------------------------------------------------
    for (cid, prod), g in df_main.groupby(["client_id", "product"]):
        # Build a clean monthly time series.
        # IMPORTANT: force float dtype early to prevent Int64 casting issues later.
        ts = (
            g.set_index("date")["target_volume"]
            .asfreq("MS")  # monthly start frequency
            .fillna(0.0)   # fill missing months with 0
            .astype("float64")  # <-- critical: keep target as float
        )

        # Merge ts + exog. ffill/bfill to fill exog gaps.
        df_merged = pd.DataFrame({"y": ts}).join(exog_master, how="left").ffill().bfill()

        # CRITICAL: Ensure y remains float even after joins/fills
        df_merged["y"] = pd.to_numeric(df_merged["y"], errors="coerce").astype("float64")

        # Add lag features (lags for y + lags for FRED features)
        feat_df = add_lags(df_merged, target_col="y", lag_cols=FRED_COLS).dropna()

        if feat_df.shape[0] <= (TEST_LEN + 6):
            logger.info(f"Skipping {cid}-{prod}: Insufficient history.")
            continue

        # A) Statistical Tests -------------------------------------------------
        stat_res = perform_statistical_tests(feat_df, cid, prod)
        stat_test_results.append(stat_res)

        # B) Save Modeling Data ------------------------------------------------
        dump_df = feat_df.copy()
        dump_df["client_id"] = cid
        dump_df["product"] = prod
        all_modeling_data.append(dump_df)

        # Split (train vs validation)
        val_idx = feat_df.index[-TEST_LEN:]
        train_idx = feat_df.index[:-TEST_LEN]
        df_train = feat_df.loc[train_idx]
        df_val = feat_df.loc[val_idx]
        ts_full = df_merged["y"]

        # Features:
        # - all lag_* columns (includes target lags and fred_lag_* columns)
        # - calendar cols concurrently
        lag_feats = [c for c in feat_df.columns if "lag" in c]
        cal_feats = [c for c in CALENDAR_COLS if c in feat_df.columns]
        feature_cols = lag_feats + cal_feats

        preds = {}
        models_cache = {}

        # C) Naive Models ------------------------------------------------------
        preds["SeasonalNaive"] = ts_full.shift(12).reindex(val_idx).astype("float64")
        preds["SeasonalNaiveGR"] = seasonal_naive_gr_series(ts_full, val_idx)
        preds["Native3m"] = native_3m_growth_series(ts_full, val_idx)

        # D) XGBoost -----------------------------------------------------------
        try:
            xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
            xgb.fit(df_train[feature_cols], df_train["y"])
            p_xgb = xgb.predict(df_val[feature_cols])
            preds["XGBoost"] = pd.Series(np.maximum(0.0, p_xgb), index=val_idx, dtype="float64")
            models_cache["XGBoost"] = xgb

            gain = xgb.get_booster().get_score(importance_type="gain")
            imp_sorted = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)
            xgb_imp_rows.append(
                {"client_id": cid, "product": prod, "importance_json": json.dumps(imp_sorted)}
            )
        except Exception as e:
            logger.warning(f"XGB failed {cid}-{prod}: {e}")

        # E) MLR (Statsmodels OLS) ---------------------------------------------
        try:
            X_tr_ols = sm.add_constant(df_train[feature_cols])
            X_val_ols = sm.add_constant(df_val[feature_cols], has_constant="add")

            ols = sm.OLS(df_train["y"], X_tr_ols).fit()
            p_mlr = ols.predict(X_val_ols)
            preds["MLR"] = pd.Series(np.maximum(0.0, p_mlr), index=val_idx, dtype="float64")
            models_cache["MLR"] = ols

            summ_text = (
                f"\n--- Client: {cid} | Prod: {prod} | Model: MLR (OLS) ---\n"
                f"{ols.summary().as_text()}\n"
            )
            model_summaries.append(summ_text)
        except Exception as e:
            logger.warning(f"MLR Failed for {cid}-{prod}: {e}")

        # F) SARIMA ------------------------------------------------------------
        try:
            sar = SARIMAX(
                ts_full.loc[train_idx],
                order=(1, 1, 1),
                seasonal_order=(0, 1, 0, 12),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            sar_res = sar.fit(disp=False)
            p_sar = sar_res.forecast(steps=len(val_idx))
            preds["SARIMA"] = pd.Series(np.maximum(0.0, p_sar.values), index=val_idx, dtype="float64")
            models_cache["SARIMA"] = sar_res
        except Exception:
            pass

        # G) Evaluate & Select Winner -----------------------------------------
        smapes = {}
        y_true = ts_full.loc[val_idx].astype("float64")

        for m, pser in preds.items():
            if pser is None or pser.isna().all():
                continue

            pser = pser.astype("float64")
            s = smape(y_true, pser)
            smapes[m] = s

            audit_rows.append(
                {
                    "client_id": cid,
                    "product": prod,
                    "model": m,
                    "SMAPE": s,
                    "MAPE": mape(y_true, pser),
                    "RMSE": rmse(y_true, pser),
                }
            )

        if not smapes:
            continue

        # Tie-breaker priority list (lower index preferred)
        priority = ["Native3m", "MLR", "XGBoost", "SeasonalNaiveGR", "SeasonalNaive", "SARIMA"]
        best_model = min(smapes, key=lambda k: (smapes[k], priority.index(k) if k in priority else 99))
        logger.info(f"· {cid} [{prod}] Winner: {best_model} ({smapes[best_model]:.2f}%)")

        # H) Recursive Forecast (15 mo) ----------------------------------------
        last_date = ts_full.index.max()
        future_dates = pd.date_range(last_date, periods=FORECAST_HORIZON + 1, freq="MS")[1:]

        # history_df stores the evolving timeline including future months.
        # CRITICAL: keep y float so we can assign float predictions.
        history_df = df_merged.copy()
        history_df["y"] = pd.to_numeric(history_df["y"], errors="coerce").astype("float64")

        for fd in future_dates:
            # Build a new future row with the same columns as history_df.
            # Start with NaNs; explicitly set y as float NaN to preserve dtype.
            new_row = pd.DataFrame(index=[fd], columns=history_df.columns)
            new_row["y"] = np.nan  # ensures y column remains float-compatible

            # 1) Fill exog values from master (preferred), else carry-forward last known.
            if fd in exog_master.index:
                for c in ALL_EXOG_COLS:
                    new_row.at[fd, c] = float(exog_master.at[fd, c])
            else:
                # If exog master doesn't have this future month, carry forward last row's exog.
                for c in ALL_EXOG_COLS:
                    new_row.at[fd, c] = float(history_df.iloc[-1][c])

            # Append to history
            history_df = pd.concat([history_df, new_row], axis=0)

            # CRITICAL: Ensure after concat y is still float (concat can sometimes coerce)
            history_df["y"] = pd.to_numeric(history_df["y"], errors="coerce").astype("float64")

            # 2) Build lag features for the tail window (enough rows to compute max lag).
            tail_df = history_df.iloc[-(max(LAGS) + 5) :].copy()
            feat_tail = add_lags(tail_df, target_col="y", lag_cols=FRED_COLS)

            # Pull the feature row for fd
            X_pred = feat_tail.iloc[[-1]][feature_cols]

            # 3) Predict next value based on winning model
            pred_val = 0.0

            if best_model == "MLR" and "MLR" in models_cache:
                X_pred_ols = sm.add_constant(X_pred, has_constant="add")
                pred_val = float(models_cache["MLR"].predict(X_pred_ols).iloc[0])

            elif best_model == "XGBoost" and "XGBoost" in models_cache:
                pred_val = float(models_cache["XGBoost"].predict(X_pred)[0])

            elif best_model == "SARIMA" and "SARIMA" in models_cache:
                # SARIMA model here was trained on train_idx only, so a true recursive SARIMA
                # would require refit/update. Keeping your original behavior (no recursive SARIMA).
                # Fallback to seasonal naive if needed.
                pred_val = float(history_df.iloc[-13]["y"]) if len(history_df) >= 13 else 0.0

            elif best_model == "Native3m":
                if len(history_df) >= 16:
                    base = float(history_df.iloc[-13]["y"])
                    curr_3m = float(history_df.iloc[-4:-1]["y"].mean())
                    prior_3m = float(history_df.iloc[-16:-13]["y"].mean())
                    gf = (curr_3m / prior_3m) if prior_3m > 0 else 1.0
                    pred_val = base * gf
                else:
                    pred_val = float(history_df.iloc[-2]["y"])

            elif best_model == "SeasonalNaiveGR":
                if len(history_df) >= 13:
                    base = float(history_df.iloc[-13]["y"])
                    curr_3m = float(history_df.iloc[-4:-1]["y"].mean())
                    prev_3m = float(history_df.iloc[-7:-4]["y"].mean())
                    r = (curr_3m / prev_3m - 1.0) if prev_3m > 0 else 0.0
                    pred_val = base * (1.0 + r)
                else:
                    pred_val = float(history_df.iloc[-2]["y"])

            else:
                # SeasonalNaive default
                pred_val = float(history_df.iloc[-13]["y"]) if len(history_df) >= 13 else 0.0

            # Enforce non-negative and float
            pred_val = float(max(0.0, pred_val))

            # Assign prediction (THIS is where you crashed before due to Int64 dtype)
            history_df.at[fd, "y"] = pred_val

            fx_tag = f"{FX_ID_PREFIX.get(best_model, 'other')}_{STAMP:%Y%m%d}"
            forecasts.append(
                {
                    "fx_date": fd.strftime("%Y-%m-01"),
                    "client_id": cid,
                    "product": prod,
                    "fx_vol": int(round(pred_val)),
                    "fx_id": fx_tag,
                    "fx_status": "A",
                    "load_ts": STAMP.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        audit_rows.append(
            {
                "client_id": cid,
                "product": prod,
                "model": f"{best_model}_WINNER",
                "SMAPE": smapes[best_model],
                "MAPE": 0,
                "RMSE": 0,
            }
        )

    # 4) Save Outputs ----------------------------------------------------------
    if forecasts:
        fx_df = pd.DataFrame(forecasts)
        logger.info(f"Writing {len(fx_df)} forecast rows locally to {LOCAL_CSV}...")
        fx_df.to_csv(LOCAL_CSV, index=False)

        if audit_rows:
            pd.DataFrame(audit_rows).to_csv(AUDIT_CSV, index=False)
        if stat_test_results:
            pd.DataFrame(stat_test_results).to_csv(STATS_CSV, index=False)
        if xgb_imp_rows:
            pd.DataFrame(xgb_imp_rows).to_csv(XGB_VIZ_CSV, index=False)
        if all_modeling_data:
            pd.concat(all_modeling_data).to_csv(MODELING_DATA_CSV, index=True)
        if model_summaries:
            with open(SUMMARIES_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(model_summaries))

        logger.info(r"✓ All files saved to C:\WFM_Scripting\")
    else:
        logger.warning("No forecasts generated.")


# ------------------- Main Execution ------------------------------------------
if __name__ == "__main__":
    initialize_globals()
    try:
        run_forecast_logic()
    except Exception as exc:
        # Always log the error first
        logger.error(f"Script failed: {exc}")

        # scripthelper EmailManager.handle_error raises again (as your trace shows),
        # so wrap it to avoid masking the original stack if desired.
        if "email_manager" in globals():
            try:
                email_manager.handle_error("Forecast Script Failed", exc, is_test=is_test)
            except Exception:
                # Avoid infinite exception chains; the original error is already logged.
                pass

        # Re-raise so scheduler sees failure status
        raise
