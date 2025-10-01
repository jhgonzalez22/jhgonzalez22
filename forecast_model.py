# ───────────────── workload_forecast_multi_model.py ─────────────────
"""
CX Platforms – Workload Drivers Forecast (TotalOrders)

Pipeline (PC/GBQ):
  1) Pull monthly client time series (TotalOrders only) and FRED economic data from BigQuery.
  2) Engineer features (lags restricted to 3/6/12, rolling(3), seasonal sin/cos + month dummies, FRED).
  3) Train & evaluate candidate models on the last 3 months (validation) using SMAPE.
  4) Select the best model per client and produce a 15-month recursive forecast (uses future FRED via forward fill).
  5) Append forecasts to GBQ table, inactivate older duplicates, save local CSVs, and log success.

Candidate models:
  • XGBoost                — Tree-based regressor on lag/seasonal/rolling/economic features.
  • Ridge (log-space)      — Linear model on selected features; predicts log(y) then expm1.
  • Lasso (log-space)      — Sparse linear model; predicts log(y) then expm1.
  • WeightedLagBlend       — Convex blend of lag_3, lag_6, lag_12 (grid search; min seasonal weight on lag12).
  • SeasonalNaive3mGR      — Seasonal naive (t−12) adjusted by 3-month growth vs prior-year same window.

Notes:
  • Uses ONLY TotalOrders as the target.
  • Validation window = 3 months (TEST_LEN = 3).
  • Forecast horizon = 15 months.
  • Output FX schema: fx_date, client_id, fx_vol, fx_id, fx_status, load_ts
      - fx_id: <model_prefix>_<YYYYMMDD>  (underscores)
      - fx_status: "forecast" for newest rows; older rows are set to "inactive".
"""

# ------------------- standard imports -----------------------
import os, sys, warnings, pytz
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")  # Keep logs tidy

# ------------------- helper-package path --------------------
# Ensure the path to your scripthelper module is correct for your PC
sys.path.append(r'C:\WFM_Scripting\Automation')
from scripthelper import Config, Logger, BigQueryManager, EmailManager

# ------------------- initialise scripthelper ---------------
config            = Config(rpt_id=288)
logger            = Logger(config)
email_manager     = EmailManager(config)
bigquery_manager  = BigQueryManager(config)

# ------------------- file / table paths --------------------
# Pull WD view (TotalOrders) and FRED macro data
GBQ_VIEW_QUERY = """
SELECT
  MonthOfOrder AS date,
  client_id,
  TotalOrders  AS target_volume
FROM `tax_clnt_svcs.view_cx_nontax_platforms_workload_drivers`
WHERE MonthOfOrder < DATE_TRUNC(CURRENT_DATE(), MONTH)
"""
FRED_QUERY = """
SELECT
  Date AS date,
  UNRATE,
  HSN1F,
  FEDFUNDS,
  MORTGAGE30US
FROM `tax_clnt_svcs.fred`
"""
DEST_TABLE = "tax_clnt_svcs.cx_nontax_platforms_workload_drivers_fx"

# --- Local CSV outputs (Windows paths) ---
LOCAL_CSV = r"C:\WFM_Scripting\cx_nontax_workload_drivers_fx_results.csv"        # forecasts (FX schema)
AUDIT_CSV = r"C:\WFM_Scripting\cx_nontax_workload_drivers_fx_model_eval.csv"     # per-client per-model metrics

# ------------------- forecast parameters --------------------
FORECAST_HORIZON = 15     # months ahead
TEST_LEN         = 3      # last 3 months for validation
LAGS             = (3,6,12)  # restrict to 3/6/12 only everywhere

# WeightedLagBlend search constraints
GRID_STEP        = 0.05   # grid step for weights
MIN_SEASONAL_W   = 0.15   # minimum weight on lag_12 to enforce seasonality

# Timestamp (America/Chicago) for IDs and load_ts
STAMP_TZ = pytz.timezone("America/Chicago")
STAMP    = datetime.now(STAMP_TZ)

# Prefixes for fx_id based on winning model (underscored)
FX_ID_PREFIX = {
    "XGBoost":           "xgb_workload",
    "Ridge":             "ridge_workload",
    "Lasso":             "lasso_workload",
    "WeightedLagBlend":  "wblend_workload",
    "SeasonalNaive3mGR": "snaive3mgr_workload",
}

# ------------------- metric + helper functions -------------------------
def mape(actual, forecast):
    """Mean Absolute Percentage Error (%), safe for zeros in actuals."""
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    return float(np.mean(np.abs((a - f) / np.where(a == 0, 1, a))) * 100.0)

def smape(actual, forecast):
    """Symmetric MAPE (%), robust to zeros and scale."""
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    denom = (np.abs(a) + np.abs(f)) / 2.0
    out = np.where(denom == 0.0, 0.0, np.abs(a - f) / denom)
    return float(np.mean(out) * 100.0)

def rmse(actual, forecast):
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    return float(np.sqrt(np.mean((a - f) ** 2)))

def safe_expm1(x, lo=-20.0, hi=20.0):
    """Clips linear output before expm1 to avoid overflow/underflow."""
    return float(np.expm1(np.clip(x, lo, hi)))

# ------------------- feature engineering -------------------
def build_supervised_frame(ts: pd.Series, econ_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a supervised learning frame aligned to the time series index:
      y (target), lags lag_3/lag_6/lag_12, rolling mean/std over last 3 (based on y.shift(1)),
      deterministic seasonality (sin/cos + month dummies),
      and FRED macro variables merged by date.
    """
    feat = pd.DataFrame({"y": ts})
    # Lags (restricted)
    for L in LAGS:
        feat[f"lag_{L}"] = feat["y"].shift(L)

    # Rolling from prior observations
    lag1 = feat["y"].shift(1)
    feat["rolling_mean_3"] = lag1.rolling(3).mean()
    feat["rolling_std_3"]  = lag1.rolling(3).std()

    # Seasonal signals
    months = feat.index.month
    feat["sin_month"] = np.sin(2 * np.pi * months / 12.0)
    feat["cos_month"] = np.cos(2 * np.pi * months / 12.0)
    for mo in range(1, 13):
        feat[f"mo_{mo}"] = (months == mo).astype(float)

    # Merge FRED (econ_df index is date)
    feat = feat.join(econ_df, how="left")

    # Drop initial rows with NaN due to lags/rolls or missing econ after join
    feat.dropna(inplace=True)
    return feat

def extend_fred_for_horizon(fred_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Forward-fill future months with the last known FRED values.
    Assumes monthly data; returns a Date-indexed DataFrame covering history + horizon.
    """
    fred_df = fred_df.copy()
    fred_df["date"] = pd.to_datetime(fred_df["date"])
    fred_df = fred_df.sort_values("date").set_index("date")

    # Ensure monthly MS index across historical span
    first = fred_df.index.min().to_period("M").to_timestamp("MS")
    last  = fred_df.index.max().to_period("M").to_timestamp("MS")
    full_hist_index = pd.date_range(first, last, freq="MS")
    fred_df = fred_df.reindex(full_hist_index)

    # Forward-fill history; backfill leading if necessary
    fred_df = fred_df.ffill().bfill()

    # Build future index and repeat the last known row for each future month
    future_index = pd.date_range(last + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    if len(future_index):
        last_row = fred_df.iloc[[-1]].copy()
        future_ext = pd.concat([last_row] * len(future_index), ignore_index=True)
        future_ext.index = future_index
        fred_df = pd.concat([fred_df, future_ext], axis=0)

    return fred_df

# ------------------- seasonal naive with growth ------------
def seasonal_naive_3mgr_val(ts: pd.Series, val_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Validation-time SeasonalNaive3mGR:
    yhat_t = y[t-12] * (1 + r), where r ~ (mean(last 3 months) / mean(same 3 months prior year) - 1).
    """
    s12 = ts.shift(12)
    out = []
    for d in val_index:
        base = s12.get(d, np.nan)
        pred_d = np.nan
        try:
            pos = ts.index.get_loc(d)
            if np.isfinite(base) and pos >= 15:
                cur3  = ts.iloc[pos-3:pos].values.astype(float)
                past3 = ts.iloc[pos-15:pos-12].values.astype(float)
                past_mean = float(np.mean(past3))
                r = (float(np.mean(cur3)) / past_mean - 1.0) if past_mean > 0 else 0.0
                pred_d = float(base) * (1.0 + r)
        except Exception:
            pred_d = np.nan
        out.append(pred_d)
    return np.array(out, dtype=float)

def seasonal_naive_3mgr_forecast(history: deque) -> float:
    """
    Forecast-time SeasonalNaive3mGR using current history deque (latest at end).
    """
    n = len(history)
    base = history[-12] if n >= 12 else history[-1]
    r = 0.0
    if n >= 15:
        cur3  = list(history)[-3:]
        past3 = list(history)[-15:-12]
        past_mean = float(np.mean(past3))
        r = (float(np.mean(cur3)) / past_mean - 1.0) if past_mean > 0 else 0.0
    return max(0.0, float(base) * (1.0 + r))

# ------------------- weighted lag blend (3/6/12) -----------
def weight_grid_3_6_12(step=GRID_STEP, min_w12=MIN_SEASONAL_W):
    """Generate (w3, w6, w12) with w>=0, sum=1, and w12 >= min_w12."""
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    for w3 in vals:
        for w6 in vals:
            w12 = 1.0 - (w3 + w6)
            if w12 < -1e-12:
                continue
            w12 = max(0.0, w12)
            if w12 + 1e-12 >= min_w12:
                yield (float(w3), float(w6), float(w12))

# ------------------- MAIN EXECUTION -------------------------
try:
    logger.info("Starting Workload Drivers Forecasting Script (Rpt 288, GBQ)…")

    # 1) Pull data from BigQuery ----------------------------------------------
    logger.info("Querying WD view (TotalOrders only)…")
    wd_df = bigquery_manager.run_gbq_sql(GBQ_VIEW_QUERY, return_dataframe=True)
    if not isinstance(wd_df, pd.DataFrame) or wd_df.empty:
        raise ValueError("Workload view returned no data.")

    logger.info("Querying FRED economic data…")
    fred_src = bigquery_manager.run_gbq_sql(FRED_QUERY, return_dataframe=True)
    if not isinstance(fred_src, pd.DataFrame) or fred_src.empty:
        raise ValueError("FRED economic data table returned no data.")

    # Basic prep
    wd_df["date"] = pd.to_datetime(wd_df["date"])
    fred_src["date"] = pd.to_datetime(fred_src["date"])

    # Extend FRED out through the forecast horizon (pure forward fill)
    fred_full = extend_fred_for_horizon(fred_src, FORECAST_HORIZON)
    fred_full = fred_full[["UNRATE", "HSN1F", "FEDFUNDS", "MORTGAGE30US"]].ffill().bfill()
    logger.info(f"✓ FRED forward-filled through {fred_full.index.max().date()}.")

    # 2) Iterate by client: features, models, validation ----------------------
    forecasts, audit_rows = [], []

    # Ensure monthly MS frequency alignment per client
    wd_df = wd_df.sort_values(["client_id", "date"])

    for cid, g in wd_df.groupby("client_id", sort=False):
        client = g[["date", "target_volume"]].copy()
        client = client.set_index("date").sort_index()
        client = client.asfreq("MS")  # month start

        # Minimal history check
        if client["target_volume"].dropna().shape[0] < 18:
            logger.info(f"· Skipping {cid:<25} – Insufficient history ({client.shape[0]} rows).")
            continue

        # Build supervised frame (lags 3/6/12, rolling(3), seasonals, FRED)
        feat = build_supervised_frame(client["target_volume"], fred_full)
        if feat.empty or feat.shape[0] <= TEST_LEN:
            logger.info(f"· Skipping {cid:<25} – No usable rows after feature engineering.")
            continue

        # Train/validation split (last 3 months for validation)
        train, valid = feat.iloc[:-TEST_LEN].copy(), feat.iloc[-TEST_LEN:].copy()
        X_tr, X_val = train.drop(columns="y"), valid.drop(columns="y")
        y_tr, y_val = train["y"].values, valid["y"].values
        val_index   = valid.index

        preds, models = {}, {}

        # --- A) XGBoost ---
        xgb = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )
        xgb.fit(X_tr, y_tr)
        preds["XGBoost"] = xgb.predict(X_val)
        models["XGBoost"] = xgb

        # Columns XGB actually used (fallback to all)
        try:
            gain = xgb.get_booster().get_score(importance_type="gain")
            good_cols = [c for c in X_tr.columns if gain.get(c, 0) > 0] or list(X_tr.columns)
        except Exception:
            good_cols = list(X_tr.columns)

        # --- B) Ridge (log-space) ---
        ridge = Ridge().fit(X_tr[good_cols], np.log1p(y_tr))
        preds["Ridge"] = np.array([safe_expm1(z) for z in ridge.predict(X_val[good_cols])], dtype=float)
        models["Ridge"] = (ridge, good_cols)

        # --- C) Lasso (log-space) ---
        lasso = Lasso().fit(X_tr[good_cols], np.log1p(y_tr))
        preds["Lasso"] = np.array([safe_expm1(z) for z in lasso.predict(X_val[good_cols])], dtype=float)
        models["Lasso"] = (lasso, good_cols)

        # --- D) SeasonalNaive3mGR (validation) ---
        ts = client["target_volume"]
        preds["SeasonalNaive3mGR"] = seasonal_naive_3mgr_val(ts, val_index)

        # --- E) WeightedLagBlend (3/6/12) ---
        wlf = pd.DataFrame({
            "y":    feat["y"],
            "lag3": feat["lag_3"],
            "lag6": feat["lag_6"],
            "lag12":feat["lag_12"]
        }).dropna()

        wlf_tr = wlf.loc[train.index.intersection(wlf.index)]
        wlf_va = wlf.loc[valid.index.intersection(wlf.index)]

        if not wlf_tr.empty and not wlf_va.empty:
            best_w, best_s = (1.0, 0.0, 0.0), np.inf
            ytr = wlf_tr["y"].values
            l3  = wlf_tr["lag3"].values
            l6  = wlf_tr["lag6"].values
            l12 = wlf_tr["lag12"].values

            # Tune on training set (SMAPE)
            for w3, w6, w12 in weight_grid_3_6_12(GRID_STEP, MIN_SEASONAL_W):
                yhat = w3*l3 + w6*l6 + w12*l12
                s = smape(ytr, yhat)
                if s < best_s:
                    best_s, best_w = s, (w3, w6, w12)

            preds["WeightedLagBlend"] = best_w[0]*wlf_va["lag3"].values + best_w[1]*wlf_va["lag6"].values + best_w[2]*wlf_va["lag12"].values
            models["WeightedLagBlend"] = {"w3": best_w[0], "w6": best_w[1], "w12": best_w[2]}

        # Remove models that produced no finite predictions
        drop_keys = [k for k, p in preds.items() if not np.isfinite(p).any()]
        for k in drop_keys:
            preds.pop(k, None); models.pop(k, None)
        if not preds:
            logger.info(f"· Skipping {cid:<25} – No valid model predictions.")
            continue

        # 3) Evaluate on SMAPE (and log MAPE/RMSE for audit)
        smapes = {}
        for m, p in preds.items():
            mask = np.isfinite(p)
            if mask.any():
                yv, pv = y_val[mask], p[mask]
                smapes[m] = smape(yv, pv)
                audit_rows.append({
                    "client_id": cid,
                    "model": m,
                    "SMAPE": smapes[m],
                    "MAPE": mape(yv, pv),
                    "RMSE": rmse(yv, pv)
                })

        if not smapes:
            logger.info(f"· Skipping {cid:<25} – No models could be scored.")
            continue

        best_model = min(smapes, key=smapes.get)
        logger.info(f"· {cid:<25} Best = {best_model:<18} (SMAPE: {smapes[best_model]:.2f})")

        # 4) Recursive 15-month forecast with the winner -----------------------
        # Prepare history deque from the raw time series (latest at end)
        hist = deque(ts.dropna().tolist(), maxlen=48)
        if not hist:
            logger.info(f"· Skipping {cid:<25} – No history after dropna().")
            continue

        # Future date range (month-start)
        future_idx = pd.date_range(ts.index[-1], periods=FORECAST_HORIZON + 1, freq="MS")[1:]
        fx_tag = f"{FX_ID_PREFIX[best_model]}_{STAMP:%Y%m%d}"
        load_ts_str = STAMP.strftime("%Y-%m-%d %H:%M:%S")

        # Snapshot training columns for feature ordering
        xgb_cols = list(X_tr.columns) if best_model == "XGBoost" else None
        lin_cols = models[best_model][1] if best_model in ("Ridge", "Lasso") else None
        mdl_obj  = models.get(best_model)

        # Econ values for forecast months (already forward-filled)
        econ_full = fred_full

        for d in future_idx:
            # Build a feature row consistent with training schema
            if best_model == "XGBoost":
                row = {}
                # lags from history (fallbacks)
                row["lag_3"]  = hist[-3]  if len(hist) >= 3  else hist[-1]
                row["lag_6"]  = hist[-6]  if len(hist) >= 6  else hist[-1]
                row["lag_12"] = hist[-12] if len(hist) >= 12 else hist[-1]
                last3 = list(hist)[-3:] if len(hist) >= 3 else [hist[-1]]*3
                row["rolling_mean_3"] = float(np.mean(last3))
                row["rolling_std_3"]  = float(np.std(last3))
                m = d.month
                row["sin_month"] = np.sin(2*np.pi*m/12.0)
                row["cos_month"] = np.cos(2*np.pi*m/12.0)
                for mo in range(1, 13):
                    row[f"mo_{mo}"] = 1.0 if mo == m else 0.0
                # Econ features (forward-filled)
                econ = econ_full.loc[d] if d in econ_full.index else econ_full.iloc[-1]
                for fred_col in ["UNRATE", "HSN1F", "FEDFUNDS", "MORTGAGE30US"]:
                    if xgb_cols and fred_col in xgb_cols:
                        row[fred_col] = float(econ.get(fred_col, np.nan))
                # order columns exactly as training
                Xn = np.array([[row.get(c, 0.0) for c in xgb_cols]], dtype=float)
                if np.isnan(Xn).any():
                    Xn = np.nan_to_num(Xn, nan=0.0)
                pred = float(mdl_obj.predict(Xn)[0])

            elif best_model in ("Ridge", "Lasso"):
                model, cols = mdl_obj  # (estimator, selected_columns)
                row = {}
                row["lag_3"]  = hist[-3]  if len(hist) >= 3  else hist[-1]
                row["lag_6"]  = hist[-6]  if len(hist) >= 6  else hist[-1]
                row["lag_12"] = hist[-12] if len(hist) >= 12 else hist[-1]
                last3 = list(hist)[-3:] if len(hist) >= 3 else [hist[-1]]*3
                row["rolling_mean_3"] = float(np.mean(last3))
                row["rolling_std_3"]  = float(np.std(last3))
                m = d.month
                row["sin_month"] = np.sin(2*np.pi*m/12.0)
                row["cos_month"] = np.cos(2*np.pi*m/12.0)
                for mo in range(1, 13):
                    row[f"mo_{mo}"] = 1.0 if mo == m else 0.0
                econ = econ_full.loc[d] if d in econ_full.index else econ_full.iloc[-1]
                for fred_col in ["UNRATE", "HSN1F", "FEDFUNDS", "MORTGAGE30US"]:
                    if fred_col in cols:
                        row[fred_col] = float(econ.get(fred_col, np.nan))

                # Ensure every expected column exists; build matrix in order
                Xn = np.array([[row.get(c, 0.0) for c in cols]], dtype=float)
                if np.isnan(Xn).any():
                    Xn = np.nan_to_num(Xn, nan=0.0)

                pred = safe_expm1(model.predict(Xn)[0])

            elif best_model == "WeightedLagBlend":
                w = mdl_obj  # dict of weights
                comps = []
                comps.append(w["w3"]  * (hist[-3]  if len(hist) >= 3  else hist[-1]))
                comps.append(w["w6"]  * (hist[-6]  if len(hist) >= 6  else hist[-1]))
                comps.append(w["w12"] * (hist[-12] if len(hist) >= 12 else hist[-1]))
                pred = float(np.sum(comps))

            else:  # SeasonalNaive3mGR
                pred = seasonal_naive_3mgr_forecast(hist)

            pred = 0.0 if not np.isfinite(pred) else max(0.0, float(pred))
            hist.append(pred)

            forecasts.append({
                "fx_date":   d.strftime("%Y-%m-01"),
                "client_id": cid,
                "fx_vol":    int(round(pred)),
                "fx_id":     fx_tag,                  # underscore format
                "fx_status": "forecast",              # latest rows marked as "forecast"
                "load_ts":   load_ts_str
            })

        # Also record the winning model headline metric row into audit
        audit_rows.append({
            "client_id": cid,
            "model": f"{best_model}_WINNER",
            "SMAPE": smapes[best_model],
            "MAPE": np.nan,
            "RMSE": np.nan
        })

    # 5) Push to GBQ, deduplicate (inactivate older), save CSVs, log ----------
    if not forecasts:
        logger.warning("No forecasts were generated. Exiting without GBQ writes.")
        sys.exit(0)

    fx_df = pd.DataFrame(forecasts)[["fx_date", "client_id", "fx_vol", "fx_id", "fx_status", "load_ts"]]
    logger.info(f"Appending {len(fx_df):,} forecast rows to {DEST_TABLE}…")
    ok = bigquery_manager.import_data_to_bigquery(fx_df, DEST_TABLE, gbq_insert_action="append", auto_convert_df=True)
    if not ok:
        raise Exception("Failed to append forecast data to BigQuery.")

    # Mark older duplicates as inactive (keep only the latest load_ts as 'forecast')
    logger.info("Inactivating older overlapping forecasts in destination table…")
    dedup_sql = f"""
    UPDATE `{DEST_TABLE}` AS t
    SET fx_status = 'inactive'
    WHERE EXISTS (
        SELECT 1
        FROM `{DEST_TABLE}` AS sub
        WHERE sub.client_id = t.client_id
          AND sub.fx_date   = t.fx_date
          AND sub.load_ts   > t.load_ts
    );
    """
    bigquery_manager.run_gbq_sql(dedup_sql, return_dataframe=False)
    logger.info("✓ Older forecast rows marked as inactive.")

    # Save local CSVs (forecasts + audit metrics)
    fx_df.to_csv(LOCAL_CSV, index=False)
    pd.DataFrame(audit_rows).to_csv(AUDIT_CSV, index=False)
    logger.info(f"✓ Local CSVs saved:\n    - {LOCAL_CSV}\n    - {AUDIT_CSV}")

    # Log success to GBQ audit table (per your helper)
    bigquery_manager.update_log_in_bigquery()
    logger.info("✓ Workload forecasting script (Rpt 288) completed successfully.")

except Exception as exc:
    # On any unhandled error, log it and send a notification email
    email_manager.handle_error("Workload Forecasting Script Failure (Rpt 288)", exc, is_test=True)

(venv_Master) PS C:\WFM_Scripting\Forecasting> & C:/Scripting/Python_envs/venv_Master/Scripts/python.exe c:/WFM_Scripting/Forecasting/Rpt_288_File.py
Traceback (most recent call last):
  File "c:\WFM_Scripting\Forecasting\Rpt_288_File.py", line 523, in <module>
    email_manager.handle_error("Workload Forecasting Script Failure (Rpt 288)", exc, is_test=True)
  File "C:\WFM_Scripting\Automation\scripthelper.py", line 1170, in handle_error
    raise exception
  File "c:\WFM_Scripting\Forecasting\Rpt_288_File.py", line 253, in <module>
    fred_full = extend_fred_for_horizon(fred_src, FORECAST_HORIZON)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\WFM_Scripting\Forecasting\Rpt_288_File.py", line 164, in extend_fred_for_horizon
    first = fred_df.index.min().to_period("M").to_timestamp("MS")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "period.pyx", line 1996, in pandas._libs.tslibs.period._Period.to_timestamp
AttributeError: 'pandas._libs.tslibs.offsets.MonthBegin' object has no attribute '_period_dtype_code'
