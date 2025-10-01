# ───────────────── workload_forecast_multi_model.py ─────────────────
"""
CX Platforms – Workload Drivers Forecast (TotalOrders)

Pipeline (PC/GBQ):
  1) Pull monthly client time series (TotalOrders only) from BigQuery.
  2) Build features limited to lags 3/6/12 (to avoid overfitting).
  3) Benchmark 6 models on the last 6 months using SMAPE:
      - SeasonalNaive
      - SeasonalNaiveGR
      - SeasonalNaive3mDiffGR
      - WeightedLagBlend
      - OLS (log-space)
      - MA_GrowthAdj (volatility-aware: MA3 baseline + growth ratio)
     * Seasonal models attempt when windows exist.
     * Lag models evaluate only if lag rows exist (no hard InsufficientHistory stop).
  4) Select best model per client (lowest SMAPE).
  5) Produce recursive 15-month forecast.
  6) Append to GBQ, inactivate older duplicates, save CSVs, log success.

Output FX schema: fx_date, client_id, fx_vol, fx_id, fx_status, load_ts
"""

# ------------------- standard imports -----------------------
import os, sys, warnings, pytz
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------- helper-package path --------------------
sys.path.append(r'C:\WFM_Scripting\Automation')
from scripthelper import Config, Logger, BigQueryManager, EmailManager

# ------------------- initialise scripthelper ---------------
config            = Config(rpt_id=288)
logger            = Logger(config)
email_manager     = EmailManager(config)
bigquery_manager  = BigQueryManager(config)

# ------------------- file / table paths --------------------
GBQ_VIEW_QUERY = """
SELECT
  MonthOfOrder AS date,
  client_id,
  TotalOrders  AS target_volume
FROM `tax_clnt_svcs.view_cx_nontax_platforms_workload_drivers`
WHERE MonthOfOrder < '2025-01-01'
"""
DEST_TABLE = "tax_clnt_svcs.cx_nontax_platforms_workload_drivers_fx"

# --- Local CSV outputs ---
LOCAL_CSV = r"C:\WFM_Scripting\cx_nontax_workload_drivers_fx_results.csv"
AUDIT_CSV = r"C:\WFM_Scripting\cx_nontax_workload_drivers_fx_model_eval.csv"

# ------------------- parameters -----------------------------
FORECAST_HORIZON = 15      # months ahead
TEST_LEN         = 6       # last 6 months for validation
LAGS             = (3, 6, 12)
GRID_STEP        = 0.05    # WeightedLagBlend search
MIN_W12          = 0.0     # allow zero; set >0.0 to enforce seasonality minimum

STAMP_TZ = pytz.timezone("America/Chicago")
STAMP    = datetime.now(STAMP_TZ)

FX_ID_PREFIX = {
    "SeasonalNaive":           "snaive_workload",
    "SeasonalNaiveGR":         "snaive_gr_workload",
    "SeasonalNaive3mDiffGR":   "snaive_diffgr_workload",
    "WeightedLagBlend":        "wblend_workload",
    "OLS":                     "ols_workload",
    "MA_GrowthAdj":            "magr_workload",
}

# ------------------- metrics -------------------------------
def smape(a, f):
    a = np.asarray(a, dtype=float)
    f = np.asarray(f, dtype=float)
    denom = (np.abs(a) + np.abs(f)) / 2.0
    out = np.where(denom == 0.0, 0.0, np.abs(a - f) / denom)
    return float(np.mean(out) * 100.0)

def mape(a, f):
    a = np.asarray(a, dtype=float)
    f = np.asarray(f, dtype=float)
    return float(np.mean(np.abs((a - f) / np.where(a == 0, 1, a))) * 100.0)

def rmse(a, f):
    a = np.asarray(a, dtype=float)
    f = np.asarray(f, dtype=float)
    return float(np.sqrt(np.mean((a - f) ** 2)))

# ------------------- helpers: features & windows ------------
def add_lags(ts: pd.Series) -> pd.DataFrame:
    """Return DataFrame with y and lag_3/lag_6/lag_12."""
    df = pd.DataFrame({"y": ts})
    for L in LAGS:
        df[f"lag_{L}"] = df["y"].shift(L)
    return df

def recent_3m_growth(ts: pd.Series, pos: int) -> float:
    """r_recent(t) = mean(y[t-3:t]) / mean(y[t-6:t-3]) - 1"""
    if pos < 6:
        return 0.0
    cur3  = ts.iloc[pos-3:pos].values.astype(float)
    prev3 = ts.iloc[pos-6:pos-3].values.astype(float)
    prev_mean = float(np.mean(prev3))
    return (float(np.mean(cur3)) / prev_mean - 1.0) if prev_mean > 0 else 0.0

def prior_year_3m_growth(ts: pd.Series, pos: int) -> float:
    """r_py(t) = mean(y[t-15:t-12]) / mean(y[t-18:t-15]) - 1"""
    if pos < 18:
        return 0.0
    win_py  = ts.iloc[pos-15:pos-12].values.astype(float)
    prev_py = ts.iloc[pos-18:pos-15].values.astype(float)
    prev_mean = float(np.mean(prev_py))
    return (float(np.mean(win_py)) / prev_mean - 1.0) if prev_mean > 0 else 0.0

# ------------------- seasonal models (validation) -----------
def seasonal_naive_series(ts: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    return ts.shift(12).reindex(idx)

def seasonal_naive_gr_series(ts: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    out = []
    for d in idx:
        if d not in ts.index:
            out.append(np.nan); continue
        pos = ts.index.get_loc(d)
        base = ts.shift(12).get(d, np.nan)
        if not np.isfinite(base):
            out.append(np.nan); continue
        r = recent_3m_growth(ts, pos)
        out.append(max(0.0, float(base) * (1.0 + r)))
    return pd.Series(out, index=idx, dtype=float)

def seasonal_naive_3m_diff_gr_series(ts: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    out = []
    for d in idx:
        if d not in ts.index:
            out.append(np.nan); continue
        pos = ts.index.get_loc(d)
        base = ts.shift(12).get(d, np.nan)
        if not np.isfinite(base):
            out.append(np.nan); continue
        r  = recent_3m_growth(ts, pos)
        rp = prior_year_3m_growth(ts, pos)
        out.append(max(0.0, float(base) * (1.0 + (r - rp))))
    return pd.Series(out, index=idx, dtype=float)

# ------------------- Weighted Lag Blend ---------------------
def weight_grid(step=GRID_STEP, min_w12=MIN_W12):
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    for w3 in vals:
        for w6 in vals:
            w12 = 1.0 - (w3 + w6)
            if w12 < -1e-12:
                continue
            w12 = max(0.0, w12)
            if w12 + 1e-12 >= min_w12:
                yield (float(w3), float(w6), float(w12))

# ------------------- OLS (log-space, closed form) -----------
def fit_ols_logspace(X: np.ndarray, y: np.ndarray):
    """Solve β for log1p(y) = Xβ."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    y = np.where(np.isfinite(y), y, 0.0)
    y = np.clip(y, 0.0, None)
    y_log = np.log1p(y)
    beta, *_ = np.linalg.lstsq(X, y_log, rcond=None)
    return beta

def predict_ols_logspace(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    z = X @ beta
    z = np.clip(z, -20.0, 20.0)
    return np.expm1(z)

# ------------------- MA_GrowthAdj (validation) --------------
def magr_val_series(ts: pd.Series, val_index: pd.DatetimeIndex) -> pd.Series:
    """Volatility-aware smoothing: MA3 baseline * (1 + recent 3m growth)."""
    out = []
    for d in val_index:
        if d not in ts.index:
            out.append(np.nan); continue
        pos = ts.index.get_loc(d)
        if isinstance(pos, slice):
            pos = pos.start
        if pos < 6:
            out.append(np.nan); continue
        ma3   = float(np.nanmean(ts.iloc[pos-3:pos].values))
        prev3 = float(np.nanmean(ts.iloc[pos-6:pos-3].values))
        r = (ma3/prev3 - 1.0) if prev3 > 0 else 0.0
        out.append(max(0.0, ma3 * (1.0 + r)))
    return pd.Series(out, index=val_index, dtype=float)

# ------------------- MAIN -----------------------------------
try:
    logger.info("Starting Workload Forecasting Script (Rpt 288, GBQ)…")

    # 1) Pull data
    logger.info("Querying WD view (TotalOrders only)…")
    wd_df = bigquery_manager.run_gbq_sql(GBQ_VIEW_QUERY, return_dataframe=True)
    if not isinstance(wd_df, pd.DataFrame) or wd_df.empty:
        raise ValueError("Workload view returned no data.")

    wd_df["date"] = pd.to_datetime(wd_df["date"])
    wd_df = wd_df.sort_values(["client_id", "date"])

    forecasts, audit_rows = [], []

    for cid, g in wd_df.groupby("client_id", sort=False):
        # Base monthly series
        ts = g.set_index("date")["target_volume"].asfreq("MS").sort_index()

        # Robust fill for modeling (keeps structure; helps lags/seasonality)
        ts_model = ts.copy().ffill().bfill()

        # Build lag frame on filled series
        feat = add_lags(ts_model)

        if feat.shape[0] <= TEST_LEN:
            logger.info(f"· Skipping {cid:<25} – not enough rows to form a {TEST_LEN}-month validation window.")
            continue

        # Train/Validation split (last 6 months)
        val_index = feat.index[-TEST_LEN:]
        tr_index  = feat.index[:-TEST_LEN]

        # -------------------- Predictions dict --------------------
        preds = {}

        # Originals (validation) on filled series
        preds["SeasonalNaive"]         = seasonal_naive_series(ts_model, val_index)
        preds["SeasonalNaiveGR"]       = seasonal_naive_gr_series(ts_model, val_index)
        preds["SeasonalNaive3mDiffGR"] = seasonal_naive_3m_diff_gr_series(ts_model, val_index)

        # WeightedLagBlend & OLS only where complete lags exist
        wlf    = feat[["y", "lag_3", "lag_6", "lag_12"]].dropna()
        wlf_tr = wlf.loc[wlf.index.intersection(tr_index)]
        wlf_va = wlf.loc[wlf.index.intersection(val_index)]
        wblend_weights, ols_beta = None, None

        if not wlf_tr.empty and not wlf_va.empty:
            # Weighted Lag Blend (tune on TRAIN)
            best_w, best_s = (1.0, 0.0, 0.0), np.inf
            ytr = wlf_tr["y"].values
            l3  = wlf_tr["lag_3"].values
            l6  = wlf_tr["lag_6"].values
            l12 = wlf_tr["lag_12"].values
            for w3, w6, w12 in weight_grid(GRID_STEP, MIN_W12):
                yhat = w3*l3 + w6*l6 + w12*l12
                s = smape(ytr, yhat)
                if np.isfinite(s) and s < best_s:
                    best_s, best_w = s, (w3, w6, w12)
            wblend_weights = {"w3": best_w[0], "w6": best_w[1], "w12": best_w[2]}
            preds["WeightedLagBlend"] = pd.Series(
                best_w[0]*wlf_va["lag_3"].values
                + best_w[1]*wlf_va["lag_6"].values
                + best_w[2]*wlf_va["lag_12"].values,
                index=wlf_va.index, dtype=float
            )

            # OLS (log-space)
            Xtr = np.column_stack([
                np.ones(len(wlf_tr)),
                wlf_tr["lag_3"].values,
                wlf_tr["lag_6"].values,
                wlf_tr["lag_12"].values,
            ]).astype(np.float64)
            beta = fit_ols_logspace(Xtr, wlf_tr["y"].values.astype(np.float64))
            Xva = np.column_stack([
                np.ones(len(wlf_va)),
                wlf_va["lag_3"].values,
                wlf_va["lag_6"].values,
                wlf_va["lag_12"].values,
            ]).astype(np.float64)
            preds["OLS"] = pd.Series(predict_ols_logspace(Xva, beta), index=wlf_va.index, dtype=float)
            ols_beta = beta

        # MA_GrowthAdj (validation) on filled series
        preds["MA_GrowthAdj"] = magr_val_series(ts_model, val_index)

        # -------------------- Evaluate (SMAPE) --------------------
        smapes = {}
        for m, pser in preds.items():
            if not isinstance(pser, pd.Series):
                pser = pd.Series(pser, index=val_index, dtype=float)
            y_true = ts_model.reindex(pser.index).astype(float)  # eval against filled ground truth
            idx = y_true.index[y_true.notna() & pser.notna()]
            if len(idx) > 0:
                yv = y_true.loc[idx].to_numpy(dtype=float)
                pv = pser.loc[idx].to_numpy(dtype=float)
                smapes[m] = smape(yv, pv)
                audit_rows.append({
                    "client_id": cid,
                    "model": m,
                    "SMAPE": smapes[m],
                    "MAPE": mape(yv, pv),
                    "RMSE": rmse(yv, pv)
                })

        if not smapes:
            logger.info(f"· Skipping {cid:<25} – no models could be evaluated (insufficient overlapping windows).")
            continue

        # Selection with tie-break priority
        priority = ["SeasonalNaive3mDiffGR", "SeasonalNaiveGR", "SeasonalNaive",
                    "WeightedLagBlend", "OLS", "MA_GrowthAdj"]
        best_model = min(smapes, key=lambda k: (smapes[k], priority.index(k) if k in priority else 999))
        logger.info(f"· {cid:<25} Best = {best_model:<20} (SMAPE: {smapes[best_model]:.2f})")

        # ===== Recursive 15-month forecast with the winner =====
        hist = deque(ts_model.dropna().tolist(), maxlen=120)
        if not hist:
            logger.info(f"· Skipping {cid:<25} – No history after dropna().")
            continue

        last_date = ts_model.index.max()
        future_idx = pd.date_range(last_date, periods=FORECAST_HORIZON + 1, freq="MS")[1:]
        fx_tag = f"{FX_ID_PREFIX[best_model]}_{STAMP:%Y%m%d}"
        load_ts_str = STAMP.strftime("%Y-%m-%d %H:%M:%S")

        def append_row(date_obj, value):
            value = 0.0 if not np.isfinite(value) else max(0.0, float(value))
            forecasts.append({
                "fx_date":   date_obj.strftime("%Y-%m-01"),
                "client_id": cid,
                "fx_vol":    int(round(value)),
                "fx_id":     fx_tag,
                "fx_status": "forecast",
                "load_ts":   load_ts_str
            })

        # Recursive generation
        if best_model == "MA_GrowthAdj":
            h = list(ts_model.dropna())
            for d in future_idx:
                if len(h) < 6:
                    pred = h[-1]
                else:
                    ma3   = float(np.mean(h[-3:]))
                    prev3 = float(np.mean(h[-6:-3]))
                    r     = (ma3/prev3 - 1.0) if prev3 > 0 else 0.0
                    pred  = ma3 * (1.0 + r)
                h.append(pred)
                append_row(d, pred)

        else:
            for d in future_idx:
                n = len(hist)
                L = list(hist)  # convert deque to list for safe slicing

                if best_model == "SeasonalNaive":
                    base = L[-12] if n >= 12 else L[-1]
                    pred = float(base)

                elif best_model == "SeasonalNaiveGR":
                    base = L[-12] if n >= 12 else L[-1]
                    if n >= 6:
                        cur3  = float(np.mean(L[-3:]))
                        prev3 = float(np.mean(L[-6:-3]))
                        r = (cur3/prev3 - 1.0) if prev3 > 0 else 0.0
                    else:
                        r = 0.0
                    pred = float(base) * (1.0 + r)

                elif best_model == "SeasonalNaive3mDiffGR":
                    base = L[-12] if n >= 12 else L[-1]
                    if n >= 6:
                        cur3  = float(np.mean(L[-3:]))
                        prev3 = float(np.mean(L[-6:-3]))
                        r = (cur3/prev3 - 1.0) if prev3 > 0 else 0.0
                    else:
                        r = 0.0
                    if n >= 18:
                        win_py  = float(np.mean(L[-15:-12]))
                        prev_py = float(np.mean(L[-18:-15]))
                        rpy = (win_py/prev_py - 1.0) if prev_py > 0 else 0.0
                    else:
                        rpy = 0.0
                    pred = float(base) * (1.0 + (r - rpy))

                elif best_model == "WeightedLagBlend" and wblend_weights is not None:
                    x3  = L[-3]  if n >= 3  else L[-1]
                    x6  = L[-6]  if n >= 6  else L[-1]
                    x12 = L[-12] if n >= 12 else L[-1]
                    pred = wblend_weights["w3"]*x3 + wblend_weights["w6"]*x6 + wblend_weights["w12"]*x12

                elif best_model == "OLS" and ols_beta is not None:
                    x3  = L[-3]  if n >= 3  else L[-1]
                    x6  = L[-6]  if n >= 6  else L[-1]
                    x12 = L[-12] if n >= 12 else L[-1]
                    Xn  = np.array([[1.0, x3, x6, x12]], dtype=np.float64)
                    pred = float(predict_ols_logspace(Xn, ols_beta)[0])

                else:
                    pred = float(L[-1])

                hist.append(pred)
                append_row(d, pred)

        # Winner audit marker
        audit_rows.append({
            "client_id": cid,
            "model": f"{best_model}_WINNER",
            "SMAPE": smapes[best_model],
            "MAPE": np.nan,
            "RMSE": np.nan
        })

    # -------------------- Push to GBQ & save --------------------
    if not forecasts:
        logger.warning("No forecasts were generated. Exiting without GBQ writes.")
        sys.exit(0)

    fx_df = pd.DataFrame(forecasts)[["fx_date", "client_id", "fx_vol", "fx_id", "fx_status", "load_ts"]]
    logger.info(f"Appending {len(fx_df):,} forecast rows to {DEST_TABLE}…")
    ok = bigquery_manager.import_data_to_bigquery(
        fx_df, DEST_TABLE, gbq_insert_action="append", auto_convert_df=True
    )
    if not ok:
        raise Exception("Failed to append forecast data to BigQuery.")

    # Inactivate older overlapping rows (keep only the latest load_ts as 'forecast')
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
