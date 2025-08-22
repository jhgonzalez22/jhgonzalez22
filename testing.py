# ───────────── ticket_forecast_with_drivers_v3_prod.py ─────────────
"""
CX Platforms – Ticket-volume forecast (TEST w/ Case2OrderRatio)

Models per client_id (per BU):
  • XGBoost                — tree regressor on lag_1..lag_12 + sin/cos month (+ drivers if present)
  • Ridge (log-space)      — linear on XGB-selected features; predicts log1p(y) → expm1; nonneg enforced
  • Lasso (log-space)      — linear (sparse) as above; nonneg enforced
  • WeightedLagBlend       — convex blend of {lag1, lag12, lag24}; weights tuned on 3m SMAPE
        ▸ For tickets, we PREFER provided columns (if present):
            total_cases_opened_lag12     ≈ y_{t-12}
            lag12(total_cases_opened_lag12) ≈ y_{t-24}
          Else, fall back to engineered lags (lag_12, lag_24).
  • SeasonalNaive3mGR      — seasonal naive (t−12) scaled by recent 3-month YoY growth (on target)
        r = mean(t−1..t−3) / mean(t−13..t−15) − 1;   pred = base(t−12) * (1 + r)
  • WeightedLagBlendDrv    — WLB blended with a seasonal+driver bump:
        pred = λ * WLB + (1 − λ) * [ y_{t-12} * (1 + r_drv) ]
        λ tuned per client in {0.50..0.90 step 0.05} by 3m SMAPE
        r_drv is a weighted 3m YoY growth from workload drivers
             (TotalOrders, DistinctCustomers) via weights [0.50, 0.30, 0.20] for (t−1,t−2,t−3)/(t−13,t−14,t−15)
  • Case2OrderRatio (NEW)  — model on ratio_t = cases_t / orders_t
        ▸ Forecast the RATIO via WLB on {ratio lag1, lag12, lag24}
        ▸ Validation: tickets_pred = ratio_pred * actual_orders
        ▸ Forecast:   tickets_pred = ratio_pred * forecasted_orders

Validation & selection:
  • Winner selected by SMAPE on last 3 months (we also compute/log 6-month SMAPE)

Outputs:
  • Local CSVs (TEST mode; not pushing to GBQ):
       - forecast_results.csv
       - model_eval_debug.csv
       - xgb_feature_importance.csv
       - driver_forecasts.csv  (per-client driver trajectories used during recursion)

Implementation notes:
  • Non-finite predictions are guarded and clipped to nonnegative integers.
  • For ML recursion, features are rebuilt step-by-step (no np.roll mistakes).
  • Workload drivers are optional; clients without drivers still run all models.

Driver-forecasting used for XGB/Ridge/Lasso/Case2OrderRatio recursion:
  • Seasonal-naive (t−12) + recent 3-month YoY bump at the anchor (last actual)
    with weights 0.50/0.30/0.20 over months (t−1,t−2,t−3) vs (t−13,t−14,t−15).
  • The bump decays linearly to 0 by horizon h=12 to avoid overextension.
"""

# ------------------- standard imports -----------------------
import os, sys, warnings, pytz, json
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")  # keep logs tidy

# ------------------- helper-package path --------------------
sys.path.append(r"C:\WFM_Scripting\Automation")  # adjust if needed
from scripthelper import Config, Logger, BigQueryManager, EmailManager

# ------------------- initialise scripthelper ---------------
config            = Config(rpt_id=284)
logger            = Logger(config)               # plain-text logger
email_manager     = EmailManager(config)
bigquery_manager  = BigQueryManager(config)

# ------------------- file / table paths --------------------
SQL_QUERY_PATH = (r"C:\WFM_Scripting\Forecasting"
                  r"\GBQ - Non-Tax Platform Ticket Timeseries by Month.sql")

# Workload driver tables (per-BU). We UNION all available.
DRV_TABLES = [
    "tax_clnt_svcs.cx_nontax_smartfees_workload_drivers",
    "tax_clnt_svcs.cx_nontax_appscope_workload_drivers",
    "tax_clnt_svcs.cx_nontax_mercury_workload_drivers",
    "tax_clnt_svcs.cx_nontax_order_management_workload_drivers",
]

DEST_TABLE     = "tax_clnt_svcs.cx_nontax_platforms_forecast"

# Local outputs
LOCAL_CSV       = r"C:\WFM_Scripting\forecast_results.csv"          # forecasts
AUDIT_CSV       = r"C:\WFM_Scripting\model_eval_debug.csv"          # metrics + enrichments
XGB_VIZ_CSV     = r"C:\WFM_Scripting\xgb_feature_importance.csv"    # full per-client gains
DRV_FC_CSV      = r"C:\WFM_Scripting\driver_forecasts.csv"          # driver trajectories used

# ------------------- run switches ---------------------------
PUSH_TO_GBQ        = False   # << TESTING: do not push
FORECAST_HORIZON   = 15      # months out
MAX_LAGS           = 12      # lag_1..lag_12 engineered
VAL_LEN_3M         = 3       # 3-month validation for selection
VAL_LEN_6M         = 6       # also computed for context in audit
STAMP              = datetime.now(pytz.timezone("America/Chicago"))

# Weighted driver YoY smoothing profile (for driver bump)
DRV_ALPHA_PROFILE  = [0.50, 0.30, 0.20]  # map to (t-1, t-2, t-3) vs (t-13, t-14, t-15)
DRV_ALPHA_STR      = "0.50/0.30/0.20"

# Tuning grid for λ (WLB+Drivers)
LAMBDA_GRID        = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

FX_ID_PREFIX = {
    "XGBoost":                "xgb_ticket",
    "Ridge":                  "ridge_ticket",
    "Lasso":                  "lasso_ticket",
    "WeightedLagBlend":       "wlb_ticket",
    "SeasonalNaive3mGR":      "seasonal_naive3mgr_ticket",
    "WeightedLagBlendDrv":    "wlbdrv_ticket",
    "Case2OrderRatio":        "c2o_ticket"      # NEW
}

# ------------------- metric helpers -------------------------
def mape(actual, forecast):
    """Mean Absolute Percentage Error (safe for zeros)."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = np.where(actual == 0, 1.0, np.abs(actual))
    return np.mean(np.abs((actual - forecast) / denom)) * 100

def smape(actual, forecast):
    """Symmetric MAPE (%)."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = (np.abs(actual) + np.abs(forecast)) / 2.0
    denom[denom == 0] = 1.0
    return np.mean(np.abs(actual - forecast) / denom) * 100

def safe_expm1(x, lo=-20.0, hi=20.0):
    """Clip linear output before expm1 to avoid overflow to ±inf."""
    return np.expm1(np.clip(x, lo, hi))

def safe_round(x):
    """Round non-finite to 0; enforce non-negative integers."""
    x = 0.0 if (x is None or not np.isfinite(x)) else x
    return int(max(0, round(x)))

def safe_reg_metrics(y_true, y_pred):
    """NaN-safe wrapper for audit metrics."""
    if y_pred is None:
        return dict(smape=np.nan, mape=np.nan, rmse=np.nan)
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any():
        return dict(smape=np.nan, mape=np.nan, rmse=np.nan)
    return dict(
        smape = smape(y[mask], p[mask]),
        mape  = mape(y[mask], p[mask]),
        rmse  = float(np.sqrt(mean_squared_error(y[mask], p[mask]))),
    )

# ------------------- drivers: pull + transforms -------------------------
def fetch_workload_drivers(bq: BigQueryManager) -> pd.DataFrame:
    """Union drivers and aggregate overlaps."""
    frames = []
    for tbl in DRV_TABLES:
        sql = f"""
        SELECT
            CAST(MonthOfOrder AS DATE) AS date,
            client_id,
            COUNT(DISTINCT CustomerNumber) AS DistinctCustomers,
            SUM(TotalMonthlyOrders)       AS TotalOrders
        FROM `{tbl}`
        GROUP BY 1,2
        ORDER BY 1,2
        """
        try:
            df = bq.run_gbq_sql(sql, return_dataframe=True)
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df)
                logger.info(f"✓ Drivers pulled from {tbl}: {len(df):,} rows")
            else:
                logger.info(f"· No rows returned from {tbl}")
        except Exception as e:
            logger.info(f"· Skipping {tbl} (error: {e})")
    if not frames:
        return pd.DataFrame(columns=["date","client_id","DistinctCustomers","TotalOrders"])
    drv = pd.concat(frames, ignore_index=True)
    drv = (drv.groupby(["date","client_id"], as_index=False)
              .agg({"DistinctCustomers":"sum","TotalOrders":"sum"}))
    return drv

def _driver_w3m_yoy(series: pd.Series, anchor: pd.Timestamp) -> float:
    """Weighted 3m YoY at anchor using DRV_ALPHA_PROFILE; returns 0 if insufficient."""
    w = DRV_ALPHA_PROFILE
    num = (w[0]*series.get(anchor - pd.DateOffset(months=1), np.nan) +
           w[1]*series.get(anchor - pd.DateOffset(months=2), np.nan) +
           w[2]*series.get(anchor - pd.DateOffset(months=3), np.nan))
    den = (w[0]*series.get(anchor - pd.DateOffset(months=13), np.nan) +
           w[1]*series.get(anchor - pd.DateOffset(months=14), np.nan) +
           w[2]*series.get(anchor - pd.DateOffset(months=15), np.nan))
    if np.isfinite(num) and np.isfinite(den) and den != 0:
        return float(num/den - 1.0)
    return 0.0

def forecast_driver_series(series: pd.Series,
                           future_idx: pd.DatetimeIndex,
                           use_yoy_bump: bool = True):
    """
    Forecast a driver with seasonal-naive + (optional) weighted 3m YoY bump
    computed at anchor, decaying linearly to 0 by horizon h=12.
    """
    s = series.astype(float).sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS"))
    anchor = s.index[-1]
    r = _driver_w3m_yoy(s, anchor) if use_yoy_bump else 0.0

    out = {}
    for i, d in enumerate(future_idx, start=1):
        base = s.get(d - pd.DateOffset(months=12), np.nan)
        if not np.isfinite(base):
            base = s.iloc[-1] if len(s) else 0.0
        alpha = max(0.0, 1.0 - i/12.0)  # decay to 0 by 12 months
        out[d] = float(max(0.0, base * (1.0 + alpha*r)))
    return pd.Series(out), r

def compute_driver_r_yoy_weighted(drv_s: pd.Series, ref_months: pd.DatetimeIndex) -> float:
    """Weighted 3m YoY for driver-bump model."""
    w1, w2, w3 = DRV_ALPHA_PROFILE
    rs = []
    for t in ref_months:
        num = (w1*drv_s.get(t - pd.DateOffset(months=1), np.nan) +
               w2*drv_s.get(t - pd.DateOffset(months=2), np.nan) +
               w3*drv_s.get(t - pd.DateOffset(months=3), np.nan))
        den = (w1*drv_s.get(t - pd.DateOffset(months=13), np.nan) +
               w2*drv_s.get(t - pd.DateOffset(months=14), np.nan) +
               w3*drv_s.get(t - pd.DateOffset(months=15), np.nan))
        if np.isfinite(num) and np.isfinite(den) and den != 0:
            rs.append(num/den - 1.0)
    return float(np.mean(rs)) if rs else np.nan

def compute_client_driver_growth(drv_df: pd.DataFrame, cid: str, anchor_month: pd.Timestamp) -> float:
    """Average of TotalOrders & DistinctCustomers weighted YoY signals."""
    if drv_df.empty:
        return np.nan
    sub = drv_df[drv_df["client_id"] == cid].set_index("date").sort_index()
    if sub.empty:
        return np.nan
    sub = sub.reindex(pd.date_range(sub.index.min(), sub.index.max(), freq="MS"))
    r_vals = []
    if "TotalOrders" in sub.columns:
        r_o = compute_driver_r_yoy_weighted(sub["TotalOrders"], pd.DatetimeIndex([anchor_month]))
        if np.isfinite(r_o): r_vals.append(r_o)
    if "DistinctCustomers" in sub.columns:
        r_c = compute_driver_r_yoy_weighted(sub["DistinctCustomers"], pd.DatetimeIndex([anchor_month]))
        if np.isfinite(r_c): r_vals.append(r_c)
    return float(np.mean(r_vals)) if r_vals else np.nan

# ------------------- core modeling helpers -------------------------
def _recent_target_yoy(series: pd.Series, anchor: pd.Timestamp) -> float:
    """Weighted 3m YoY on target; returns NaN if insufficient."""
    w = [0.5, 0.3, 0.2]
    num = (w[0]*series.get(anchor - pd.DateOffset(months=1), np.nan) +
           w[1]*series.get(anchor - pd.DateOffset(months=2), np.nan) +
           w[2]*series.get(anchor - pd.DateOffset(months=3), np.nan))
    den = (w[0]*series.get(anchor - pd.DateOffset(months=13), np.nan) +
           w[1]*series.get(anchor - pd.DateOffset(months=14), np.nan) +
           w[2]*series.get(anchor - pd.DateOffset(months=15), np.nan))
    if np.isfinite(num) and np.isfinite(den) and den != 0:
        return float(num/den - 1.0)
    return np.nan

# ===================================================================
#   PIPELINE: EXTRACT → MODEL & FORECAST → (OPTIONAL) PUSH → SAVE
# ===================================================================

def extract_data():
    """Read tickets & drivers from GBQ, prep dtypes."""
    logger.info(f"Reading SQL file:\n    {SQL_QUERY_PATH}")
    tickets_df = bigquery_manager.run_gbq_sql(SQL_QUERY_PATH, return_dataframe=True)
    if tickets_df.empty:
        raise RuntimeError("BigQuery returned zero rows for tickets")
    logger.info(f"✓ Pulled {tickets_df.shape[0]:,} ticket rows")

    drv_df = fetch_workload_drivers(bigquery_manager)
    if drv_df.empty:
        logger.info("· No workload driver rows found (models will still run).")
    else:
        logger.info(f"✓ Combined driver rows: {drv_df.shape[0]:,}") 

    # Feature engineering basics
    df = tickets_df.copy()
    df["date"]      = pd.to_datetime(df["date"])
    df["month"]     = df["date"].dt.month
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df              = df.sort_values(['client_id', 'date'])
    df              = df.query("date >= '2023-01-01'")

    if not drv_df.empty:
        drv_df["date"] = pd.to_datetime(drv_df["date"])
        drv_df = drv_df.sort_values(["client_id","date"])

    return df, drv_df

def model_and_forecast(df: pd.DataFrame, drv_df: pd.DataFrame):
    """Train/evaluate per client, select winner, forecast 15m, save CSVs (no GBQ push here)."""
    forecasts, audit_rows, xgb_imp_rows = [], [], []
    driver_fc_rows = []

    for cid in df['client_id'].unique():
        client_df = (df[df['client_id'] == cid].set_index('date').sort_index())
        ts = client_df['total_cases_opened']

        if len(ts) < (MAX_LAGS + VAL_LEN_6M + 1):
            logger.info(f"· Skip {cid:<25} – only {len(ts)} pts")
            continue

        # Supervised lag frame for ML models
        feat = pd.DataFrame({'y': ts})
        for lag in range(1, MAX_LAGS + 1):
            feat[f'lag_{lag}'] = feat['y'].shift(lag)
        feat['sin_month'] = np.sin(2 * np.pi * feat.index.month / 12)
        feat['cos_month'] = np.cos(2 * np.pi * feat.index.month / 12)

        # Preferred provided lags
        if 'total_cases_opened_lag12' in client_df.columns:
            lag12_ext = client_df['total_cases_opened_lag12'].copy()
            lag24_ext = lag12_ext.shift(12)
        else:
            lag12_ext = pd.Series(index=feat.index, dtype=float)
            lag24_ext = pd.Series(index=feat.index, dtype=float)

        # Join drivers (if present)
        client_drivers = None
        if not drv_df.empty:
            client_drivers = drv_df[drv_df["client_id"] == cid][["date","DistinctCustomers","TotalOrders"]].copy()
            if not client_drivers.empty:
                client_drivers = client_drivers.set_index("date").sort_index()
                feat = feat.join(client_drivers[["DistinctCustomers","TotalOrders"]], how="left")

        core_cols = ['y'] + [f'lag_{k}' for k in range(1, MAX_LAGS+1)] + ['sin_month','cos_month']
        feat = feat.dropna(subset=core_cols).copy()

        if len(feat) < (VAL_LEN_6M + 1):
            logger.info(f"· Skip {cid:<25} – not enough post-lag rows")
            continue
        train_6m, valid_6m = feat.iloc[:-VAL_LEN_6M], feat.iloc[-VAL_LEN_6M:]
        train_3m, valid_3m = feat.iloc[:-VAL_LEN_3M], feat.iloc[-VAL_LEN_3M:]

        X_tr_6, X_val_6  = train_6m.drop(columns='y'), valid_6m.drop(columns='y')
        y_tr_6, y_val_6  = train_6m['y'], valid_6m['y']
        X_tr_3, X_val_3  = train_3m.drop(columns='y'), valid_3m.drop(columns='y')
        y_tr_3, y_val_3  = train_3m['y'], valid_3m['y']

        preds_3m, preds_6m, models, extras = {}, {}, {}, {}

        # ---------- XGBoost ----------
        xgb = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=50, learning_rate=0.05, max_depth=4, random_state=42
        )
        xgb.fit(X_tr_6, y_tr_6)
        models['XGBoost'] = xgb
        preds_3m['XGBoost'] = xgb.predict(X_val_3)
        preds_6m['XGBoost'] = xgb.predict(X_val_6)

        gain = xgb.get_booster().get_score(importance_type='gain')
        imp_sorted = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)
        xgb_imp_rows.append({"client_id": cid,
                             "feature_importance_gain_json": json.dumps(imp_sorted)})
        good_cols = [c for c in X_tr_6.columns if gain.get(c, 0) > 0] or X_tr_6.columns.tolist()

        # ---------- Ridge / Lasso (log-space) ----------
        y_tr_log = np.log1p(y_tr_6)
        ridge = Ridge().fit(X_tr_6[good_cols], y_tr_log)
        models['Ridge'] = ridge
        preds_3m['Ridge'] = np.expm1(ridge.predict(X_val_3[good_cols]))
        preds_6m['Ridge'] = np.expm1(ridge.predict(X_val_6[good_cols]))

        lasso = Lasso().fit(X_tr_6[good_cols], y_tr_log)
        models['Lasso'] = lasso
        preds_3m['Lasso'] = np.expm1(lasso.predict(X_val_3[good_cols]))
        preds_6m['Lasso'] = np.expm1(lasso.predict(X_val_6[good_cols]))

        # ---------- helper for WLB ----------
        def get_pref_lag_cols(block_idx: pd.DatetimeIndex):
            l1  = feat['lag_1'].reindex(block_idx).astype(float).values
            l12 = (lag12_ext.reindex(block_idx).combine_first(
                   feat['lag_12'].reindex(block_idx))).astype(float).values
            l24_fbk = feat['lag_24'].reindex(block_idx) if 'lag_24' in feat.columns else pd.Series(index=block_idx, dtype=float)
            l24 = (lag24_ext.reindex(block_idx).combine_first(l24_fbk)).astype(float).values
            return l1, l12, l24

        # ---------- WLB ----------
        w_grid = np.arange(0, 1.0 + 1e-9, 0.05)
        l1_v, l12_v, l24_v = get_pref_lag_cols(valid_3m.index)
        y_v = y_val_3.values
        # Mask only where the target is NaN. Lags with NaN will be zeroed, matching prediction logic.
        mask_v = np.isfinite(y_v)
        best_weights, best_s = (1.0, 0.0, 0.0), np.inf
        if mask_v.any():
            y_v_masked = y_v[mask_v]
            # Replace NaN lags with 0, consistent with wlb_block_preds
            l1m  = np.nan_to_num(l1_v[mask_v], 0.0)
            l12m = np.nan_to_num(l12_v[mask_v], 0.0)
            l24m = np.nan_to_num(l24_v[mask_v], 0.0)
            for w1 in w_grid:
                for w2 in w_grid:
                    w3 = 1 - (w1 + w2)
                    if w3 < -1e-12: continue
                    w3 = max(0.0, w3)
                    preds_try = w1*l1m + w2*l12m + w3*l24m
                    s = smape(y_v_masked, preds_try)
                    if s < best_s:
                        best_weights, best_s = (w1, w2, w3), s
        w1, w2, w3 = best_weights
        extras['WeightedLagBlend'] = {"w_lag1": w1, "w_lag12": w2, "w_lag24": w3}

        def wlb_block_preds(idx: pd.DatetimeIndex) -> np.ndarray:
            l1b, l12b, l24b = get_pref_lag_cols(idx)
            l24b = np.where(np.isfinite(l24b), l24b, 0.0)
            return w1*np.nan_to_num(l1b, 0.0) + w2*np.nan_to_num(l12b, 0.0) + w3*l24b

        preds_3m['WeightedLagBlend'] = wlb_block_preds(valid_3m.index)
        preds_6m['WeightedLagBlend'] = wlb_block_preds(valid_6m.index)

        # ---------- SeasonalNaive3mGR ----------
        anchor = ts.index[-1]
        r_tgt = _recent_target_yoy(ts, anchor)

        def seasonal_naive_3mgr(idx: pd.DatetimeIndex) -> np.ndarray:
            vals = []
            for d in idx:
                base = ts.get(d - pd.DateOffset(months=12), np.nan)
                if not np.isfinite(base):
                    vals.append(np.nan)
                else:
                    bump = 1.0 + (r_tgt if np.isfinite(r_tgt) else 0.0)
                    vals.append(base * bump)
            return np.array(vals, dtype=float)

        preds_3m['SeasonalNaive3mGR'] = seasonal_naive_3mgr(valid_3m.index)
        preds_6m['SeasonalNaive3mGR'] = seasonal_naive_3mgr(valid_6m.index)

        # ---------- WLB + Drivers ----------
        r_drv = np.nan
        if client_drivers is not None and not client_drivers.empty:
            r_drv = compute_client_driver_growth(drv_df, cid, anchor)

        if np.isfinite(r_drv):
            def wlbdrv_block_preds(idx: pd.DatetimeIndex, lam: float) -> np.ndarray:
                wlb  = wlb_block_preds(idx)
                sdrv = []
                for d in idx:
                    base = ts.get(d - pd.DateOffset(months=12), np.nan)
                    sdrv.append(base * (1.0 + r_drv) if np.isfinite(base) else np.nan)
                sdrv = np.array(sdrv, dtype=float)
                return lam*wlb + (1.0 - lam)*sdrv

            best_lam, best_s = None, np.inf
            for lam in LAMBDA_GRID:
                p_try = wlbdrv_block_preds(valid_3m.index, lam)
                mask  = np.isfinite(p_try) & np.isfinite(y_val_3.values)
                if not mask.any(): continue
                s = smape(y_val_3.values[mask], p_try[mask])
                if s < best_s:
                    best_s, best_lam = s, lam
            if best_lam is not None:
                preds_3m['WeightedLagBlendDrv'] = wlbdrv_block_preds(valid_3m.index, best_lam)
                preds_6m['WeightedLagBlendDrv'] = wlbdrv_block_preds(valid_6m.index, best_lam)
                extras['WeightedLagBlendDrv'] = {"lambda": best_lam, "r_drv": r_drv}
        else:
            logger.info(f"· {cid:<25} WLB+Drivers skipped (no/insufficient drivers)")

        # ---------- NEW: Case2OrderRatio ----------
        # Build ratio = cases / orders; needs TotalOrders in client_drivers on the same index as 'ts'
        if (client_drivers is not None) and ('TotalOrders' in client_drivers.columns):
            ord_series = client_drivers['TotalOrders'].reindex(ts.index).astype(float)
            # Avoid divide-by-zero
            denom = ord_series.replace(0, np.nan)
            ratio = (ts / denom).replace([np.inf, -np.inf], np.nan).dropna()

            # Need enough ratio points to make lags+validation
            if len(ratio) >= (MAX_LAGS + VAL_LEN_6M + 1):
                ratio_feat = pd.DataFrame({'r': ratio})
                for lag in range(1, MAX_LAGS + 1):
                    ratio_feat[f'lag_{lag}'] = ratio_feat['r'].shift(lag)

                ratio_feat = ratio_feat.dropna().copy()
                if len(ratio_feat) >= (VAL_LEN_6M + 1):
                    r_tr_6, r_val_6 = ratio_feat.iloc[:-VAL_LEN_6M], ratio_feat.iloc[-VAL_LEN_6M:]
                    r_tr_3, r_val_3 = ratio_feat.iloc[:-VAL_LEN_3M], ratio_feat.iloc[-VAL_LEN_3M:]

                    # Tune WLB on ratio via 3m SMAPE measured on TICKETS
                    w_grid = np.arange(0, 1.0 + 1e-9, 0.05)
                    def wlbr_pred(idx):
                        l1 = ratio_feat['lag_1'].reindex(idx).astype(float).values
                        l12 = ratio_feat['lag_12'].reindex(idx).astype(float).values if 'lag_12' in ratio_feat.columns else np.full(len(idx), np.nan)
                        l24 = ratio_feat['lag_24'].reindex(idx).astype(float).values if 'lag_24' in ratio_feat.columns else np.full(len(idx), np.nan)
                        return l1, l12, l24

                    l1_v, l12_v, l24_v = wlbr_pred(r_val_3.index)
                    # Build target tickets & actual orders on validation
                    y_v = ts.reindex(r_val_3.index).values
                    o_v = ord_series.reindex(r_val_3.index).values
                    # Keep only indices where we have all needed inputs
                    mask_v = np.isfinite(y_v) & np.isfinite(o_v)
                    best_w, best_s = (1.0, 0.0, 0.0), np.inf
                    if mask_v.any():
                        y_mv, o_mv = y_v[mask_v], o_v[mask_v]
                        # Replace NaN lags with 0, consistent with prediction function
                        l1m  = np.nan_to_num(l1_v[mask_v], 0.0)
                        l12m = np.nan_to_num(l12_v[mask_v], 0.0)
                        l24m = np.nan_to_num(l24_v[mask_v], 0.0)
                        for w1r in w_grid:
                            for w2r in w_grid:
                                w3r = 1 - (w1r + w2r)
                                if w3r < -1e-12: continue
                                w3r = max(0.0, w3r)
                                ratio_pred = w1r*l1m + w2r*l12m + w3r*l24m
                                y_pred = np.nan_to_num(ratio_pred, 0.0) * o_mv
                                s = smape(y_mv, y_pred)
                                if s < best_s:
                                    best_s, best_w = s, (w1r, w2r, w3r)

                        # store 3m & 6m predictions for audit
                        def predict_tickets(idx, weights):
                            l1b, l12b, l24b = wlbr_pred(idx)
                            l24b = np.where(np.isfinite(l24b), l24b, 0.0)
                            ratio_pred = weights[0]*np.nan_to_num(l1b, 0.0) + \
                                         weights[1]*np.nan_to_num(l12b,0.0) + \
                                         weights[2]*l24b
                            orders_here = ord_series.reindex(idx).astype(float).values
                            orders_here = np.nan_to_num(orders_here, 0.0)
                            return ratio_pred * orders_here

                        preds_3m['Case2OrderRatio'] = predict_tickets(r_val_3.index, best_w)
                        preds_6m['Case2OrderRatio'] = predict_tickets(r_val_6.index, best_w)
                        extras['Case2OrderRatio'] = {"w_r_lag1": best_w[0], "w_r_lag12": best_w[1], "w_r_lag24": best_w[2]}
            else:
                logger.info(f"· {cid:<25} Case2OrderRatio skipped (insufficient ratio history)")
        else:
            logger.info(f"· {cid:<25} Case2OrderRatio skipped (no TotalOrders driver)")

        # ---------- Evaluate & select winner ----------
        models_to_score = list(preds_3m.keys())

        smapes_3 = {m: smape(y_val_3.values, preds_3m[m])
                    for m in models_to_score
                    if np.isfinite(preds_3m[m]).all()}
        smapes_6 = {m: smape(y_val_6.values, preds_6m[m])
                    for m in models_to_score
                    if (m in preds_6m) and np.isfinite(preds_6m[m]).all()}

        xgb_top_json = json.dumps(imp_sorted[:20])
        for m in models_to_score:
            p3 = preds_3m.get(m, None)
            p6 = preds_6m.get(m, None)
            met3 = safe_reg_metrics(y_val_3.values, p3)
            met6 = safe_reg_metrics(y_val_6.values, p6)
            row = dict(
                client_id     = cid,
                model         = m,
                val_smape_3m  = met3['smape'],
                val_smape_6m  = met6['smape'],
                MAPE_3m       = met3['mape'],
                RMSE_3m       = met3['rmse'],
                alpha_profile = DRV_ALPHA_STR
            )
            if m == "XGBoost":
                row["xgb_gain_top"] = xgb_top_json
            if m == "WeightedLagBlend":
                row.update(extras.get("WeightedLagBlend", {}))
            if m == "WeightedLagBlendDrv":
                row.update(extras.get("WeightedLagBlend", {}))
                row.update(extras.get("WeightedLagBlendDrv", {}))
            if m == "Case2OrderRatio":
                row.update(extras.get("Case2OrderRatio", {}))
            audit_rows.append(row)

        if not smapes_3:
            logger.info(f"· {cid:<25} – no valid (all-finite) predictions on 3m val")
            continue

        best_model = min(smapes_3, key=smapes_3.get)
        logger.info(f"· {cid:<25} best = {best_model:<20} SMAPE3m = {smapes_3[best_model]:.2f} | SMAPE6m = {smapes_6.get(best_model, np.nan):.2f}")

        # Mark the winner on this client's recently-appended audit rows
        for r in audit_rows[-len(models_to_score):]:
            r["winner_model"] = best_model

        # ---------- 15-month forecast ----------
        future_idx = pd.date_range(ts.index[-1], periods=FORECAST_HORIZON + 1, freq='MS')[1:]
        fx_tag     = f"{FX_ID_PREFIX[best_model]}_{STAMP:%Y%m%d}"

        # Pre-compute driver trajectories + save them
        driver_fc = {}
        r_notes   = {}
        if (client_drivers is not None) and (not client_drivers.empty):
            if 'DistinctCustomers' in client_drivers.columns:
                fcC, rC = forecast_driver_series(client_drivers['DistinctCustomers'], future_idx, use_yoy_bump=True)
                driver_fc['DistinctCustomers'] = fcC
                r_notes['r_DistinctCustomers'] = rC
            if 'TotalOrders' in client_drivers.columns:
                fcO, rO = forecast_driver_series(client_drivers['TotalOrders'], future_idx, use_yoy_bump=True)
                driver_fc['TotalOrders'] = fcO
                r_notes['r_TotalOrders'] = rO

            for d in future_idx:
                driver_fc_rows.append({
                    "fx_date": d.strftime('%Y-%m-%d'),
                    "client_id": cid,
                    "DistinctCustomers_fc": float(driver_fc.get('DistinctCustomers', pd.Series()).get(d, np.nan)),
                    "TotalOrders_fc": float(driver_fc.get('TotalOrders', pd.Series()).get(d, np.nan)),
                    "r_DistinctCustomers": r_notes.get("r_DistinctCustomers", np.nan),
                    "r_TotalOrders": r_notes.get("r_TotalOrders", np.nan),
                    "method": "seasonal_naive_3mYoY_decay",
                    "alpha_profile": DRV_ALPHA_STR,
                    "load_ts": STAMP
                })

        def append_fx(date_obj, vol):
            forecasts.append(dict(
                fx_date   = date_obj.strftime('%Y-%m-%d'),
                client_id = cid,
                vol_type  = "tickets",
                fx_vol    = safe_round(vol),
                fx_id     = fx_tag,
                fx_status = "A",
                load_ts   = STAMP
            ))

        # ---- recursive generation per winner ----
        if best_model == 'XGBoost':
            cols  = X_tr_6.columns
            last_vals = deque(feat['y'].iloc[-MAX_LAGS:].tolist(), maxlen=MAX_LAGS)
            last_drv_vals = None
            if client_drivers is not None and not client_drivers.empty:
                last_drv_vals = client_drivers.loc[:feat.index[-1]].iloc[-1].to_dict()

            for d in future_idx:
                row = {}
                for k in range(1, MAX_LAGS+1):
                    row[f'lag_{k}'] = float(last_vals[-k]) if len(last_vals) >= k else float(last_vals[-1] if len(last_vals) else 0.0)
                row['sin_month'] = np.sin(2*np.pi*d.month/12)
                row['cos_month'] = np.cos(2*np.pi*d.month/12)
                if 'DistinctCustomers' in cols:
                    if 'DistinctCustomers' in driver_fc and d in driver_fc['DistinctCustomers'].index:
                        row['DistinctCustomers'] = float(driver_fc['DistinctCustomers'].loc[d])
                    elif last_drv_vals is not None:
                        row['DistinctCustomers'] = float(last_drv_vals.get('DistinctCustomers', np.nan))
                if 'TotalOrders' in cols:
                    if 'TotalOrders' in driver_fc and d in driver_fc['TotalOrders'].index:
                        row['TotalOrders'] = float(driver_fc['TotalOrders'].loc[d])
                    elif last_drv_vals is not None:
                        row['TotalOrders'] = float(last_drv_vals.get('TotalOrders', np.nan))

                xrow = np.array([row.get(c, np.nan) for c in cols], dtype=float).reshape(1,-1)
                pred = float(models['XGBoost'].predict(xrow)[0])
                last_vals.append(pred)
                append_fx(d, pred)

        elif best_model in ('Ridge', 'Lasso'):
            cols  = good_cols
            last_vals = deque(feat['y'].iloc[-MAX_LAGS:].tolist(), maxlen=MAX_LAGS)
            mdl   = models[best_model]
            last_drv_vals = None
            if client_drivers is not None and not client_drivers.empty:
                last_drv_vals = client_drivers.loc[:feat.index[-1]].iloc[-1].to_dict()

            def driver_value(col, date_obj):
                if col in driver_fc and date_obj in driver_fc[col].index:
                    return float(driver_fc[col].loc[date_obj])
                if last_drv_vals is not None and (col in last_drv_vals):
                    try:
                        v = float(last_drv_vals[col])
                        if np.isfinite(v): return v
                    except Exception:
                        pass
                return 0.0

            for d in future_idx:
                row = {}
                for k in range(1, MAX_LAGS+1):
                    row[f'lag_{k}'] = float(last_vals[-k]) if len(last_vals) >= k else float(last_vals[-1] if len(last_vals) else 0.0)
                row['sin_month'] = np.sin(2*np.pi*d.month/12)
                row['cos_month'] = np.cos(2*np.pi*d.month/12)
                if 'DistinctCustomers' in cols:
                    row['DistinctCustomers'] = driver_value('DistinctCustomers', d)
                if 'TotalOrders' in cols:
                    row['TotalOrders'] = driver_value('TotalOrders', d)

                xrow = np.array([row.get(c, 0.0) for c in cols], dtype=float).reshape(1,-1)
                raw  = float(mdl.predict(xrow)[0])
                pred = safe_expm1(raw)
                last_vals.append(pred)
                append_fx(d, pred)

        elif best_model == 'WeightedLagBlend':
            w1 = extras['WeightedLagBlend']['w_lag1']
            w2 = extras['WeightedLagBlend']['w_lag12']
            w3 = extras['WeightedLagBlend']['w_lag24']
            hist_s = ts.copy()  # Use a Series to handle dates correctly
            for d in future_idx:
                # Lag 1 is always from the last value in our history
                y_t1 = hist_s.iloc[-1] if not hist_s.empty else np.nan

                # Lag 12: prefer external, then historical (matches validation logic)
                y_t12_from_ext = lag12_ext.get(d, np.nan)
                y_t12_from_hist = hist_s.get(d - pd.DateOffset(months=12), np.nan)
                y_t12 = y_t12_from_ext if pd.notna(y_t12_from_ext) else y_t12_from_hist

                # Lag 24: prefer external, then historical
                y_t24_from_ext = lag24_ext.get(d, np.nan)
                y_t24_from_hist = hist_s.get(d - pd.DateOffset(months=24), np.nan)
                y_t24 = y_t24_from_ext if pd.notna(y_t24_from_ext) else y_t24_from_hist

                pred = (w1*(0.0 if not np.isfinite(y_t1)  else y_t1)  +
                        w2*(0.0 if not np.isfinite(y_t12) else y_t12) +
                        w3*(0.0 if not np.isfinite(y_t24) else y_t24))
                hist_s.loc[d] = pred
                append_fx(d, pred)

        elif best_model == 'SeasonalNaive3mGR':
            hist_s = ts.copy()
            for d in future_idx:
                base = hist_s.get(d - pd.DateOffset(months=12), np.nan)
                if not np.isfinite(base):
                    base = hist_s.iloc[-1] if not hist_s.empty else 0.0
                bump = 1.0 + (r_tgt if np.isfinite(r_tgt) else 0.0)
                pred = base * bump
                hist_s.loc[d] = pred
                append_fx(d, pred)

        elif best_model == 'WeightedLagBlendDrv':
            lam  = extras['WeightedLagBlendDrv']['lambda']
            rdrv = extras['WeightedLagBlendDrv']['r_drv']
            w1, w2, w3 = extras['WeightedLagBlend']['w_lag1'], extras['WeightedLagBlend']['w_lag12'], extras['WeightedLagBlend']['w_lag24']
            hist_s = ts.copy()
            for d in future_idx:
                y_t1 = hist_s.iloc[-1] if not hist_s.empty else np.nan
                y_t12_from_ext = lag12_ext.get(d, np.nan)
                y_t12_from_hist = hist_s.get(d - pd.DateOffset(months=12), np.nan)
                y_t12 = y_t12_from_ext if pd.notna(y_t12_from_ext) else y_t12_from_hist
                y_t24_from_ext = lag24_ext.get(d, np.nan)
                y_t24_from_hist = hist_s.get(d - pd.DateOffset(months=24), np.nan)
                y_t24 = y_t24_from_ext if pd.notna(y_t24_from_ext) else y_t24_from_hist

                wlb = (w1*(0.0 if not np.isfinite(y_t1)  else y_t1)  +
                       w2*(0.0 if not np.isfinite(y_t12) else y_t12) +
                       w3*(0.0 if not np.isfinite(y_t24) else y_t24))

                sdrv_base = y_t12  # Use the same preferred y_t12
                sdrv = (sdrv_base * (1.0 + rdrv)) if np.isfinite(sdrv_base) else (hist_s.iloc[-1] if not hist_s.empty else 0.0)

                pred = lam * wlb + (1.0 - lam) * sdrv
                hist_s.loc[d] = pred
                append_fx(d, pred)

        elif best_model == 'Case2OrderRatio':
            # Forecast ratio with tuned WLB weights, then multiply by forecasted orders
            best_w = (extras.get('Case2OrderRatio') or {})
            wr1, wr12, wr24 = best_w.get('w_r_lag1', 1.0), best_w.get('w_r_lag12', 0.0), best_w.get('w_r_lag24', 0.0)

            # Build ratio history again (robust to missing/zero)
            if (client_drivers is not None) and ('TotalOrders' in client_drivers.columns):
                ord_series = client_drivers['TotalOrders'].reindex(ts.index).astype(float).replace(0, np.nan)
                ratio_s = (ts / ord_series).replace([np.inf, -np.inf], np.nan).dropna()

                for d in future_idx:
                    # WLB on ratio, using date-based lookups
                    r1  = ratio_s.iloc[-1] if not ratio_s.empty else 0.0
                    # Fallback to last known ratio (r1) if seasonal lag is missing
                    r12 = ratio_s.get(d - pd.DateOffset(months=12), r1)
                    r24 = ratio_s.get(d - pd.DateOffset(months=24), r1)
                    ratio_pred = wr1*r1 + wr12*r12 + wr24*r24

                    # Orders forecast (fallback to last known if missing)
                    ord_fc = driver_fc.get('TotalOrders', pd.Series()).get(d)
                    if not pd.notna(ord_fc) or not np.isfinite(ord_fc):
                        ord_fc = float(client_drivers['TotalOrders'].iloc[-1]) if (client_drivers is not None and not client_drivers.empty) else 0.0

                    y_pred = max(0.0, ratio_pred * max(0.0, ord_fc))
                    ratio_s.loc[d] = max(0.0, ratio_pred) # Append to series for next step
                    append_fx(d, y_pred)
            else:
                # Fallback: carry last observed (should rarely hit because we gated earlier)
                last = ts.iloc[-1]
                for d in future_idx:
                    append_fx(d, last)

        else:
            # Safety: carry last observed
            last = ts.iloc[-1]
            for d in future_idx:
                append_fx(d, last)

    # ------------- collect DFs -------------
    fx_df = pd.DataFrame(forecasts)[
        ["fx_date", "client_id", "vol_type", "fx_vol", "fx_id", "fx_status", "load_ts"]
    ]
    audit_df   = pd.DataFrame(audit_rows)
    xgb_imp_df = pd.DataFrame(xgb_imp_rows)
    drv_fc_df  = pd.DataFrame(driver_fc_rows)

    # Sort for readability
    if not fx_df.empty:
        fx_df = fx_df.sort_values(["client_id","fx_date"])
    if not audit_df.empty:
        audit_df = audit_df.sort_values(["client_id","model"])
    if not xgb_imp_df.empty:
        xgb_imp_df = xgb_imp_df.sort_values(["client_id"])
    if not drv_fc_df.empty:
        drv_fc_df = drv_fc_df.sort_values(["client_id","fx_date"])

    # Save local CSVs (TEST)
    fx_df.to_csv(LOCAL_CSV, index=False)
    audit_df.to_csv(AUDIT_CSV, index=False)
    xgb_imp_df.to_csv(XGB_VIZ_CSV, index=False)
    if not drv_fc_df.empty:
        drv_fc_df.to_csv(DRV_FC_CSV, index=False)
    logger.info(f"✓ CSVs saved to\n    {LOCAL_CSV}\n    {AUDIT_CSV}\n    {XGB_VIZ_CSV}\n    {DRV_FC_CSV}")

    return fx_df, audit_df, xgb_imp_df, drv_fc_df

def push_results(fx_df: pd.DataFrame):
    """Optionally push to GBQ + inactivate old rows (skipped in TEST)."""
    if PUSH_TO_GBQ:
        logger.info(f"Pushing {len(fx_df)} rows to {DEST_TABLE} …")
        ok = bigquery_manager.import_data_to_bigquery(
                fx_df, DEST_TABLE, gbq_insert_action="append", auto_convert_df=True)
        if not ok:
            raise RuntimeError("GBQ import failed")

        dedup_sql = f"""
        UPDATE `{DEST_TABLE}` AS tgt
           SET fx_status = 'I'
         WHERE EXISTS (
              SELECT 1 FROM `{DEST_TABLE}` AS src
               WHERE src.client_id = tgt.client_id
                 AND src.vol_type  = tgt.vol_type
                 AND src.fx_date   = tgt.fx_date
                 AND src.load_ts   > tgt.load_ts );
        """
        bigquery_manager.run_gbq_sql(dedup_sql, return_dataframe=False)
        logger.info("✓ Older forecasts set to inactive")

        bigquery_manager.update_log_in_bigquery()
        logger.info("✓ Success run-log to GBQ")
    else:
        logger.info("TEST mode: skipping GBQ push & dedup/update.")

def main():
    try:
        logger.info("Starting Ticket Forecasting Script (TEST w/ Case2OrderRatio) …")
        df, drv_df = extract_data()
        fx_df, audit_df, xgb_imp_df, drv_fc_df = model_and_forecast(df, drv_df)
        push_results(fx_df)  # no-op in TEST unless PUSH_TO_GBQ=True
        logger.info("✓ Ticket forecasting completed (TEST mode)")
    except Exception as exc:
        email_manager.handle_error("Ticket Forecasting Script Failure", exc, is_test=True)
        raise

if __name__ == "__main__":
    main()
# ─────────────────────────────────────────────────────────────
