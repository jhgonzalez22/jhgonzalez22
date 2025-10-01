#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast pipeline with detailed comments.

What it does
------------
1) Loads 'wd.xlsx' from the SAME folder as this script.
2) Uses only: MonthOfOrder, client_id, TotalOrders.
3) Builds lags for 3, 6, 12 months (ONLY these to avoid overfitting).
4) Benchmarks 5 models per client:
      - SeasonalNaive                -> y[t-12]
      - SeasonalNaiveGR              -> y[t-12] * (1 + recent 3-month growth)
      - SeasonalNaive3mDiffGR        -> y[t-12] * (1 + (recent 3m growth - prior-year same-window 3m growth))
      - WeightedLagBlend (dynamic)   -> w3*lag3 + w6*lag6 + w12*lag12 (weights learned on TRAIN ONLY)
      - OLS (log-space)              -> log1p(y) ~ [1, lag3, lag6, lag12] (closed-form)
5) Validates over the last 6 months with SMAPE.
   - Seasonal models always attempt when the required windows exist.
   - Lag models evaluate only if lag rows exist (no hard "InsufficientHistory" stop).
6) Selects the best (lowest SMAPE) model per client and makes a recursive 15-month forecast.
7) Writes:
      - Desktop/val_smape_by_client_model_YYYYMMDD_HHMM.csv
      - Desktop/forecasts_fx_YYYYMMDD_HHMM.csv  (schema: fx_date, client_id, fx_vol, fx_id, fx_status, load_ts)
         * Only forecast rows; you’ll join to actuals later.

Notes
-----
- 'fx_id' uses underscores: <client_id>_<model_used>_<timestamp>.
- 'fx_status' is "forecast".
- We cap the SeasonalNaive3mDiffGR growth-difference contribution to +/- 25% by default (tunable).
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd

# ==============================
# --------- CONFIGURE ----------
# ==============================
FILE_PATH = "wd.xlsx"                                   # Excel is in the SAME folder as this script
DATE_COL = "MonthOfOrder"
CLIENT_COL = "client_id"
TARGET_COL = "TotalOrders"

VAL_MONTHS = 6                                          # size of the validation window (last N months)
HORIZON = 15                                            # recursive forecast horizon (months ahead)
CAP_DIFF = 0.25                                         # cap for SeasonalNaive3mDiffGR growth-diff (+/- 25%)
SHRINK_WLB = 0.30                                       # shrink dynamic WLB weights toward equal weights (0..1)

# Output directory (Desktop) + timestamped filenames
OUT_DIR = "/Users/jhonnatangonzalez/Desktop"
STAMP = datetime.now().strftime("%Y%m%d_%H%M")
VAL_PATH = os.path.join(OUT_DIR, f"val_smape_by_client_model_{STAMP}.csv")
FX_PATH  = os.path.join(OUT_DIR, f"forecasts_fx_{STAMP}.csv")


# ==============================
# --------- UTILITIES ----------
# ==============================
def smape(y_true, y_pred) -> float:
    """
    Symmetric MAPE (%) with 0/0 -> 0.
    SMAPE = mean( |y - yhat| / ((|y| + |yhat|)/2) ) * 100
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    out = np.where(denom == 0.0, 0.0, np.abs(y_true - y_pred) / denom)
    return float(np.mean(out) * 100.0)


def ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Parse DATE_COL to datetime (coerce errors to NaT)."""
    d = df.copy()
    if not np.issubdtype(d[date_col].dtype, np.datetime64):
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    return d


def add_lags_3_6_12(df: pd.DataFrame, group_cols, target_col: str) -> pd.DataFrame:
    """
    Add only lag3, lag6, lag12 (nothing else).
    We must sort by group + date before shifting.
    """
    d = df.sort_values(group_cols + [DATE_COL]).copy()
    for L in (3, 6, 12):
        d[f"{target_col}_lag{L}"] = d.groupby(group_cols, dropna=False)[target_col].shift(L)
    return d


# ==============================
# --- SEASONAL GROWTH LOGIC ----
# ==============================
def geom_mean_3m_growth(window_vals: pd.Series) -> float:
    """
    Geometric mean of 3 monthly ratios across 4 points: [t-3, t-2, t-1, t]
    G = ((y_t/y_{t-1})*(y_{t-1}/y_{t-2})*(y_{t-2}/y_{t-3}))^(1/3) - 1
    An epsilon prevents divide-by-zero edge cases.
    """
    y = pd.Series(window_vals, dtype=float)
    if y.size < 4 or y.isna().any():
        return 0.0
    eps = 1e-9
    r1 = (y.iloc[-1] + eps) / (y.iloc[-2] + eps)
    r2 = (y.iloc[-2] + eps) / (y.iloc[-3] + eps)
    r3 = (y.iloc[-3] + eps) / (y.iloc[-4] + eps)
    gm = (r1 * r2 * r3) ** (1.0 / 3.0) - 1.0
    if not np.isfinite(gm):
        return 0.0
    return float(gm)


def seasonal_naive(series: pd.Series, t_idx: int) -> float:
    """Plain seasonal naive: forecast equals y[t-12]."""
    return float(series.iloc[t_idx - 12])


def seasonal_naive_gr(series: pd.Series, t_idx: int) -> float:
    """
    Seasonal naive with *current* 3-month growth:
      yhat_t = y[t-12] * (1 + G_recent)
      where G_recent is 3-month geometric mean growth over [t-3..t].
    """
    base = float(series.iloc[t_idx - 12])
    recent_window = series.iloc[(t_idx - 3):(t_idx + 1)]
    g_recent = geom_mean_3m_growth(recent_window)
    return max(base * (1.0 + g_recent), 0.0)


def seasonal_naive_3m_diff_gr(series: pd.Series, t_idx: int, cap: float) -> float:
    """
    Seasonal naive with *difference of growth* at the same seasonal window:
      yhat_t = y[t-12] * (1 + (G_recent - G_prior))
        G_recent: 3m growth over [t-3..t]
        G_prior:  3m growth over [t-15..t-12] (the same seasonal window last year)
    We cap the delta to +/- 'cap' to avoid extreme swings.
    """
    base = float(series.iloc[t_idx - 12])
    recent_window = series.iloc[(t_idx - 3):(t_idx + 1)]
    prior_window  = series.iloc[(t_idx - 15):(t_idx - 11)]
    g_recent = geom_mean_3m_growth(recent_window)
    g_prior  = geom_mean_3m_growth(prior_window)
    delta = float(np.clip(g_recent - g_prior, -cap, cap))
    return max(base * (1.0 + delta), 0.0)


# ==============================
# --- DYNAMIC WLB WEIGHTING ----
# ==============================
def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute Pearson correlation with degeneracy checks."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(c):
        return 0.0
    return float(abs(c))


def compute_dynamic_wlb_weights(train_df: pd.DataFrame, target_col: str, shrink=SHRINK_WLB) -> dict:
    """
    Train-only dynamic weights for WeightedLagBlend using absolute Pearson correlation.
    We then shrink toward equal weights for stability.

    Returns {'w3':..., 'w6':..., 'w12':...} summing to 1.
    """
    feats = [f"{target_col}_lag3", f"{target_col}_lag6", f"{target_col}_lag12"]
    d = train_df.dropna(subset=feats + [target_col]).copy()
    if d.empty:
        return {"w3": 1/3, "w6": 1/3, "w12": 1/3}

    s3  = _safe_corr(d[target_col].values, d[f"{target_col}_lag3"].values)
    s6  = _safe_corr(d[target_col].values, d[f"{target_col}_lag6"].values)
    s12 = _safe_corr(d[target_col].values, d[f"{target_col}_lag12"].values)

    raw = np.array([s3, s6, s12], float)
    base = raw / raw.sum() if raw.sum() > 0 else np.array([1/3, 1/3, 1/3], float)

    eq = np.array([1/3, 1/3, 1/3], float)
    w = (1 - shrink) * base + shrink * eq
    w = np.clip(w, 0.05, 0.85)  # avoid extreme dominance
    w = w / w.sum()
    return {"w3": float(w[0]), "w6": float(w[1]), "w12": float(w[2])}


# ==============================
# ------ OLS (LOG SPACE) -------
# ==============================
def ols_logspace_fit_predict(train_X: np.ndarray, train_y: np.ndarray, val_X: np.ndarray):
    """
    Closed-form OLS in log1p space:
        log1p(y) ~ [1, lag3, lag6, lag12]
    Returns (predictions in original scale via expm1, beta).
    """
    Y = np.log1p(train_y.astype(float)).reshape(-1, 1)
    X = np.asarray(train_X, dtype=float)
    Xd = np.c_[np.ones(X.shape[0]), X]
    beta = np.linalg.pinv(Xd.T @ Xd) @ Xd.T @ Y
    yhat_log = (np.c_[np.ones(val_X.shape[0]), np.asarray(val_X, dtype=float)] @ beta).ravel()
    yhat = np.expm1(yhat_log)
    yhat[yhat < 0.0] = 0.0
    return yhat, beta


# ==============================
# -------- VALIDATION ----------
# ==============================
def validate_models_last6(df_client: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate the last 6 months SMAPE for:
      - SeasonalNaive
      - SeasonalNaiveGR
      - SeasonalNaive3mDiffGR
      - WeightedLagBlend (dynamic)  [only if lag rows exist]
      - OLS (log-space)             [only if lag rows exist]

    Seasonal models use the full series (no dropping early rows).
    Lag models evaluate only where lag3/6/12 rows exist; otherwise they’re skipped.
    Returns a dataframe sorted by SMAPE_6m asc: columns [model, SMAPE_6m, (optional) w3, w6, w12].
    """
    # Keep/clean only the columns we need
    d = df_client[[DATE_COL, CLIENT_COL, TARGET_COL]].sort_values(DATE_COL).reset_index(drop=True)
    d = ensure_datetime(d, DATE_COL)

    # Full series of target for seasonal methods
    full_y = d[TARGET_COL].astype(float).reset_index(drop=True)

    # Build absolute indices for the last VAL_MONTHS months
    idx_map = pd.Series(np.arange(len(d)), index=d[DATE_COL].astype("datetime64[ns]"))
    val_dates = d[DATE_COL].iloc[-VAL_MONTHS:].values
    val_abs_idx = idx_map.loc[val_dates].values
    y_val = full_y.iloc[val_abs_idx].values

    # --- Seasonal predictions (attempt whenever windows available) ---
    preds_sn, preds_sng, preds_snd = [], [], []
    for t in val_abs_idx:
        t = int(t)
        # SeasonalNaive needs t-12
        preds_sn.append(seasonal_naive(full_y, t) if (t - 12) >= 0 else np.nan)
        # SeasonalNaiveGR needs t-12 and [t-3..t]
        preds_sng.append(seasonal_naive_gr(full_y, t) if (t - 12) >= 0 and (t - 3) >= 0 else np.nan)
        # SeasonalNaive3mDiffGR needs t-12, [t-3..t], and [t-15..t-12]
        preds_snd.append(seasonal_naive_3m_diff_gr(full_y, t, CAP_DIFF) if (t - 12) >= 0 and (t - 3) >= 0 and (t - 15) >= 0 else np.nan)

    rows = [
        {"model": "SeasonalNaive",         "SMAPE_6m": smape(y_val, preds_sn)},
        {"model": "SeasonalNaiveGR",       "SMAPE_6m": smape(y_val, preds_sng)},
        {"model": "SeasonalNaive3mDiffGR", "SMAPE_6m": smape(y_val, preds_snd)},
    ]

    # --- Lag-based predictions (only if lag rows exist) ---
    d_lag = add_lags_3_6_12(d, [CLIENT_COL], TARGET_COL)
    need_cols = [f"{TARGET_COL}_lag3", f"{TARGET_COL}_lag6", f"{TARGET_COL}_lag12", TARGET_COL]
    d_lag_ok = d_lag.dropna(subset=need_cols).copy()

    # Require at least VAL_MONTHS rows to align a lag-based val set
    if not d_lag_ok.empty and len(d_lag_ok) >= VAL_MONTHS + 1:
        train = d_lag_ok.iloc[:-VAL_MONTHS].copy()
        val   = d_lag_ok.iloc[-VAL_MONTHS:].copy()
        y_val_lag = val[TARGET_COL].values

        # WeightedLagBlend with dynamic weights (train only)
        wlb_w = compute_dynamic_wlb_weights(train, TARGET_COL, shrink=SHRINK_WLB)
        preds_wlb = (
            wlb_w["w3"]  * val[f"{TARGET_COL}_lag3"].values +
            wlb_w["w6"]  * val[f"{TARGET_COL}_lag6"].values +
            wlb_w["w12"] * val[f"{TARGET_COL}_lag12"].values
        )
        rows.append({
            "model": "WeightedLagBlend",
            "SMAPE_6m": smape(y_val_lag, preds_wlb),
            "w3": wlb_w["w3"], "w6": wlb_w["w6"], "w12": wlb_w["w12"],
        })

        # OLS log-space (lags 3/6/12)
        feat_cols = [f"{TARGET_COL}_lag3", f"{TARGET_COL}_lag6", f"{TARGET_COL}_lag12"]
        yhat_ols, _ = ols_logspace_fit_predict(train[feat_cols].values, train[TARGET_COL].values, val[feat_cols].values)
        rows.append({"model": "OLS", "SMAPE_6m": smape(y_val_lag, yhat_ols)})

    out = pd.DataFrame(rows).dropna(subset=["SMAPE_6m"])
    return out.sort_values("SMAPE_6m").reset_index(drop=True)


# ==============================
# -------- FORECASTING ---------
# ==============================
def recursive_forecast(df_client: pd.DataFrame, model_name: str, horizon: int, wlb_weights=None) -> pd.DataFrame:
    """
    Make a recursive 'horizon'-month forecast using the selected model.
    At each step, append the prediction so subsequent steps can use it
    (for lags or growth windows).
    Returns a dataframe with columns [DATE_COL, "Predicted_TotalOrders"].
    """
    d = df_client[[DATE_COL, CLIENT_COL, TARGET_COL]].sort_values(DATE_COL).reset_index(drop=True)
    d = ensure_datetime(d, DATE_COL)

    y = d[TARGET_COL].astype(float).tolist()                # working series
    last_date = pd.to_datetime(d[DATE_COL]).iloc[-1]        # last actual month

    # Pre-fit OLS on all lag-usable rows if OLS is chosen
    beta = None
    if model_name == "OLS":
        base = add_lags_3_6_12(d, [CLIENT_COL], TARGET_COL).dropna()
        if not base.empty:
            feat_cols = [f"{TARGET_COL}_lag3", f"{TARGET_COL}_lag6", f"{TARGET_COL}_lag12"]
            Y = np.log1p(base[TARGET_COL].astype(float)).values.reshape(-1, 1)
            X = np.asarray(base[feat_cols].values, dtype=float)
            Xd = np.c_[np.ones(X.shape[0]), X]
            beta = np.linalg.pinv(Xd.T @ Xd) @ Xd.T @ Y
        else:
            # If we have no usable lag rows, fall back to SeasonalNaive during recursion.
            model_name = "SeasonalNaive"

    out_dates, out_vals = [], []
    for h in range(1, horizon + 1):
        t = len(y)  # index being predicted (0-based)
        if t - 12 < 0:
            # If the series is shorter than 12 points (unlikely for your data), fallback to last observed
            yhat = y[-1]
        else:
            if model_name == "SeasonalNaive":
                yhat = seasonal_naive(pd.Series(y), t)
            elif model_name == "SeasonalNaiveGR":
                yhat = seasonal_naive_gr(pd.Series(y), t)
            elif model_name == "SeasonalNaive3mDiffGR":
                yhat = seasonal_naive_3m_diff_gr(pd.Series(y), t, CAP_DIFF)
            elif model_name == "WeightedLagBlend":
                if t - 3 < 0 or t - 6 < 0 or t - 12 < 0:
                    yhat = y[-1]
                else:
                    w3  = (wlb_weights or {}).get("w3", 1/3)
                    w6  = (wlb_weights or {}).get("w6", 1/3)
                    w12 = (wlb_weights or {}).get("w12", 1/3)
                    yhat = w3 * y[t-3] + w6 * y[t-6] + w12 * y[t-12]
            elif model_name == "OLS" and beta is not None:
                if t - 3 < 0 or t - 6 < 0 or t - 12 < 0:
                    yhat = y[-1]
                else:
                    row = np.array([y[t-3], y[t-6], y[t-12]], dtype=float)
                    yhat_log = float((np.r_[1.0, row] @ beta).item())
                    yhat = float(np.expm1(yhat_log))
            else:
                yhat = y[-1]

        yhat = max(float(yhat), 0.0)  # prevent negatives
        y.append(yhat)

        # Next calendar month in YYYY-MM-01 format
        fdate = (last_date + pd.DateOffset(months=h)).to_period("M").to_timestamp()
        out_dates.append(fdate)
        out_vals.append(yhat)

    return pd.DataFrame({DATE_COL: out_dates, "Predicted_TotalOrders": out_vals})


# ==============================
# ------------- MAIN -----------
# ==============================
def main():
    # Ensure input and output locations exist
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Input file not found in this folder: {FILE_PATH}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Read ONLY the columns we need (MonthOfOrder, client_id, TotalOrders)
    df = pd.read_excel(FILE_PATH, usecols=[DATE_COL, CLIENT_COL, TARGET_COL])
    df = ensure_datetime(df, DATE_COL)

    # Containers for saving
    val_rows = []     # per-client validation tables
    fx_rows  = []     # final forecasts in fx_* schema

    # Iterate by client
    for cid, g in df.groupby(CLIENT_COL):
        g = g.sort_values(DATE_COL).reset_index(drop=True).copy()

        # ---- VALIDATE (last 6 months SMAPE) ----
        val_table = validate_models_last6(g)

        # Choose the best model (fallback to SeasonalNaive if needed)
        if val_table.empty or val_table["SMAPE_6m"].isna().all():
            best_model = "SeasonalNaive"
            best_smape = np.nan
            wlb_weights = {"w3": 1/3, "w6": 1/3, "w12": 1/3}
        else:
            best_row = val_table.loc[val_table["SMAPE_6m"].idxmin()]
            best_model = str(best_row["model"])
            best_smape = float(best_row["SMAPE_6m"])

            # Pull WLB weights if the best model is WLB (for consistent forecasting)
            if best_model == "WeightedLagBlend" and {"w3","w6","w12"}.issubset(val_table.columns):
                wlb_weights = {
                    "w3": float(best_row.get("w3", 1/3)),
                    "w6": float(best_row.get("w6", 1/3)),
                    "w12": float(best_row.get("w12", 1/3)),
                }
            else:
                # Optionally compute weights for logging consistency (not strictly needed otherwise)
                d_lag = add_lags_3_6_12(g, [CLIENT_COL], TARGET_COL).dropna()
                if not d_lag.empty and len(d_lag) >= VAL_MONTHS + 1:
                    train_only = d_lag.iloc[:-VAL_MONTHS].copy()
                    wlb_weights = compute_dynamic_wlb_weights(train_only, TARGET_COL, shrink=SHRINK_WLB)
                else:
                    wlb_weights = {"w3": 1/3, "w6": 1/3, "w12": 1/3}

        # Keep the validation table (tagged with client)
        if not val_table.empty:
            tmp = val_table.copy()
            tmp.insert(0, CLIENT_COL, cid)
            val_rows.append(tmp)

        # ---- FORECAST (15 months ahead, recursive) ----
        fcst = recursive_forecast(g, best_model, HORIZON, wlb_weights=wlb_weights)
        # Build fx_* rows as requested (ONLY forecasts; no actuals)
        # fx_date: YYYY-MM-01; client_id: cid; fx_vol: predicted TotalOrders
        # fx_id: <client_id>_<model_used>_<timestamp>  (underscores)
        # fx_status: "forecast"; load_ts: current timestamp "YYYY-MM-DD HH:MM:SS"
        fx_id = f"{cid}_{best_model}_{STAMP}".replace(" ", "_")
        load_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fx_rows.append(pd.DataFrame({
            "fx_date":   fcst[DATE_COL].dt.strftime("%Y-%m-01"),
            "client_id": cid,
            "fx_vol":    fcst["Predicted_TotalOrders"].astype(float),
            "fx_id":     fx_id,
            "fx_status": "forecast",
            "load_ts":   load_ts,
        }))

        # Console summary
        extra = ""
        if best_model == "WeightedLagBlend":
            extra = f" | WLB weights (w3,w6,w12)=({wlb_weights['w3']:.2f},{wlb_weights['w6']:.2f},{wlb_weights['w12']:.2f})"
        print(f"[{cid}] best_model={best_model} | SMAPE_6m={best_smape:.2f}{extra}")

    # ---- WRITE OUTPUTS ----
    if val_rows:
        pd.concat(val_rows, ignore_index=True).to_csv(VAL_PATH, index=False)
    if fx_rows:
        pd.concat(fx_rows, ignore_index=True).to_csv(FX_PATH, index=False)

    print("\nSaved files on Desktop:")
    print("  -", VAL_PATH if val_rows else "(no validation summary written)")
    print("  -", FX_PATH  if fx_rows  else "(no forecasts file written)")


# Entry point
if __name__ == "__main__":
    main()
