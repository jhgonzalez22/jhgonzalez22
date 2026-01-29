```python
"""
CX Platforms Phone Forecast — v2.5.1 (Rolling CV + Driver Lag Search + Seasonality + Direct Multi-Horizon + Hybrids)
===============================================================================================================
Purpose:
  Forecasts monthly phone call volumes using a tournament of statistical + ML models with:
    - Rolling-origin (walk-forward) evaluation (multiple folds)
    - Driver selection using lag-search + stability on last 24 months (Spearman correlation)
    - Seasonality features (month_of_year + sin/cos encoding)
    - Two-stage approach: forecast RATE (Offered per Business Day) then convert to monthly volume
    - Ratio-on-rate option (rate/orders) for stability
    - Direct multi-horizon forecasting (1..H months) for supported models (XGB + Ridge)
    - Hybrid models that blend a seasonal/stat baseline with driver-based or ML models

Models in the Tournament:
  Baselines:
    - WLB1: Weighted Lag Blend (lag1/2/3)
    - WLB2: Weighted Lag Blend (lag1/3/6)
    - SeasonalNaive: lag-12
    - SeasonalNaiveGR: lag-12 * growth-rate (recent 3mo growth proxy)
    - Native3m: rolling 3-month mean
  Driver models:
    - RatioRateModel: pred_rate = avg(rate / orders_lag) * orders_lag
  Direct (multi-horizon) models:
    - XGB_DirectH: 1 model per horizon, predicts future rate directly
    - Ridge_DirectH: 1 model per horizon, predicts future rate directly
  Hybrid models (simple, stable blends):
    - Hybrid_WLB1_Ratio: 70% WLB1 + 30% RatioRateModel
    - Hybrid_SNaiveGR_Ratio: 60% SeasonalNaiveGR + 40% RatioRateModel
    - Hybrid_WLB2_XGB: 60% WLB2 + 40% XGB_DirectH
    - Hybrid_SNaiveGR_XGB: 50% SeasonalNaiveGR + 50% XGB_DirectH

Selection:
  - Rolling-origin evaluation:
      Train up to t, validate t+1..t+6
      Slide forward 3 months
  - Winner chosen by lowest AVERAGE SMAPE across folds

Outputs:
  - forecast_results_phone.csv (volumes)
  - model_eval_debug_phone.csv (tournament results)
  - modeling_data_phone.csv (merged modeling frame)
  - xgb_feature_importance_phone.csv (basic feature importance snapshot)
  - model_summaries_phone.txt (human-readable summary)
  - statistical_tests_phone.csv (run metadata / sanity)

Author: Automation Team
Last Updated: 2026-01-29
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

# Ensure this points to your automation folder
sys.path.append(r"C:\WFM_Scripting\Automation")
from scripthelper import Config, Logger, BigQueryManager, EmailManager

# ------------------- 1) Configuration & File Paths -------------------
config = Config(rpt_id=283)
logger = Logger(config)
email_manager = EmailManager(config)
bq_manager = BigQueryManager(config)

SQL_QUERY_PATH = r"C:\WFM_Scripting\Forecasting\GBQ - Non-Tax Platform Phone Timeseries by Month.sql"

LOCAL_CSV         = r"C:\WFM_Scripting\forecast_results_phone.csv"
AUDIT_CSV         = r"C:\WFM_Scripting\model_eval_debug_phone.csv"
STATS_CSV         = r"C:\WFM_Scripting\statistical_tests_phone.csv"
XGB_VIZ_CSV       = r"C:\WFM_Scripting\xgb_feature_importance_phone.csv"
MODELING_DATA_CSV = r"C:\WFM_Scripting\modeling_data_phone.csv"
SUMMARIES_FILE    = r"C:\WFM_Scripting\model_summaries_phone.txt"

# ------------------- 2) Forecasting Parameters -------------------
BACKCAST_START = "2025-01-01"   # used for output range (backcast window)
HORIZON = 15                   # forecast horizon (months ahead)

# Rolling-origin evaluation params
VAL_H = 6                      # validate next 6 months per fold
SLIDE = 3                      # slide forward 3 months between folds
MIN_TRAIN = 18                 # minimum training points before allowing a fold

# Driver selection params
DRIVER_LAGS = [0, 1, 2, 3, 6, 12]
DRIVER_WINDOW = 24             # last 24 months window for stability test
MIN_DRIVER_SAMPLES = 12
MIN_DRIVER_ABS_CORR = 0.25     # minimum abs Spearman corr to enable driver models confidently

STAMP = datetime.now()

# Parent driver map (roll sub-clients into a shared workload bucket)
DRIVER_MAP = {
    "FNC - CMS": "FNC",
    "FNC - Ports": "FNC",
    "Mercury Integrations": "Mercury",
    "Appraisal Scope": "Appraisal Scope"
}

# ------------------- 3) Utility / Safety Functions -------------------
def to_float_array(x, name="array"):
    """
    Robust conversion to float array with no NaN/Inf.
    Handles pandas nullable arrays (masked / Float64) which can throw conversion errors.
    """
    if isinstance(x, (pd.Series, pd.Index)):
        s = pd.to_numeric(x, errors="coerce")
        try:
            arr = s.to_numpy(dtype=float, na_value=0.0)  # best for nullable dtypes
        except TypeError:
            arr = np.asarray(s.fillna(0.0).astype(float).values, dtype=float)
    else:
        arr = np.asarray(x, dtype=float)

    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def smape(actual, forecast):
    """
    Symmetric MAPE in %.
    """
    a = to_float_array(actual, "actual")
    f = to_float_array(forecast, "forecast")
    denom = (np.abs(a) + np.abs(f)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(a - f) / denom) * 100.0)


def add_seasonality_features(df, date_col="date"):
    """
    Adds month-of-year and cyclical encodings (sin/cos) to capture seasonality smoothly.
    """
    df = df.copy()
    df["month_of_year"] = df[date_col].dt.month.astype(int)
    df["sin_moy"] = np.sin(2 * np.pi * df["month_of_year"] / 12.0)
    df["cos_moy"] = np.cos(2 * np.pi * df["month_of_year"] / 12.0)
    return df


def safe_div(n, d):
    """
    Safe division that returns 0 when denominator is 0/NaN and cleans infs.
    """
    n = pd.to_numeric(n, errors="coerce")
    d = pd.to_numeric(d, errors="coerce").replace(0, np.nan)
    out = n / d
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def rolling_origin_folds(n, val_h=6, slide=3, min_train=18):
    """
    Generate rolling-origin folds for series length n.

    Fold definition:
      train = [0 : train_end)
      valid = [train_end : train_end + val_h)

    Then slide train_end forward by `slide`.
    """
    folds = []
    train_end = min_train
    while (train_end + val_h) <= n:
        folds.append((train_end, train_end, train_end + val_h))
        train_end += slide
    return folds


def choose_best_driver_lag(c_df):
    """
    Driver selection with lag-search + stability on last 24 months.
    Uses Spearman correlation between target_rate and TotalOrders lag.

    Returns:
      best_lag (int or None), best_corr (float)
    """
    tmp = c_df.sort_values("date").tail(DRIVER_WINDOW).copy()

    tmp["target_rate"] = pd.to_numeric(tmp["target_rate"], errors="coerce")
    tmp["TotalOrders"] = pd.to_numeric(tmp["TotalOrders"], errors="coerce")

    best_lag = None
    best_corr = 0.0

    for lag in DRIVER_LAGS:
        orders_lag = tmp["TotalOrders"] if lag == 0 else tmp["TotalOrders"].shift(lag)
        use = pd.DataFrame({"y": tmp["target_rate"], "x": orders_lag}).dropna()

        if len(use) < MIN_DRIVER_SAMPLES:
            continue

        corr = use["y"].corr(use["x"], method="spearman")
        if pd.isna(corr):
            continue

        if abs(corr) > abs(best_corr):
            best_corr = float(corr)
            best_lag = int(lag)

    return best_lag, float(best_corr)


# ------------------- 4) Model Helpers: Validation Predictions -------------------
def preds_wlb1(valid_df):
    """WLB1: 0.6*lag1 + 0.2*lag2 + 0.2*lag3"""
    return to_float_array((0.6 * valid_df["lag_1"]) + (0.2 * valid_df["lag_2"]) + (0.2 * valid_df["lag_3"]))


def preds_wlb2(valid_df):
    """WLB2: 0.6*lag1 + 0.25*lag3 + 0.15*lag6"""
    return to_float_array((0.6 * valid_df["lag_1"]) + (0.25 * valid_df["lag_3"]) + (0.15 * valid_df["lag_6"]))


def preds_seasonal_naive(train_df, valid_df):
    """
    Seasonal Naive: predict = lag-12 for each validation row (already precomputed as lag_12).
    """
    return to_float_array(valid_df["lag_12"])


def preds_seasonal_naive_gr(train_df, valid_df):
    """
    Seasonal Naive with Growth Rate (GR):
      pred = lag-12 * gr
    Growth proxy:
      gr = (last_train / train[-4]) if possible else 1.0
    """
    train_y = pd.to_numeric(train_df["target_rate"], errors="coerce").dropna()
    if len(train_y) >= 4 and train_y.iloc[-4] != 0:
        gr = float(train_y.iloc[-1] / train_y.iloc[-4])
    else:
        gr = 1.0
    return to_float_array(valid_df["lag_12"]) * gr


def preds_native3m(train_df, valid_df):
    """
    Native3m: 3-month rolling mean shifted by 1 (validation already has lagged target via lag_1 etc.)
    We'll approximate validation prediction as mean(lag1, lag2, lag3) for each validation row.
    """
    return to_float_array((valid_df["lag_1"] + valid_df["lag_2"] + valid_df["lag_3"]) / 3.0)


def fit_ratio_rate_model(train_df, driver_series_name):
    """
    Ratio-on-rate:
      ratio = target_rate / driver
      avg_ratio = mean(ratio) on training (last 12 valid points if possible)
    """
    y = pd.to_numeric(train_df["target_rate"], errors="coerce")
    d = pd.to_numeric(train_df[driver_series_name], errors="coerce").replace(0, np.nan)

    ratio = (y / d).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return 0.0

    # Use tail(12) for recency stability if available
    avg_ratio = float(ratio.tail(12).mean())
    if not np.isfinite(avg_ratio):
        avg_ratio = 0.0
    return avg_ratio


def preds_ratio_rate_model(valid_df, driver_series_name, avg_ratio):
    """
    Predict rate from driver:
      pred_rate = avg_ratio * driver_valid
    """
    drv = pd.to_numeric(valid_df[driver_series_name], errors="coerce").fillna(0.0)
    return to_float_array(avg_ratio * drv)


def fit_direct_models(train_df, feature_cols, horizon, model_type="xgb"):
    """
    Direct multi-horizon models:
      For each horizon h in 1..horizon:
        y_h = target_rate shifted -h
        fit model on rows where y_h is known
    """
    models = {}
    base = train_df.copy()

    base[feature_cols] = base[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    base["target_rate"] = pd.to_numeric(base["target_rate"], errors="coerce")

    for h in range(1, horizon + 1):
        base[f"y_h{h}"] = base["target_rate"].shift(-h)
        fit_df = base.dropna(subset=[f"y_h{h}"] + feature_cols)

        if len(fit_df) < 12:
            continue

        X = fit_df[feature_cols].values
        y = fit_df[f"y_h{h}"].astype(float).values

        if model_type == "xgb":
            mod = XGBRegressor(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                reg_lambda=1.0,
            )
        else:
            mod = Ridge(alpha=2.0, random_state=42)

        mod.fit(X, y)
        models[h] = mod

    return models


def predict_direct_from_origin(models_by_h, origin_row_df, feature_cols, horizon):
    """
    Predict horizons 1..H from ONE origin row (last train row).
    """
    Xo = origin_row_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    preds = []
    for h in range(1, horizon + 1):
        mod = models_by_h.get(h)
        if mod is None:
            preds.append(np.nan)
        else:
            preds.append(float(mod.predict(Xo)[0]))

    arr = np.array(preds, dtype=float)
    if np.all(np.isnan(arr)):
        arr = np.zeros(horizon, dtype=float)
    else:
        mean_val = float(np.nanmean(arr))
        arr = np.where(np.isnan(arr), mean_val, arr)

    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


# ------------------- 5) Forecast Generation Helpers (H-month horizon arrays) -------------------
def recursive_forecast_wlb(last_vals, horizon, mode="wlb1"):
    """
    Recursive forecast for WLBs using predicted rates as they become available.

    last_vals: list/array of historical rates in chronological order (must include enough history).
    mode:
      - 'wlb1' uses last1,last2,last3
      - 'wlb2' uses last1,last3,last6
    """
    hist = list(map(float, last_vals))
    preds = []

    for _ in range(horizon):
        # Fetch needed lags from the evolving history list
        lag1 = hist[-1] if len(hist) >= 1 else 0.0
        lag2 = hist[-2] if len(hist) >= 2 else lag1
        lag3 = hist[-3] if len(hist) >= 3 else lag1
        lag6 = hist[-6] if len(hist) >= 6 else lag3

        if mode == "wlb2":
            p = (0.6 * lag1) + (0.25 * lag3) + (0.15 * lag6)
        else:
            p = (0.6 * lag1) + (0.2 * lag2) + (0.2 * lag3)

        p = max(0.0, float(p))
        preds.append(p)
        hist.append(p)

    return np.array(preds, dtype=float)


def seasonal_naive_horizon(series_by_date, future_dates, gr=1.0):
    """
    Seasonal naive horizon prediction:
      pred(t) = y(t-12) * gr
    If y(t-12) is unavailable, fallback to last observed value.

    series_by_date: pd.Series indexed by date (monthly start)
    future_dates: list-like of pd.Timestamp
    """
    s = series_by_date.copy().sort_index()
    last_val = float(pd.to_numeric(s.dropna().iloc[-1], errors="coerce")) if not s.dropna().empty else 0.0
    preds = []
    for d in future_dates:
        back = d - pd.DateOffset(months=12)
        if back in s.index and pd.notna(s.loc[back]):
            base = float(s.loc[back])
        else:
            base = last_val
        p = max(0.0, base * float(gr))
        preds.append(p)
        s.loc[d] = p  # allow chaining for horizons > 12 if needed
    return np.array(preds, dtype=float)


def native3m_horizon(last_vals, horizon):
    """
    Recursive 3-month mean forecast.
    """
    hist = list(map(float, last_vals))
    preds = []
    for _ in range(horizon):
        if len(hist) >= 3:
            p = float(np.mean(hist[-3:]))
        elif len(hist) > 0:
            p = float(np.mean(hist))
        else:
            p = 0.0
        p = max(0.0, p)
        preds.append(p)
        hist.append(p)
    return np.array(preds, dtype=float)


# ------------------- 6) Main Execution Pipeline -------------------
try:
    logger.info("Starting Pipeline v2.5.1 - Rolling CV + Driver Lag Search + Direct Multi-Horizon + Hybrids...")

    # ---------- Step A: FRED (History + FX) ----------
    fred_sql = """
    SELECT Date as date, UNRATE, HSN1F, FEDFUNDS, MORTGAGE30US
    FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.fred`
    UNION ALL
    SELECT Date as date, UNRATE, HSN1F, FEDFUNDS, MORTGAGE30US
    FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.fred_fx`
    QUALIFY ROW_NUMBER() OVER(PARTITION BY Date ORDER BY Forecast_Date DESC) = 1
    ORDER BY date
    """
    df_fred = bq_manager.run_gbq_sql(fred_sql, return_dataframe=True)
    if not isinstance(df_fred, pd.DataFrame) or df_fred.empty:
        raise ValueError("FRED query returned no data. Check fred/fred_fx tables or permissions.")

    df_fred["date"] = pd.to_datetime(df_fred["date"])
    fred_cols = ["UNRATE", "HSN1F", "FEDFUNDS", "MORTGAGE30US"]
    df_fred[fred_cols] = df_fred[fred_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill()

    # ---------- Step B: Workload Drivers (Orders) History + FX ----------
    order_hist_sql = """
    SELECT MonthOfOrder as date, client_id, TotalOrders
    FROM tax_clnt_svcs.view_cx_nontax_platforms_workload_drivers
    """
    order_fx_sql = """
    SELECT fx_date as date, client_id, fx_vol as TotalOrders
    FROM tax_clnt_svcs.cx_nontax_platforms_workload_drivers_fx
    WHERE fx_status = 'A'
    """
    df_ord_hist = bq_manager.run_gbq_sql(order_hist_sql, return_dataframe=True)
    df_ord_fx = bq_manager.run_gbq_sql(order_fx_sql, return_dataframe=True)

    if not isinstance(df_ord_hist, pd.DataFrame):
        df_ord_hist = pd.DataFrame(columns=["date", "client_id", "TotalOrders"])
    if not isinstance(df_ord_fx, pd.DataFrame):
        df_ord_fx = pd.DataFrame(columns=["date", "client_id", "TotalOrders"])

    df_orders = pd.concat([df_ord_hist, df_ord_fx], ignore_index=True)
    df_orders = df_orders.drop_duplicates(subset=["date", "client_id"], keep="last")
    df_orders["date"] = pd.to_datetime(df_orders["date"])
    df_orders["TotalOrders"] = pd.to_numeric(df_orders["TotalOrders"], errors="coerce").fillna(0.0).astype(float)

    # ---------- Step C: Base Calls (Phone) ----------
    df_calls = bq_manager.run_gbq_sql(SQL_QUERY_PATH, return_dataframe=True)
    if not isinstance(df_calls, pd.DataFrame) or df_calls.empty:
        raise ValueError("Base call query returned no data. Check SQL_QUERY_PATH output.")

    df_calls["date"] = pd.to_datetime(df_calls["date"])
    df_calls["Total_Offered"] = pd.to_numeric(df_calls["Total_Offered"], errors="coerce").fillna(0.0).astype(float)
    df_calls["business_day_count"] = pd.to_numeric(df_calls["business_day_count"], errors="coerce").fillna(21.0).astype(float)

    # Target RATE per business day
    df_calls["target_rate"] = np.where(
        df_calls["business_day_count"] > 0,
        df_calls["Total_Offered"] / df_calls["business_day_count"],
        0.0
    ).astype(float)

    # join_key aligns client call series to the appropriate driver bucket
    df_calls["join_key"] = df_calls["client_id"].map(DRIVER_MAP).fillna(df_calls["client_id"])

    # ---------- Step D: Master Merge ----------
    df = pd.merge(df_calls, df_fred, on="date", how="left")
    df = pd.merge(df, df_orders.rename(columns={"client_id": "join_key"}), on=["date", "join_key"], how="left")

    # Fill driver/market missing within each client series
    metric_cols = ["TotalOrders"] + fred_cols
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[metric_cols] = df.groupby("client_id")[metric_cols].ffill().bfill().fillna(0.0)

    # Seasonality features
    df = add_seasonality_features(df, date_col="date")

    # ---------- Step E: Lag Features ----------
    df = df.sort_values(["client_id", "date"])
    for l in [1, 2, 3, 6, 12]:
        df[f"lag_{l}"] = df.groupby("client_id")["target_rate"].shift(l)
        df[f"order_lag_{l}"] = df.groupby("client_id")["TotalOrders"].shift(l)
        df[f"mkt_lag_{l}"] = df.groupby("client_id")["MORTGAGE30US"].shift(l)

    # Ratio-on-rate metric (rate per order)
    df["rate_per_order"] = safe_div(df["target_rate"], df["TotalOrders"]).astype(float)

    # Save merged modeling data for inspection
    df.to_csv(MODELING_DATA_CSV, index=False)

    # Output containers
    forecast_results = []
    audit_log = []
    xgb_importance_rows = []

    # Features for direct models
    direct_feats = [
        "lag_1", "lag_3", "lag_6", "lag_12",
        "TotalOrders", "MORTGAGE30US", "UNRATE",
        "month_of_year", "sin_moy", "cos_moy"
    ]
    direct_feats = [f for f in direct_feats if f in df.columns]  # safety

    # ---------- Step F: Client Modeling Loop ----------
    for cid in df["client_id"].dropna().unique():
        c_df = df[df["client_id"] == cid].copy().sort_values("date")

        # Ensure enough history for rolling folds + lag-12 dependencies
        if c_df["target_rate"].dropna().shape[0] < (MIN_TRAIN + VAL_H + 12):
            logger.warning(f"Skipping {cid}: insufficient history (needs >= {MIN_TRAIN + VAL_H + 12} points ideally).")
            continue

        meta = c_df[["bu", "client", "groups"]].iloc[-1].to_dict()

        # ---------- F1) Driver selection (orders lag) ----------
        best_lag, best_corr = choose_best_driver_lag(c_df)
        # driver series name used in training/valid frames
        driver_series_name = "TotalOrders" if best_lag in (None, 0) else f"order_lag_{best_lag}"

        # If we don't have the lag column (rare), fall back to TotalOrders
        if driver_series_name not in c_df.columns:
            driver_series_name = "TotalOrders"

        driver_ok = (best_lag is not None) and (abs(best_corr) >= MIN_DRIVER_ABS_CORR)

        # ---------- F2) Rolling-origin evaluation ----------
        n = len(c_df)
        folds = rolling_origin_folds(n, val_h=VAL_H, slide=SLIDE, min_train=MIN_TRAIN)
        if not folds:
            logger.warning(f"Skipping {cid}: no folds generated (check MIN_TRAIN/VAL_H).")
            continue

        # Tournament candidates (include hybrids)
        model_names = [
            "WLB1", "WLB2",
            "SeasonalNaive", "SeasonalNaiveGR", "Native3m",
            "RatioRateModel",
            "XGB_DirectH", "Ridge_DirectH",
            "Hybrid_WLB1_Ratio", "Hybrid_SNaiveGR_Ratio",
            "Hybrid_WLB2_XGB", "Hybrid_SNaiveGR_XGB"
        ]

        fold_scores = {m: [] for m in model_names}

        # For audit transparency, keep per-fold winners too
        fold_winner_rows = []

        for (train_end, v_start, v_end) in folds:
            train_df = c_df.iloc[:train_end].copy()
            valid_df = c_df.iloc[v_start:v_end].copy()

            # Validation actuals (rate)
            y_val = to_float_array(valid_df["target_rate"])

            # Build baseline predictions for this fold
            preds = {}

            preds["WLB1"] = preds_wlb1(valid_df)
            preds["WLB2"] = preds_wlb2(valid_df)
            preds["SeasonalNaive"] = preds_seasonal_naive(train_df, valid_df)
            preds["SeasonalNaiveGR"] = preds_seasonal_naive_gr(train_df, valid_df)
            preds["Native3m"] = preds_native3m(train_df, valid_df)

            # Ratio-on-rate (only if driver relationship is decent, otherwise keep but expect it to lose)
            avg_ratio = 0.0
            if driver_ok and driver_series_name in train_df.columns:
                avg_ratio = fit_ratio_rate_model(train_df, driver_series_name)
                preds["RatioRateModel"] = preds_ratio_rate_model(valid_df, driver_series_name, avg_ratio)
            else:
                preds["RatioRateModel"] = np.zeros_like(y_val, dtype=float)

            # Direct models:
            # Train direct models on fold train_df and predict next VAL_H months from the ORIGIN = last train row
            # This matches the "train to t, validate t+1..t+6" idea.
            origin_row = train_df.iloc[[-1]].copy()

            # XGB Direct
            xgb_models = fit_direct_models(train_df, direct_feats, horizon=VAL_H, model_type="xgb")
            preds["XGB_DirectH"] = predict_direct_from_origin(xgb_models, origin_row, direct_feats, horizon=VAL_H)

            # Ridge Direct
            ridge_models = fit_direct_models(train_df, direct_feats, horizon=VAL_H, model_type="ridge")
            preds["Ridge_DirectH"] = predict_direct_from_origin(ridge_models, origin_row, direct_feats, horizon=VAL_H)

            # ---------------- Hybrids (blend RATE predictions) ----------------
            # Fixed blend weights (stable and easy to reason about)
            preds["Hybrid_WLB1_Ratio"] = (0.7 * preds["WLB1"]) + (0.3 * preds["RatioRateModel"])
            preds["Hybrid_SNaiveGR_Ratio"] = (0.6 * preds["SeasonalNaiveGR"]) + (0.4 * preds["RatioRateModel"])
            preds["Hybrid_WLB2_XGB"] = (0.6 * preds["WLB2"]) + (0.4 * preds["XGB_DirectH"])
            preds["Hybrid_SNaiveGR_XGB"] = (0.5 * preds["SeasonalNaiveGR"]) + (0.5 * preds["XGB_DirectH"])

            # Score each model on this fold
            fold_smape = {}
            for m in model_names:
                p = preds[m]
                # Defensive alignment
                if len(p) != len(y_val):
                    p = np.resize(p, y_val.shape)
                s = smape(y_val, p)
                fold_scores[m].append(s)
                fold_smape[m] = s

            # fold winner (for transparency)
            fold_winner = min(fold_smape, key=fold_smape.get)
            fold_winner_rows.append({
                "client_id": cid,
                "fold_train_end_idx": train_end,
                "fold_valid_start_idx": v_start,
                "fold_valid_end_idx": v_end,
                "fold_winner": fold_winner,
                "fold_winner_smape": float(fold_smape[fold_winner]),
                "driver_lag": best_lag,
                "driver_corr_spearman": best_corr
            })

        # Average SMAPE across folds
        avg_scores = {m: float(np.mean(fold_scores[m])) if fold_scores[m] else 9999.0 for m in model_names}
        winner = min(avg_scores, key=avg_scores.get)

        # Log model tournament summary for this client
        audit_log.append({
            "client_id": cid,
            "winner": winner,
            "winner_avg_smape": avg_scores[winner],
            "driver_lag": best_lag,
            "driver_corr_spearman": best_corr,
            "driver_series_name": driver_series_name,
            **{f"avg_smape_{m}": avg_scores[m] for m in model_names}
        })

        # Also append fold-level winner rows (so you can inspect stability)
        audit_log.extend(fold_winner_rows)

        # ---------- F3) Fit final models on FULL history and generate final horizon forecast ----------
        # We forecast RATE for the next HORIZON months, then convert to monthly volume using business_day_count.

        # Establish last historical date and future monthly dates
        last_date = c_df["date"].max()
        future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=HORIZON, freq="MS")

        # For conversion to volume, we need business_day_count for future months.
        # We will:
        #   - use df_calls if the month exists (sometimes your base query includes future calendar rows),
        #   - else approximate using numpy.busday_count as a fallback.
        def business_days_for_month(month_start_ts):
            ser = df_calls.loc[(df_calls["client_id"] == cid) & (df_calls["date"] == month_start_ts), "business_day_count"]
            if not ser.empty:
                v = float(pd.to_numeric(ser.iloc[0], errors="coerce") or 21.0)
                return v if v > 0 else 21.0
            # Fallback approximation
            start = month_start_ts.date()
            end = (month_start_ts + pd.offsets.MonthEnd(0)).date()
            bd = float(np.busday_count(start, end))  # end exclusive
            return bd if bd > 0 else 21.0

        # Build lookup tables for future drivers (orders) and market (mortgage, unrate)
        # Orders are bucketed by DRIVER_MAP (join_key)
        driver_key = DRIVER_MAP.get(cid, cid)
        orders_lookup = (
            df_orders[df_orders["client_id"] == driver_key]
            .set_index("date")["TotalOrders"]
            .to_dict()
        )

        fred_lookup = (
            df_fred.set_index("date")[["MORTGAGE30US", "UNRATE"]]
            .to_dict(orient="index")
        )

        # Historical target series for seasonal naive
        y_series = c_df.set_index("date")["target_rate"].copy()
        y_series = pd.to_numeric(y_series, errors="coerce").fillna(0.0)

        # Compute growth rate for SeasonalNaiveGR on full history
        y_hist = y_series.dropna()
        if len(y_hist) >= 4 and y_hist.iloc[-4] != 0:
            gr_full = float(y_hist.iloc[-1] / y_hist.iloc[-4])
        else:
            gr_full = 1.0

        # Build RATE forecasts for each base model (as arrays), so hybrids can be produced cleanly
        # 1) WLBs / Native3m use recursive rate forecasting from last values
        last_vals = list(y_series.values)
        rate_wlb1 = recursive_forecast_wlb(last_vals, HORIZON, mode="wlb1")
        rate_wlb2 = recursive_forecast_wlb(last_vals, HORIZON, mode="wlb2")
        rate_native3m = native3m_horizon(last_vals, HORIZON)

        # 2) Seasonal naive models use lag-12 values
        rate_snaive = seasonal_naive_horizon(y_series, future_dates, gr=1.0)
        rate_snaive_gr = seasonal_naive_horizon(y_series, future_dates, gr=gr_full)

        # 3) RatioRateModel uses best orders lag and avg ratio from full history
        #    We need driver series for future months. If lag>0, driver for month t is orders at t-lag.
        #    We'll build a driver_future array aligned to future_dates.
        if best_lag is None:
            best_lag = 0

        # avg_ratio computed from full history using driver series name
        if driver_series_name not in c_df.columns:
            c_df[driver_series_name] = c_df["TotalOrders"] if best_lag == 0 else c_df["TotalOrders"].shift(best_lag)
        avg_ratio_full = fit_ratio_rate_model(c_df.dropna(subset=["target_rate"]).copy(), driver_series_name)

        driver_future = []
        for d in future_dates:
            # if lag is L, driver at month d uses orders at (d - L months)
            d_src = d - pd.DateOffset(months=int(best_lag))
            ov = float(orders_lookup.get(d_src, 0.0))
            if ov == 0.0:
                # fallback: last known orders from history
                last_orders = pd.to_numeric(c_df["TotalOrders"], errors="coerce").dropna()
                ov = float(last_orders.iloc[-1]) if not last_orders.empty else 0.0
            driver_future.append(ov)

        rate_ratio = np.array([max(0.0, avg_ratio_full * float(v)) for v in driver_future], dtype=float)

        # 4) Direct multi-horizon models trained on full history; predict from origin=last historical row
        origin_row = c_df.iloc[[-1]].copy()

        # Ensure origin has contemporaneous drivers that are known at origin time
        # (These are already in c_df). For Direct models we predict horizons from origin.
        xgb_models_full = fit_direct_models(c_df, direct_feats, horizon=HORIZON, model_type="xgb")
        rate_xgb = predict_direct_from_origin(xgb_models_full, origin_row, direct_feats, horizon=HORIZON)

        ridge_models_full = fit_direct_models(c_df, direct_feats, horizon=HORIZON, model_type="ridge")
        rate_ridge = predict_direct_from_origin(ridge_models_full, origin_row, direct_feats, horizon=HORIZON)

        # Hybrids (final horizon arrays)
        rate_h_wlb1_ratio = (0.7 * rate_wlb1) + (0.3 * rate_ratio)
        rate_h_snaivegr_ratio = (0.6 * rate_snaive_gr) + (0.4 * rate_ratio)
        rate_h_wlb2_xgb = (0.6 * rate_wlb2) + (0.4 * rate_xgb)
        rate_h_snaivegr_xgb = (0.5 * rate_snaive_gr) + (0.5 * rate_xgb)

        # Map model name -> horizon array (RATE)
        rate_by_model = {
            "WLB1": rate_wlb1,
            "WLB2": rate_wlb2,
            "SeasonalNaive": rate_snaive,
            "SeasonalNaiveGR": rate_snaive_gr,
            "Native3m": rate_native3m,
            "RatioRateModel": rate_ratio,
            "XGB_DirectH": rate_xgb,
            "Ridge_DirectH": rate_ridge,
            "Hybrid_WLB1_Ratio": rate_h_wlb1_ratio,
            "Hybrid_SNaiveGR_Ratio": rate_h_snaivegr_ratio,
            "Hybrid_WLB2_XGB": rate_h_wlb2_xgb,
            "Hybrid_SNaiveGR_XGB": rate_h_snaivegr_xgb
        }

        # Feature importance snapshot for XGB (h=1 model only) if present
        try:
            mod_h1 = xgb_models_full.get(1)
            if mod_h1 is not None:
                imps = mod_h1.feature_importances_
                for feat, imp in zip(direct_feats, imps):
                    xgb_importance_rows.append({
                        "client_id": cid,
                        "horizon": 1,
                        "feature": feat,
                        "importance": float(imp)
                    })
        except Exception:
            pass

        # Final chosen rate horizon
        chosen_rates = rate_by_model.get(winner, rate_wlb1)
        chosen_rates = np.nan_to_num(np.asarray(chosen_rates, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        chosen_rates = np.maximum(0.0, chosen_rates)

        # Convert RATE -> VOLUME for each future month
        # volume = rate * business_day_count
        for i, d in enumerate(future_dates):
            biz_days = business_days_for_month(d)
            vol = int(max(0.0, float(chosen_rates[i])) * biz_days)

            forecast_results.append({
                "fx_date": d.date(),
                "bu": meta.get("bu"),
                "client": meta.get("client"),
                "client_id": cid,
                "groups": meta.get("groups"),
                "vol_type": "phone",
                "fx_vol": vol,
                "fx_rate": float(chosen_rates[i]),
                "fx_id": f"{winner}_v251",
                "fx_status": "A",
                "load_ts": STAMP
            })

    # ------------------- 7) Export Outputs -------------------
    if forecast_results:
        pd.DataFrame(forecast_results).to_csv(LOCAL_CSV, index=False)
        pd.DataFrame(audit_log).to_csv(AUDIT_CSV, index=False)

        if xgb_importance_rows:
            pd.DataFrame(xgb_importance_rows).to_csv(XGB_VIZ_CSV, index=False)

        # Stats / run metadata
        pd.DataFrame([{
            "run_ts": STAMP,
            "version": "v2.5.1",
            "backcast_start": BACKCAST_START,
            "horizon": HORIZON,
            "val_h": VAL_H,
            "slide": SLIDE,
            "min_train": MIN_TRAIN,
            "driver_window": DRIVER_WINDOW,
            "driver_lags_tested": str(DRIVER_LAGS),
            "notes": "Rolling CV + Spearman driver lag selection + seasonality + direct multi-horizon + hybrids"
        }]).to_csv(STATS_CSV, index=False)

        # Human-readable summary
        with open(SUMMARIES_FILE, "w") as f:
            f.write(f"Forecast Run: {STAMP}\n")
            f.write("=" * 60 + "\n\n")
            # Keep summary simple: only rows that are top-level client results (not fold rows)
            client_rows = [r for r in audit_log if "winner_avg_smape" in r]
            for r in client_rows:
                f.write(
                    f"Client: {r['client_id']:<22} | Winner: {r['winner']:<18} | "
                    f"Avg SMAPE: {r['winner_avg_smape']:.2f}% | "
                    f"DriverLag: {r['driver_lag']} | Corr: {r['driver_corr_spearman']:.2f}\n"
                )

        logger.info("✓ Pipeline Complete. Outputs saved to C:\\WFM_Scripting\\")
    else:
        logger.warning("No forecasts were generated (forecast_results empty).")

except Exception as e:
    # scripthelper handle_error re-raises after notifying
    email_manager.handle_error("V2.5.1 Failure", e, is_test=True)
```
Here’s a copy-paste “Gemini instructions” prompt you can use to have it build/maintain this exact kind of script. I wrote it like a spec + constraints so it won’t drift, and it tells Gemini what *must* be in the final output.

---

## Gemini build instructions (copy/paste)

Build a **single, runnable Python script** for forecasting **monthly phone call volumes** by client_id using a **tournament** of models. The script must be production-ready, heavily commented, and follow the workflow below. Do **not** return partial code—return the **entire script**.

### Environment / dependencies

* Python 3.x on Windows
* Uses my local module: `C:\WFM_Scripting\Automation\scripthelper.py`
* Must import: `Config, Logger, BigQueryManager, EmailManager` from `scripthelper`
* Uses: `pandas`, `numpy`, `xgboost`, `sklearn` (Ridge), and standard libs only.
* Must be robust to pandas **nullable dtypes** (Float64/Int64/masked arrays). Avoid dtype conversion errors.

### Paths / config

* `Config(rpt_id=283)`
* SQL query file for calls:

  * `SQL_QUERY_PATH = r"C:\WFM_Scripting\Forecasting\GBQ - Non-Tax Platform Phone Timeseries by Month.sql"`
* Output files:

  * `C:\WFM_Scripting\forecast_results_phone.csv`
  * `C:\WFM_Scripting\model_eval_debug_phone.csv`
  * `C:\WFM_Scripting\modeling_data_phone.csv`
  * `C:\WFM_Scripting\xgb_feature_importance_phone.csv`
  * `C:\WFM_Scripting\model_summaries_phone.txt`
  * `C:\WFM_Scripting\statistical_tests_phone.csv`

### Data inputs from BigQuery (must do these queries)

1. **Calls base** from SQL_QUERY_PATH; expects at least:

   * `date` (month start), `client_id`, `Total_Offered`, `business_day_count`, plus metadata columns `bu`, `client`, `groups`
2. **FRED** history + forecast union:

```sql
SELECT Date as date, UNRATE, HSN1F, FEDFUNDS, MORTGAGE30US
FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.fred`
UNION ALL
SELECT Date as date, UNRATE, HSN1F, FEDFUNDS, MORTGAGE30US
FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.fred_fx`
QUALIFY ROW_NUMBER() OVER(PARTITION BY Date ORDER BY Forecast_Date DESC) = 1
ORDER BY date
```

3. **Orders workload drivers** history + forecast union:

```sql
SELECT MonthOfOrder as date, client_id, TotalOrders
FROM tax_clnt_svcs.view_cx_nontax_platforms_workload_drivers
```

```sql
SELECT fx_date as date, client_id, fx_vol as TotalOrders
FROM tax_clnt_svcs.cx_nontax_platforms_workload_drivers_fx
WHERE fx_status = 'A'
```

### Normalization / target definition

* Convert to **rate per business day**:

  * `target_rate = Total_Offered / business_day_count` (if business_day_count <=0 then 0)
* Forecast **rate**, then convert back:

  * `monthly_volume = forecast_rate * business_day_count`

### Driver mapping

Include a mapping so some client series use shared driver buckets:

```python
DRIVER_MAP = {
  "FNC - CMS": "FNC",
  "FNC - Ports": "FNC",
  "Mercury Integrations": "Mercury",
  "Appraisal Scope": "Appraisal Scope"
}
```

* Calls should have `join_key = DRIVER_MAP.get(client_id, client_id)`
* Orders should merge using `join_key`

### Feature engineering (must include)

* Lags on target_rate: `lag_1, lag_2, lag_3, lag_6, lag_12`
* Lags on orders: `order_lag_1, order_lag_2, order_lag_3, order_lag_6, order_lag_12`
* Lags on mortgage rate: `mkt_lag_1, mkt_lag_2, mkt_lag_3, mkt_lag_6, mkt_lag_12`
* Seasonality features:

  * `month_of_year` (1–12)
  * `sin_moy`, `cos_moy` cyclical encoding
* Ratio-on-rate feature:

  * `rate_per_order = target_rate / TotalOrders` (safe divide; handle 0 orders)

### Critical robustness requirements (must implement)

* Implement a helper like `to_float_array()` that safely converts pandas Series (including nullable masked arrays) into `float64 numpy arrays` with no NaN/Inf.
* Implement `smape()` using `to_float_array()` so SMAPE never throws dtype conversion errors.
* All BigQuery numeric columns must be forced with `pd.to_numeric(..., errors='coerce')` then `fillna(...)` and `astype(float)` where appropriate.

### Model tournament (must include these models)

Baselines:

* WLB1: `0.6*lag1 + 0.2*lag2 + 0.2*lag3`
* WLB2: `0.6*lag1 + 0.25*lag3 + 0.15*lag6`
* SeasonalNaive: `lag_12`
* SeasonalNaiveGR: `lag_12 * growth_rate` where growth_rate is a recent-growth proxy (example: last / value 3 months ago, with guards)
* Native3m: rolling mean of last 3 months (use lags or true rolling)

Driver model:

* RatioRateModel: estimate `avg_ratio = mean(target_rate / orders_lag)` on training (use last 12 valid points if possible); predict `rate = avg_ratio * orders_lag`
* This model must use a **driver lag selected by lag-search** (see below)

Direct multi-horizon models:

* XGB_DirectH: train one model per horizon `h=1..HORIZON` using `y_h = target_rate.shift(-h)`
* Ridge_DirectH: same approach with Ridge regression
* Use features including: lags, orders, mortgage rate, unemployment, seasonality.

Hybrid models (must include at least 4)

* Blend rate forecasts from stable baselines + driver/ML models, with fixed weights:

  * Hybrid_WLB1_Ratio = 70% WLB1 + 30% RatioRate
  * Hybrid_SNaiveGR_Ratio = 60% SNaiveGR + 40% RatioRate
  * Hybrid_WLB2_XGB = 60% WLB2 + 40% XGB
  * Hybrid_SNaiveGR_XGB = 50% SNaiveGR + 50% XGB

### Rolling-origin (walk-forward) evaluation (must implement)

Instead of a single “last 6 months” split:

* Use multiple folds:

  * train up to t, validate t+1..t+6
  * slide forward by 3 months
* Parameters:

  * `VAL_H=6`, `SLIDE=3`, `MIN_TRAIN=18`
* For each fold, compute SMAPE for each model on the fold validation window.
* Select the winner model for the client by **lowest average SMAPE across folds**.

### Driver lag-search + stability (must implement)

* Test order lags: `{0,1,2,3,6,12}`
* Use last 24 months of data (stable window) if available.
* Choose best lag by highest absolute **Spearman** correlation between `target_rate` and `orders_lag`.
* Enforce minimum samples (>=12).
* Only treat driver as “strong” if abs(corr) >= 0.25; otherwise driver models still exist but expected to lose.

### Final forecasting (must implement)

* Generate forecasts for the next `HORIZON=15` months after last historical month.
* Forecast **rate** using the winning model’s horizon logic:

  * WLB/Native -> recursive horizon using predicted rates
  * SeasonalNaive/GR -> use t-12 with growth factor
  * RatioRate -> use selected orders lag and future orders (with fallback to last known)
  * XGB/Ridge Direct -> use horizon-specific models
  * Hybrids -> blend the component rate horizons
* Convert rate -> volume using `business_day_count` for each month:

  * Use business_day_count from df_calls if available; otherwise approximate business days with `np.busday_count()`.

### Outputs / logging

* Save:

  * forecast results with fields: `fx_date, bu, client, client_id, groups, vol_type, fx_vol, fx_rate, fx_id, fx_status, load_ts`
  * audit file includes: winner, avg SMAPE per model, driver lag + corr, and fold-level winners
  * modeling_data CSV (full merged df)
  * XGB feature importance (at least horizon=1 model, if available)
  * summary text file (one line per client winner)
* All steps must log via `logger.info/warning/error`
* Wrap whole pipeline in try/except and call:

  * `email_manager.handle_error("V2.5.x Failure", e, is_test=True)` on exception

### Coding style requirements

* Single script, no external modules besides scripthelper and listed libraries
* Heavy in-code comments for each step
* Must be deterministic where possible (`random_state=42`)
* Defensive programming: handle missing columns, empty frames, non-numeric values, NaNs.

Return the complete final script only.

---

If you want, I can also rewrite that prompt into a shorter “Gemini 300-word version” that still preserves the critical constraints (dtype safety + rolling CV + direct multi-horizon + hybrids).
