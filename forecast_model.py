"""
CX Platforms – Phone-volume forecast (CallsOffered) — v2.3.1 (Corrected Prediction Logic + DType Hardening)

This script forecasts monthly phone call volumes by benchmarking a comprehensive suite of models,
mirroring the advanced approach of the ticket forecasting pipeline (Rpt_284).

Pipeline:
  1) Fetches historical phone volume and workload driver (TotalOrders) data from BigQuery.
  2) Maps specialized clients (FNC, Mercury) to their respective drivers.
  3) Engineers features like lags (target and driver) and seasonal terms.
  4) Performs statistical pre-checks (ADF for stationarity, Pearson correlation).
  5) For each client, trains and evaluates a roster of 10 different forecasting models.
  6) Selects the winning model based on the lowest 6-month SMAPE.
  7) Generates a 15-month recursive forecast using the winning model and forecasted driver data.
  8) Pushes results to BigQuery, deactivates older forecasts, and saves detailed local audit files.
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
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")  # keep logs tidy

# ------------------- helper-package path --------------------
sys.path.append(r'C:\WFM_Scripting\Automation')  # update if necessary
from scripthelper import Config, Logger, BigQueryManager, EmailManager

# ------------------- initialise scripthelper ---------------
config            = Config(rpt_id=283)
logger            = Logger(config)
email_manager     = EmailManager(config)
bigquery_manager  = BigQueryManager(config)

# ------------------- file / table paths --------------------
SQL_QUERY_PATH = (r"C:\WFM_Scripting\Forecasting"
                  r"\GBQ - Non-Tax Platform Phone Timeseries by Month.sql")
DEST_TABLE     = "tax_clnt_svcs.cx_nontax_platforms_forecast"

# --- Expanded local outputs ---
LOCAL_CSV         = r"C:\WFM_Scripting\forecast_results_phone.csv"
AUDIT_CSV         = r"C:\WFM_Scripting\model_eval_debug_phone.csv"
STATS_CSV         = r"C:\WFM_Scripting\statistical_tests_phone.csv"
XGB_VIZ_CSV       = r"C:\WFM_Scripting\xgb_feature_importance_phone.csv"
MODELING_DATA_CSV = r"C:\WFM_Scripting\modeling_data_phone.csv"
SUMMARIES_FILE    = r"C:\WFM_Scripting\model_summaries_phone.txt"

# ------------------- forecast parameters --------------------
FORECAST_HORIZON = 15
MAX_LAGS         = 12
VAL_LEN_6M       = 6
STAMP            = datetime.now(pytz.timezone("America/Chicago"))
DRV_ALPHA_PROFILE  = [0.50, 0.30, 0.20]
DRV_ALPHA_STR      = "0.50/0.30/0.20"

# --- Expanded model roster ---
FX_ID_PREFIX = {
    "XGBoost": "xgb_phone",
    "Ridge": "ridge_phone",
    "Lasso": "lasso_phone",
    "OLS_Simple": "ols_simple_phone",
    "OLS_TotalOrdersOnly": "ols_to_phone",
    "OLS_DriverLags": "ols_drv_phone",
    "Ridge_DriverLags": "ridge_drv_phone",
    "WeightedLagBlend": "wlb_phone",
    "SeasonalNaive3mGR": "seasonal_naive3mgr_phone",
    "WeightedLagBlendDrv": "wlbdrv_phone"
}

# ------------------- metric & helper functions -------------------------
def mape(actual, forecast):
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    denom = np.where(actual == 0, 1.0, np.abs(actual))
    return np.mean(np.abs((actual - forecast) / denom)) * 100

def smape(actual, forecast):
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    denom = (np.abs(actual) + np.abs(forecast)) / 2.0
    denom[denom == 0] = 1.0
    return np.mean(np.abs(actual - forecast) / denom) * 100

def safe_expm1(x, lo=-20.0, hi=20.0):
    # Accepts scalar or array-like; always returns numpy float/array
    arr = np.asarray(x, dtype=float)
    return np.expm1(np.clip(arr, lo, hi))

def safe_round(x):
    x = 0.0 if (x is None or not np.isfinite(x)) else x
    return int(max(0, round(x)))

def safe_reg_metrics(y_true, y_pred):
    if y_pred is None: return dict(smape=np.nan, mape=np.nan, rmse=np.nan)
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any(): return dict(smape=np.nan, mape=np.nan, rmse=np.nan)
    return dict(smape=smape(y[mask], p[mask]), mape=mape(y[mask], p[mask]), rmse=float(np.sqrt(mean_squared_error(y[mask], p[mask]))))

def perform_statistical_tests(client_data: pd.DataFrame):
    results = {'client_id': client_data['client_id'].iloc[0]}
    target_col = 'CallsOffered'
    adf_series = client_data[target_col].dropna()
    if len(adf_series) > 3:
        adf_p = adfuller(adf_series)[1]
        results['adf_p_value'] = adf_p
        results['is_stationary'] = adf_p < 0.05

    target = client_data[target_col]
    correlations = {}
    potential_predictors = [col for col in client_data.columns if col.startswith(('lag_', 'drv_lag_')) or col in ['sin_month', 'cos_month', 'TotalOrders']]
    for col in potential_predictors:
        predictor = client_data[col]
        mask = target.notna() & predictor.notna()
        if mask.sum() > 2:
            t, p = target[mask].astype(float), predictor[mask].astype(float)
            if np.std(t) > 0 and np.std(p) > 0:
                corr, p_val = pearsonr(t, p)
                correlations[col] = {'correlation': corr, 'p_value': p_val}
    results['correlations'] = correlations
    logger.info(f"\n--- Statistical Tests for: {results['client_id']} ---")
    if 'adf_p_value' in results: logger.info(f"ADF P-Value: {results['adf_p_value']:.4f} (Stationary: {results['is_stationary']})")
    return results

def fetch_driver_forecasts(bq: BigQueryManager) -> pd.DataFrame:
    sql = "SELECT fx_date, client_id, fx_vol AS TotalOrders_fx FROM tax_clnt_svcs.cx_nontax_platforms_workload_drivers_fx WHERE fx_status = 'A'"
    df = bq.run_gbq_sql(sql, return_dataframe=True)
    if df.empty: return pd.DataFrame()
    df['date'] = pd.to_datetime(df['fx_date'])
    return df.drop(columns=['fx_date'])

def fetch_workload_drivers(bq: BigQueryManager) -> pd.DataFrame:
    sql = "SELECT MonthOfOrder, client_id, TotalOrders FROM tax_clnt_svcs.view_cx_nontax_platforms_workload_drivers"
    df = bq.run_gbq_sql(sql, return_dataframe=True)
    if df.empty: return pd.DataFrame()
    df['date'] = pd.to_datetime(df['MonthOfOrder'])
    return df.drop(columns=['MonthOfOrder'])

def compute_driver_r_yoy_weighted(s: pd.Series, ref_months: pd.DatetimeIndex) -> float:
    w = DRV_ALPHA_PROFILE
    rs = []
    for t in ref_months:
        num = sum(w[i] * s.get(t - pd.DateOffset(months=i+1), np.nan) for i in range(3))
        den = sum(w[i] * s.get(t - pd.DateOffset(months=i+13), np.nan) for i in range(3))
        if np.isfinite(num) and np.isfinite(den) and den != 0: rs.append(num/den - 1.0)
    return float(np.mean(rs)) if rs else np.nan

def compute_client_driver_growth(drv_df: pd.DataFrame, cid: str, anchor: pd.Timestamp) -> float:
    if drv_df.empty: return np.nan
    sub = drv_df[drv_df["client_id"] == cid].set_index("date").sort_index()
    return compute_driver_r_yoy_weighted(sub["TotalOrders"], pd.DatetimeIndex([anchor])) if not sub.empty else np.nan

# ------------------- MAIN EXECUTION -------------------------
try:
    logger.info("Starting Phone Forecasting Script (v2.3.1)...")

    # 1) Pull base data
    df = bigquery_manager.run_gbq_sql(SQL_QUERY_PATH, return_dataframe=True)
    if df.empty: raise ValueError("BigQuery returned no phone volume data.")
    logger.info(f"✓ Pulled {df.shape[0]:,} phone volume rows")

    # Basic time features
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

    # Drivers: history + forecasts
    drv_df = fetch_workload_drivers(bigquery_manager)
    drv_fx_df = fetch_driver_forecasts(bigquery_manager)
    DRIVER_MAP = {'FNC - CMS': 'FNC', 'FNC - Ports': 'FNC', 'Mercury Integrations': 'Mercury'}

    if not drv_df.empty:
        df['join_key'] = df['client_id'].map(DRIVER_MAP).fillna(df['client_id'])
        drv_df_to_join = drv_df.rename(columns={'client_id': 'join_key'})
        grouped_drivers = drv_df_to_join.groupby(['date', 'join_key'])['TotalOrders'].sum().reset_index()
        df = pd.merge(df, grouped_drivers, on=['date', 'join_key'], how='left')
        df = df.drop(columns=['join_key'])
    else:
        df['TotalOrders'] = np.nan

    # Sort + Lags
    df = df.sort_values(['client_id', 'date'])
    for lag in range(1, MAX_LAGS + 1):
        df[f'lag_{lag}'] = df.groupby('client_id')['CallsOffered'].shift(lag)
        df[f'drv_lag_{lag}'] = df.groupby('client_id')['TotalOrders'].shift(lag)

    forecasts, audit_rows, stat_test_results, all_modeling_data = [], [], [], []
    xgb_imp_rows, all_model_summaries = [], []

    # ------------- per-client loop -------------
    for cid in df['client_id'].unique():
        client_df_full = df[df['client_id'] == cid].copy()

        # Ensure all modeling columns are numeric where applicable
        numeric_cols = [c for c in client_df_full.columns
                        if ('lag_' in c) or ('Orders' in c) or ('Offered' in c) or (c in ['month', 'sin_month', 'cos_month'])]
        for col in numeric_cols:
            client_df_full[col] = pd.to_numeric(client_df_full[col], errors='coerce')

        # Stats
        if len(client_df_full) > 0:
            stat_test_results.append(perform_statistical_tests(client_df_full))

        client_df = client_df_full.set_index('date').sort_index()
        ts = client_df['CallsOffered']

        # Build feature frame
        feat = pd.DataFrame({'y': ts})
        base_cols = (
            ['month', 'sin_month', 'cos_month', 'TotalOrders'] +
            [f'lag_{l}' for l in range(1, MAX_LAGS + 1)] +
            [f'drv_lag_{l}' for l in range(1, MAX_LAGS + 1)]
        )
        cols_to_join = [col for col in base_cols if col in client_df.columns]
        feat = feat.join(client_df[cols_to_join])

        # Dtype hardening
        for c in feat.columns:
            if c != 'y':
                feat[c] = pd.to_numeric(feat[c], errors='coerce')

        # Must have target lags for validation
        feat.dropna(subset=['y'] + [f'lag_{k}' for k in range(1, MAX_LAGS+1)], inplace=True)
        if len(feat) < (VAL_LEN_6M + 1):
            logger.warning(f"Skipping {cid} – not enough data after lag drop.")
            continue

        all_modeling_data.append(feat.assign(client_id=cid))

        # Train/Validation split
        train_6m, valid_6m = feat.iloc[:-VAL_LEN_6M], feat.iloc[-VAL_LEN_6M:]
        X_tr_6, y_tr_6 = train_6m.drop(columns='y'), train_6m['y']
        X_val_6, y_val_6 = valid_6m.drop(columns='y'), valid_6m['y']

        # Final numeric coercion for model safety
        X_tr_6 = X_tr_6.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_val_6 = X_val_6.apply(pd.to_numeric, errors='coerce').fillna(0)
        y_tr_log = np.log1p(np.asarray(y_tr_6, dtype=float))

        preds_6m, models, extras = {}, {}, {}

        # --- MODEL 1: XGBoost (log-space) ---
        xgb = XGBRegressor(objective="reg:squarederror",
                           n_estimators=50,
                           learning_rate=0.05,
                           max_depth=4,
                           random_state=42)
        xgb.fit(X_tr_6, y_tr_log)
        preds_6m['XGBoost'] = safe_expm1(xgb.predict(X_val_6))
        models['XGBoost'] = xgb
        gain = xgb.get_booster().get_score(importance_type='gain')
        good_cols = [c for c, g in sorted(gain.items(), key=lambda i: i[1], reverse=True) if g > 0 and c in X_tr_6.columns]
        xgb_imp_rows.append({
            "client_id": cid,
            "feature_importance": json.dumps(sorted(gain.items(), key=lambda i: i[1], reverse=True))
        })

        # --- MODELS 2 & 3: Ridge / Lasso (log-space; XGB-selected features) ---
        if good_cols:
            Xtr_good = X_tr_6[good_cols]
            Xva_good = X_val_6[good_cols]

            ridge = Ridge().fit(Xtr_good, y_tr_log)
            preds_6m['Ridge'] = safe_expm1(ridge.predict(Xva_good))
            models['Ridge'] = ridge

            lasso = Lasso().fit(Xtr_good, y_tr_log)
            preds_6m['Lasso'] = safe_expm1(lasso.predict(Xva_good))
            models['Lasso'] = lasso

        # --- MODELS 4-7: OLS / Ridge variants (log-space for OLS/Ridge) ---
        ols_features = {
            'OLS_Simple': ['TotalOrders', 'month'],
            'OLS_TotalOrdersOnly': ['TotalOrders'],
            'OLS_DriverLags': [c for c in ['drv_lag_1','drv_lag_3','drv_lag_6','drv_lag_12','month'] if c in X_tr_6.columns]
        }

        for name, features in ols_features.items():
            if not features or not all(f in X_tr_6.columns for f in features):
                continue

            # Build clean training matrix for statsmodels
            Xtr_sub = X_tr_6[features].apply(pd.to_numeric, errors='coerce')
            temp_df = pd.concat([pd.Series(y_tr_log, name='y'), Xtr_sub], axis=1).dropna()
            if len(temp_df) > len(features) + 1:
                y_clean = np.asarray(temp_df['y'], dtype=float)
                X_clean = sm.add_constant(temp_df[features].astype(float))
                model = sm.OLS(y_clean, X_clean).fit()

                # Validation design matrix (coerced to float) + prediction in log-space
                Xv = sm.add_constant(X_val_6[features].apply(pd.to_numeric, errors='coerce').fillna(0), has_constant='add')
                raw_predictions = model.predict(Xv)
                preds_6m[name] = safe_expm1(np.asarray(raw_predictions, dtype=float))

                models[name], extras[name] = model, {"selected_features": features}
                all_model_summaries.append(f"--- Client:{cid}, Model:{name} ---\n{str(model.summary())}\n")

                if name == 'OLS_DriverLags':
                    ridge_drv = Ridge().fit(X_clean.drop(columns='const'), y_clean)
                    preds_6m['Ridge_DriverLags'] = safe_expm1(
                        np.asarray(ridge_drv.predict(X_val_6[features].apply(pd.to_numeric, errors='coerce').fillna(0)), dtype=float)
                    )
                    models['Ridge_DriverLags'], extras['Ridge_DriverLags'] = ridge_drv, {"selected_features": features}

        # --- MODEL 8: WeightedLagBlend (levels) ---
        lags = {l: feat.get(f'lag_{l}', pd.Series(index=valid_6m.index)).reindex(valid_6m.index) for l in [3, 6, 12]}
        mask = pd.concat(lags.values(), axis=1).notna().all(axis=1)
        best_weights, best_s = (1.0, 0.0, 0.0), np.inf
        if mask.any():
            y_v = np.asarray(y_val_6[mask].values, dtype=float)
            l3 = np.asarray(lags[3][mask].values, dtype=float)
            l6 = np.asarray(lags[6][mask].values, dtype=float)
            l12 = np.asarray(lags[12][mask].values, dtype=float)
            for w1 in np.arange(0, 1.05, 0.05):
                for w2 in np.arange(0, 1.05 - w1, 0.05):
                    w3 = 1 - w1 - w2
                    s = smape(y_v, w1 * l3 + w2 * l6 + w3 * l12)
                    if s < best_s:
                        best_weights, best_s = (w1, w2, w3), s
        w3, w6, w12 = best_weights
        preds_6m['WeightedLagBlend'] = (
            w3 * lags[3].fillna(0) + w6 * lags[6].fillna(0) + w12 * lags[12].fillna(0)
        )
        extras['WeightedLagBlend'] = {"w_lag3": w3, "w_lag6": w6, "w_lag12": w12}

        # --- MODEL 9: SeasonalNaive3mGR (levels + growth) ---
        anchor = ts.index[-1]
        r_tgt = compute_driver_r_yoy_weighted(ts, pd.DatetimeIndex([anchor]))
        preds_6m['SeasonalNaive3mGR'] = [
            ts.get(d - pd.DateOffset(months=12), np.nan) * (1.0 + (r_tgt if np.isfinite(r_tgt) else 0.0))
            for d in valid_6m.index
        ]
        extras['SeasonalNaive3mGR'] = {'r_tgt': r_tgt}

        # --- MODEL 10: WeightedLagBlendDrv (blend with driver YoY growth) ---
        r_drv = compute_client_driver_growth(drv_df, DRIVER_MAP.get(cid, cid), anchor)
        if np.isfinite(r_drv):
            sdrv_preds = [
                ts.get(d - pd.DateOffset(months=12), np.nan) * (1.0 + r_drv) for d in valid_6m.index
            ]
            best_lam, best_s = 0.5, np.inf
            base_wlb = np.asarray(preds_6m['WeightedLagBlend'].reindex(valid_6m.index).fillna(0).values, dtype=float)
            base_sdrv = np.asarray(pd.Series(sdrv_preds, index=valid_6m.index).fillna(0).values, dtype=float)
            yv = np.asarray(y_val_6.values, dtype=float)
            for lam_try in np.arange(0.5, 0.95, 0.05):
                p_try = lam_try * base_wlb + (1.0 - lam_try) * base_sdrv
                s = smape(yv, p_try)
                if s < best_s:
                    best_lam, best_s = lam_try, s
            preds_6m['WeightedLagBlendDrv'] = best_lam * base_wlb + (1.0 - best_lam) * base_sdrv
            extras['WeightedLagBlendDrv'] = {"lambda": float(best_lam), "r_drv": float(r_drv)}

        # 5) Evaluate and select winner by lowest 6-month SMAPE
        scores = {m: safe_reg_metrics(y_val_6.values, p)['smape'] for m, p in preds_6m.items()}
        valid_scores = {m: s for m, s in scores.items() if np.isfinite(s)}
        if not valid_scores:
            logger.warning(f"No valid scores for {cid}.")
            continue
        best_model = min(valid_scores, key=valid_scores.get)
        logger.info(f"✓ WINNER for {cid}: {best_model} (SMAPE: {valid_scores[best_model]:.2f})")

        for m, p in preds_6m.items():
            audit_rows.append({
                'client_id': cid,
                'model': m,
                'val_smape_6m': scores.get(m),
                'winner_model': best_model,
                **extras.get(m, {})
            })

        # 6) Produce 15-month recursive forecast
        future_idx = pd.date_range(ts.index[-1], periods=FORECAST_HORIZON + 1, freq='MS')[1:]
        hist = deque(ts.tolist(), maxlen=24)
        drv_hist = deque(client_df_full['TotalOrders'].dropna().tolist(), maxlen=24)

        driver_lookup_key = DRIVER_MAP.get(cid, cid)
        last_total_orders = client_df_full['TotalOrders'].dropna().iloc[-1] if client_df_full['TotalOrders'].notna().any() else np.nan
        driver_fx_lookup = {pd.Timestamp(d): v for d, v in drv_fx_df[drv_fx_df['client_id'] == driver_lookup_key][['date', 'TotalOrders_fx']].values}
        if driver_fx_lookup:
            logger.info(f"· {cid} will use forecasted drivers from {driver_lookup_key}")

        for d in future_idx:
            pred = np.nan
            current_total_orders = driver_fx_lookup.get(d, last_total_orders)

            if best_model in ['XGBoost', 'Ridge', 'Lasso', 'OLS_Simple', 'OLS_TotalOrdersOnly', 'OLS_DriverLags', 'Ridge_DriverLags']:
                row = {f'lag_{k}': (hist[-k] if len(hist) >= k else np.nan) for k in range(1, MAX_LAGS + 1)}
                row.update({f'drv_lag_{k}': (drv_hist[-k] if len(drv_hist) >= k else np.nan) for k in range(1, MAX_LAGS + 1)})
                row.update({
                    'month': d.month,
                    'sin_month': np.sin(2*np.pi*d.month/12),
                    'cos_month': np.cos(2*np.pi*d.month/12),
                    'TotalOrders': current_total_orders
                })
                x_row_df = pd.DataFrame([row]).apply(pd.to_numeric, errors='coerce').fillna(0)

                if best_model == 'XGBoost':
                    # align columns to training matrix
                    x_input = x_row_df.reindex(columns=X_tr_6.columns, fill_value=0)
                    pred = float(safe_expm1(xgb.predict(x_input)[0]))

                elif best_model in ['Ridge', 'Lasso']:
                    x_input = x_row_df[good_cols].reindex(columns=good_cols, fill_value=0)
                    pred = float(safe_expm1(models[best_model].predict(x_input)[0]))

                elif best_model in ['OLS_Simple', 'OLS_TotalOrdersOnly', 'OLS_DriverLags']:
                    features = extras[best_model]['selected_features']
                    Xf = sm.add_constant(x_row_df[features].reindex(columns=features, fill_value=0), has_constant='add')
                    raw_pred = float(models[best_model].predict(Xf)[0])
                    pred = float(safe_expm1(raw_pred))

                elif best_model == 'Ridge_DriverLags':
                    features = extras[best_model]['selected_features']
                    x_input = x_row_df[features].reindex(columns=features, fill_value=0)
                    raw_pred = float(models[best_model].predict(x_input)[0])
                    pred = float(safe_expm1(raw_pred))

            elif best_model == 'WeightedLagBlend':
                p = extras['WeightedLagBlend']
                y3  = hist[-3]  if len(hist)  >= 3  else (hist[-1] if hist else 0)
                y6  = hist[-6]  if len(hist)  >= 6  else (hist[-1] if hist else 0)
                y12 = hist[-12] if len(hist) >= 12 else (hist[-1] if hist else 0)
                pred = p['w_lag3']*y3 + p['w_lag6']*y6 + p['w_lag12']*y12

            elif best_model == 'SeasonalNaive3mGR':
                base = hist[-12] if len(hist) >= 12 else (hist[-1] if hist else 0)
                pred = base * (1.0 + (r_tgt if np.isfinite(r_tgt) else 0.0))

            elif best_model == 'WeightedLagBlendDrv':
                p_wlb = extras['WeightedLagBlend']
                p_drv = extras['WeightedLagBlendDrv']
                y3  = hist[-3]  if len(hist)  >= 3  else (hist[-1] if hist else 0)
                y6  = hist[-6]  if len(hist)  >= 6  else (hist[-1] if hist else 0)
                y12 = hist[-12] if len(hist) >= 12 else (hist[-1] if hist else 0)
                wlb = p_wlb['w_lag3']*y3 + p_wlb['w_lag6']*y6 + p_wlb['w_lag12']*y12
                sdrv_base = hist[-12] if len(hist) >= 12 else (hist[-1] if hist else 0)
                sdrv = sdrv_base * (1.0 + p_drv.get('r_drv', 0.0))
                pred = p_drv['lambda'] * wlb + (1.0 - p_drv['lambda']) * sdrv

            hist.append(pred)
            drv_hist.append(current_total_orders)
            forecasts.append({
                'fx_date': d,
                'client_id': cid,
                'vol_type': 'phone',
                'fx_vol': safe_round(pred),
                'fx_id': f"{FX_ID_PREFIX[best_model]}_{STAMP:%Y%m%d}",
                'fx_status': "A",
                'load_ts': STAMP
            })

    # --- 7) Push results and save all outputs ---
    if forecasts:
        fx_df = pd.DataFrame(forecasts)
        bigquery_manager.import_data_to_bigquery(
            fx_df, DEST_TABLE, gbq_insert_action="append", auto_convert_df=True
        )
        dedup_sql = (
            f"UPDATE `{DEST_TABLE}` t "
            f"SET fx_status = 'I' "
            f"WHERE fx_status = 'A' AND vol_type = 'phone' "
            f"AND EXISTS ("
            f"  SELECT 1 FROM `{DEST_TABLE}` s "
            f"  WHERE s.client_id=t.client_id "
            f"    AND s.vol_type=t.vol_type "
            f"    AND s.fx_date=t.fx_date "
            f"    AND s.load_ts > t.load_ts"
            f")"
        )
        bigquery_manager.run_gbq_sql(dedup_sql, return_dataframe=False)

        fx_df.to_csv(LOCAL_CSV, index=False)
        pd.DataFrame(audit_rows).to_csv(AUDIT_CSV, index=False)
        pd.DataFrame(xgb_imp_rows).to_csv(XGB_VIZ_CSV, index=False)
        if stat_test_results:
            pd.DataFrame(stat_test_results).to_csv(STATS_CSV, index=False)
        if all_modeling_data:
            pd.concat(all_modeling_data).to_csv(MODELING_DATA_CSV, index=True)
        with open(SUMMARIES_FILE, 'w') as f:
            f.write("\n\n".join(all_model_summaries))

        logger.info("✓ All local audit files saved.")
        bigquery_manager.update_log_in_bigquery()
        logger.info("✓ Phone forecasting completed successfully")
    else:
        logger.warning("No forecasts were generated.")

except Exception as exc:
    # Leave is_test=True so it emails but doesn't spam real channels if configured
    email_manager.handle_error("Phone Forecasting Script Failure", exc, is_test=True)
