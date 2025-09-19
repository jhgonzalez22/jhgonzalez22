"""
CX Platforms – Ticket-volume forecast (PROD) — v9 (Final)

Models per client_id (per BU):
  • XGBoost (log-space)       — Tree regressor on all available features. Used for feature selection.
  • Ridge (log-space)         — Linear regressor on XGBoost-selected features.
  • Lasso (log-space)         — Sparse linear regressor on XGBoost-selected features.
  • OLS_Simple (log-space)    — Simple linear regressor on TotalOrders and month.
  • WeightedLagBlend          — Convex blend of {lag3, lag6, lag12}; weights tuned on 6m SMAPE.
  • SeasonalNaive3mGR         — Seasonal naive (t−12) scaled by recent 3-month YoY growth on target.
  • WeightedLagBlendDrv       — WLB blended with a seasonal+driver bump.

Statistical pre-checks (per client):
  • Stationarity: Augmented Dickey–Fuller (ADF) test on the target variable.
  • Correlation: Pearson correlation for potential predictors vs. target.

Outputs:
  • Append 15-month forecasts to GBQ and inactivate older rows.
  • Local CSVs:
      - forecast_results.csv
      - model_eval_debug.csv (enriched with model diagnostics)
      - statistical_tests.csv (detailed results for all stats tests per client)
      - xgb_feature_importance.csv (full feature importance from XGBoost)
      - modeling_data.csv (final data used for training/validation)
      - model_summaries.txt (full statistical summaries for fitted models)
  • Write success run-log to GBQ.
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
sys.path.append(r"C:\WFM_Scripting\Automation")  # adjust if needed
from scripthelper import Config, Logger, BigQueryManager, EmailManager

# ------------------- initialise scripthelper ---------------
config            = Config(rpt_id=284)
logger            = Logger(config)
email_manager     = EmailManager(config)
bigquery_manager  = BigQueryManager(config)

# ------------------- file / table paths --------------------
SQL_QUERY_PATH = (r"C:\WFM_Scripting\Forecasting"
                  r"\GBQ - Non-Tax Platform Ticket Timeseries by Month.sql")
DEST_TABLE         = "tax_clnt_svcs.cx_nontax_platforms_forecast"

# Local outputs
LOCAL_CSV          = r"C:\WFM_Scripting\forecast_results.csv"
AUDIT_CSV          = r"C:\WFM_Scripting\model_eval_debug.csv"
STATS_CSV          = r"C:\WFM_Scripting\statistical_tests.csv"
XGB_VIZ_CSV        = r"C:\WFM_Scripting\xgb_feature_importance.csv"
MODELING_DATA_CSV  = r"C:\WFM_Scripting\modeling_data.csv"
SUMMARIES_FILE     = r"C:\WFM_Scripting\model_summaries.txt"

# ------------------- forecast parameters --------------------
FORECAST_HORIZON   = 15
MAX_LAGS           = 12
VAL_LEN_3M         = 3
VAL_LEN_6M         = 6
STAMP              = datetime.now(pytz.timezone("America/Chicago"))
DRV_ALPHA_PROFILE  = [0.50, 0.30, 0.20]
DRV_ALPHA_STR      = "0.50/0.30/0.20"
LAMBDA_GRID        = np.arange(0.50, 0.95, 0.05)

# Maps model names to prefixes for the forecast ID.
FX_ID_PREFIX = {
    "XGBoost": "xgb_ticket",
    "Ridge": "ridge_ticket",
    "Lasso": "lasso_ticket",
    "OLS_Simple": "ols_simple_ticket",
    "WeightedLagBlend": "wlb_ticket",
    "SeasonalNaive3mGR": "seasonal_naive3mgr_ticket",
    "WeightedLagBlendDrv": "wlbdrv_ticket"
}

# ------------------- metric & modeling helpers -------------------------
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
    return np.expm1(np.clip(x, lo, hi))

def safe_round(x):
    x = 0.0 if (x is None or not np.isfinite(x)) else x
    return int(max(0, round(x)))

def safe_reg_metrics(y_true, y_pred):
    if y_pred is None: 
        return dict(smape=np.nan, mape=np.nan, rmse=np.nan)
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any(): 
        return dict(smape=np.nan, mape=np.nan, rmse=np.nan)
    return dict(
        smape=smape(y[mask], p[mask]),
        mape=mape(y[mask], p[mask]),
        rmse=float(np.sqrt(mean_squared_error(y[mask], p[mask])))
    )

def perform_statistical_tests(client_data: pd.DataFrame):
    results = {'client_id': client_data['client_id'].iloc[0]}
    # ADF (robust logging on short series)
    adf_p = None
    adf_series = client_data['total_cases_opened'].dropna()
    if len(adf_series) > 3:
        adf_p = adfuller(adf_series)[1]
        results['adf_p_value'] = adf_p
        results['is_stationary'] = adf_p < 0.05
    target = client_data['total_cases_opened']
    correlations = {}
    potential_predictors = [c for c in client_data.columns 
                            if c.startswith('lag_') or c in ['sin_month','cos_month','TotalOrders']]
    for col in potential_predictors:
        predictor = client_data[col]
        mask = target.notna() & predictor.notna()
        if mask.sum() > 2:
            t = target[mask].astype(float)
            x = predictor[mask].astype(float)
            corr, p_val = pearsonr(t, x)
            correlations[col] = {'correlation': corr, 'p_value': p_val}
    results['correlations'] = correlations

    logger.info(f"\n--- Statistical Tests for: {results['client_id']} ---")
    if adf_p is None:
        logger.info("ADF: not enough data points.")
    else:
        logger.info(f"ADF P-Value: {adf_p:.4f} (Stationary: {adf_p < 0.05})")
    logger.info("Correlation with `total_cases_opened`:")
    logger.info(f"{'Feature':<20} | {'Correlation':>12} | {'P-Value':>10}")
    logger.info("-" * 48)
    if correlations:
        for feature, values in sorted(correlations.items(), key=lambda item: item[1]['p_value']):
            is_sig = '(*)' if values['p_value'] < 0.05 else ''
            logger.info(f"{feature:<20} | {values['correlation']:>12.4f} | {values['p_value']:>10.4f} {is_sig}")
    else:
        logger.info("No valid correlations could be computed.")
    return results

# ------------------- data fetching & processing -------------------------
def fetch_workload_drivers(bq: BigQueryManager) -> pd.DataFrame:
    sql = """
    SELECT MonthOfOrder, client_id, TotalOrders
    FROM tax_clnt_svcs.view_cx_nontax_platforms_workload_drivers
    WHERE MonthOfOrder >= '2023-01-01'
    """
    try:
        df = bq.run_gbq_sql(sql, return_dataframe=True)
        if df.empty:
            return pd.DataFrame(columns=["date","client_id","TotalOrders"])
        logger.info(f"✓ Drivers pulled: {len(df):,} rows")
        # Normalize client labels if needed
        df['client_id'] = df['client_id'].replace({
            'FNC - CMS': 'FNC',
            'FNC - Ports': 'FNC',
            'Mercury Integrations': 'Mercury'
        })
        df = df.groupby(['MonthOfOrder','client_id'], as_index=False)['TotalOrders'].sum()
        df = df.rename(columns={'MonthOfOrder':'date'})
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        logger.info(f"· Failed to fetch drivers: {e}")
        return pd.DataFrame(columns=["date","client_id","TotalOrders"])

def compute_driver_r_yoy_weighted(s: pd.Series, ref_months: pd.DatetimeIndex) -> float:
    w = DRV_ALPHA_PROFILE
    rs = []
    for t in ref_months:
        num = sum(w[i] * s.get(t - pd.DateOffset(months=i+1), np.nan)  for i in range(3))
        den = sum(w[i] * s.get(t - pd.DateOffset(months=i+13), np.nan) for i in range(3))
        if np.isfinite(num) and np.isfinite(den) and den != 0:
            rs.append(num/den - 1.0)
    return float(np.mean(rs)) if rs else np.nan

def compute_client_driver_growth(drv_df: pd.DataFrame, cid: str, anchor: pd.Timestamp) -> float:
    if drv_df.empty: 
        return np.nan
    sub = drv_df[drv_df["client_id"] == cid].set_index("date").sort_index()
    if sub.empty or "TotalOrders" not in sub.columns: 
        return np.nan
    return compute_driver_r_yoy_weighted(sub["TotalOrders"], pd.DatetimeIndex([anchor]))

# ------------------- MAIN EXECUTION -------------------------
try:
    logger.info("Starting Ticket Forecasting Script (PROD) …")
    tickets_df = bigquery_manager.run_gbq_sql(SQL_QUERY_PATH, return_dataframe=True)
    if tickets_df.empty:
        raise RuntimeError("BQ returned zero rows for tickets")
    tickets_df['date'] = pd.to_datetime(tickets_df['date'])
    logger.info(f"✓ Pulled {len(tickets_df):,} ticket rows")

    drv_df = fetch_workload_drivers(bigquery_manager)

    df = tickets_df.copy()
    df["month"]     = df["date"].dt.month
    df["sin_month"] = np.sin(2*np.pi*df["month"]/12)
    df["cos_month"] = np.cos(2*np.pi*df["month"]/12)
    df = df.sort_values(['client_id','date']).query("date >= '2023-01-01'")

    # Lag features
    for lag in range(1, MAX_LAGS+1):
        df[f'lag_{lag}'] = df.groupby('client_id')['total_cases_opened'].shift(lag)

    # Attach drivers
    if not drv_df.empty:
        df = pd.merge(df, drv_df, on=['date','client_id'], how='left')

    forecasts           = []
    audit_rows          = []
    model_parameters    = []
    stat_test_results   = []
    all_modeling_data   = []
    xgb_imp_rows        = []
    all_model_summaries = []

    for cid in df['client_id'].unique():
        client_df_full = df[df['client_id'] == cid].copy()

        # Clean numeric cols
        cols_to_clean = (['total_cases_opened','sin_month','cos_month','TotalOrders']
                         + [f'lag_{l}' for l in range(1, MAX_LAGS+1)])
        for col in cols_to_clean:
            if col in client_df_full.columns:
                client_df_full[col] = pd.to_numeric(client_df_full[col], errors='coerce')

        # Stats
        stat_results = perform_statistical_tests(client_df_full)
        stat_test_results.append(stat_results)

        client_df = client_df_full.set_index('date').sort_index()
        ts = client_df['total_cases_opened']
        if len(ts) < (MAX_LAGS + VAL_LEN_6M + 1):
            continue

        # Build feature matrix
        feat = pd.DataFrame({'y': ts})
        base_cols = ['month','sin_month','cos_month','TotalOrders'] + [f'lag_{l}' for l in range(1, MAX_LAGS+1)]
        feat = feat.join(client_df[base_cols])

        # Optional external lag_12 (if provided)
        if 'total_cases_opened_lag12' in client_df.columns:
            lag12_ext = client_df['total_cases_opened_lag12'].copy()
        else:
            lag12_ext = pd.Series(dtype=float)

        # Drop rows lacking target or lag history
        feat = feat.dropna(subset=['y'] + [f'lag_{k}' for k in range(1, MAX_LAGS+1)])
        if len(feat) < (VAL_LEN_6M + 1):
            continue

        # Save modeling data (for audit)
        feat_to_save = feat.copy()
        feat_to_save['client_id'] = cid
        all_modeling_data.append(feat_to_save)

        # Train/validation splits
        train_6m = feat.iloc[:-VAL_LEN_6M]
        valid_6m = feat.iloc[-VAL_LEN_6M:]
        valid_3m = feat.iloc[-VAL_LEN_3M:]                  # <-- define explicitly

        X_tr_6, y_tr_6 = train_6m.drop(columns='y'), train_6m['y']
        X_val_3, y_val_3 = valid_3m.drop(columns='y'), valid_3m['y']   # <-- correct slicing
        X_val_6, y_val_6 = valid_6m.drop(columns='y'), valid_6m['y']

        preds_3m, preds_6m, models, extras = {}, {}, {}, {}
        y_tr_log = np.log1p(y_tr_6)

        # ------------------- XGBoost -------------------
        xgb = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=50,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
        xgb.fit(X_tr_6, y_tr_log)
        models['XGBoost']   = xgb
        preds_3m['XGBoost'] = safe_expm1(xgb.predict(X_val_3))
        preds_6m['XGBoost'] = safe_expm1(xgb.predict(X_val_6))

        # Feature importance (gain)
        gain = xgb.get_booster().get_score(importance_type='gain')
        imp_sorted = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)
        xgb_imp_rows.append({"client_id": cid, "feature_importance_gain_json": json.dumps(imp_sorted)})
        # Keep columns that appeared with positive gain
        good_cols = [c for c, g in imp_sorted if g > 0 and c in X_tr_6.columns]

        # ------------------- Ridge / Lasso (on good_cols) -------------------
        if good_cols:
            ridge = Ridge().fit(X_tr_6[good_cols], y_tr_log)
            models['Ridge']   = ridge
            preds_3m['Ridge'] = safe_expm1(ridge.predict(X_val_3[good_cols]))
            preds_6m['Ridge'] = safe_expm1(ridge.predict(X_val_6[good_cols]))

            lasso = Lasso().fit(X_tr_6[good_cols], y_tr_log)
            models['Lasso']   = lasso
            preds_3m['Lasso'] = safe_expm1(lasso.predict(X_val_3[good_cols]))
            preds_6m['Lasso'] = safe_expm1(lasso.predict(X_val_6[good_cols]))

        # ------------------- OLS_Simple -------------------
        simple_ols_features = ['TotalOrders','month']
        if 'TotalOrders' in X_tr_6.columns and not X_tr_6['TotalOrders'].isnull().all():
            y_tr_log_simple = np.log1p(y_tr_6).rename('y')
            X_tr_simple = X_tr_6[simple_ols_features].apply(pd.to_numeric, errors='coerce')
            temp_df_simple = pd.concat([y_tr_log_simple, X_tr_simple], axis=1).dropna()
            if len(temp_df_simple) > 5:
                y_clean_simple = temp_df_simple['y']
                X_clean_simple = temp_df_simple[simple_ols_features]
                ols_simple_model = sm.OLS(y_clean_simple, sm.add_constant(X_clean_simple)).fit()
                models['OLS_Simple'] = ols_simple_model
                extras['OLS_Simple']  = {"selected_features": simple_ols_features}
                logger.info(f"\n--- OLS_Simple Model Summary for: {cid} ---\n{str(ols_simple_model.summary())}\n")
                all_model_summaries.append(
                    f"--- Client: {cid}, Model: OLS_Simple ---\n{str(ols_simple_model.summary())}\n"
                )
                X_val_3_simple = sm.add_constant(X_val_3[simple_ols_features])
                X_val_6_simple = sm.add_constant(X_val_6[simple_ols_features])
                preds_3m['OLS_Simple'] = safe_expm1(ols_simple_model.predict(X_val_3_simple))
                preds_6m['OLS_Simple'] = safe_expm1(ols_simple_model.predict(X_val_6_simple))

        # ------------------- WeightedLagBlend (grid over convex weights) -------------------
        def get_wlb_lags(block_idx):
            l3  = feat['lag_3'].reindex(block_idx).astype(float).values
            l6  = feat['lag_6'].reindex(block_idx).astype(float).values
            l12_pref = lag12_ext.reindex(block_idx) if not lag12_ext.empty else pd.Series(index=block_idx, dtype=float)
            l12_fbk  = feat['lag_12'].reindex(block_idx)
            l12 = l12_pref.combine_first(l12_fbk).astype(float).values
            return l3, l6, l12

        l3_v, l6_v, l12_v = get_wlb_lags(valid_6m.index)
        mask_v = np.isfinite(l3_v) & np.isfinite(l6_v) & np.isfinite(l12_v)
        best_weights, best_s = (1.0, 0.0, 0.0), np.inf
        if mask_v.any():
            y_v = y_val_6.values[mask_v]
            l3m, l6m, l12m = l3_v[mask_v], l6_v[mask_v], l12_v[mask_v]
            w_grid = np.arange(0, 1.05, 0.05)
            for w1 in w_grid:
                for w2 in w_grid:
                    w3 = 1.0 - (w1 + w2)
                    if w3 >= -1e-9:
                        w3 = max(0.0, w3)
                        s = smape(y_v, w1*l3m + w2*l6m + w3*l12m)
                        if s < best_s:
                            best_weights, best_s = (w1, w2, w3), s

        w3, w6, w12 = best_weights
        extras['WeightedLagBlend'] = {"w_lag3": w3, "w_lag6": w6, "w_lag12": w12}

        def wlb_block_preds(idx):
            l3b, l6b, l12b = get_wlb_lags(idx)
            return w3*np.nan_to_num(l3b) + w6*np.nan_to_num(l6b) + w12*np.nan_to_num(l12b)

        preds_3m['WeightedLagBlend'] = wlb_block_preds(valid_3m.index)
        preds_6m['WeightedLagBlend'] = wlb_block_preds(valid_6m.index)

        # ------------------- SeasonalNaive3mGR (target growth) -------------------
        anchor = ts.index[-1]
        r_tgt = compute_driver_r_yoy_weighted(ts, pd.DatetimeIndex([anchor]))

        def seasonal_naive_3mgr(idx):
            vals = []
            for d in idx:
                base = ts.get(d - pd.DateOffset(months=12), np.nan)
                vals.append(np.nan if not np.isfinite(base) else base * (1.0 + (r_tgt if np.isfinite(r_tgt) else 0.0)))
            return np.array(vals, dtype=float)

        preds_3m['SeasonalNaive3mGR'] = seasonal_naive_3mgr(valid_3m.index)
        preds_6m['SeasonalNaive3mGR'] = seasonal_naive_3mgr(valid_6m.index)
        extras['SeasonalNaive3mGR'] = {'r_tgt': r_tgt}

        # ------------------- WeightedLagBlendDrv (driver bump) -------------------
        r_drv = compute_client_driver_growth(drv_df, cid, anchor)
        if np.isfinite(r_drv):
            def wlbdrv_block_preds(idx, lam):
                wlb_vals = wlb_block_preds(idx)
                sdrv = []
                for d in idx:
                    base = ts.get(d - pd.DateOffset(months=12), np.nan)
                    sdrv.append(base * (1.0 + r_drv) if np.isfinite(base) else np.nan)
                sdrv = np.array(sdrv, dtype=float)
                return lam*wlb_vals + (1.0 - lam)*sdrv

            best_lam, best_s = None, np.inf
            for lam in LAMBDA_GRID:
                p_try = wlbdrv_block_preds(valid_3m.index, lam)
                mask  = np.isfinite(p_try) & np.isfinite(y_val_3.values)
                if mask.any():
                    s = smape(y_val_3.values[mask], p_try[mask])
                    if s < best_s:
                        best_s, best_lam = s, lam
            if best_lam is not None:
                preds_3m['WeightedLagBlendDrv'] = wlbdrv_block_preds(valid_3m.index, best_lam)
                preds_6m['WeightedLagBlendDrv'] = wlbdrv_block_preds(valid_6m.index, best_lam)
                extras['WeightedLagBlendDrv'] = {"lambda": float(best_lam), "r_drv": float(r_drv)}
        else:
            logger.info(f"· {cid:<25} WLB+Drivers skipped (no finite driver growth)")

        # ------------------- Select winner on 3m SMAPE -------------------
        if not preds_3m:
            logger.info(f"· {cid:<25} – no models could be fitted.")
            continue

        # Score all models
        models_to_score = list(preds_3m.keys())
        for m in models_to_score:
            met3 = safe_reg_metrics(y_val_3.values, preds_3m.get(m))
            met6 = safe_reg_metrics(y_val_6.values, preds_6m.get(m))
            row = dict(
                client_id=cid, model=m, val_smape_3m=met3['smape'], val_smape_6m=met6['smape'],
                MAPE_3m=met3['mape'], RMSE_3m=met3['rmse'], alpha_profile=DRV_ALPHA_STR
            )
            if m in extras:
                row.update(extras[m])
            audit_rows.append(row)

        # Winner
        smapes_3 = {m: smape(y_val_3, p) for m, p in preds_3m.items() if p is not None and np.isfinite(p).any()}
        if not smapes_3:
            continue
        best_model = min(smapes_3, key=smapes_3.get)
        for r in audit_rows[-len(models_to_score):]:
            r["winner_model"] = best_model
        model_parameters.append({"client_id": cid, "model_name": best_model, "params": extras.get(best_model, {})})

        # ------------------- Recursive 15-month forecast -------------------
        future_idx = pd.date_range(ts.index[-1], periods=FORECAST_HORIZON+1, freq='MS')[1:]
        fx_tag = f"{FX_ID_PREFIX[best_model]}_{STAMP:%Y%m%d}"

        def append_fx(d, v):
            forecasts.append(dict(
                fx_date=d.strftime('%Y-%m-%d'),
                client_id=cid,
                vol_type="tickets",
                fx_vol=safe_round(v),
                fx_id=fx_tag,
                fx_status="A",
                load_ts=STAMP
            ))

        hist = deque(ts.tolist(), maxlen=max(24, MAX_LAGS))
        # Hold TotalOrders flat at last observed for this client (unless you later add driver forecasts)
        last_total_orders = client_df_full['TotalOrders'].dropna().iloc[-1] if 'TotalOrders' in client_df_full.columns and client_df_full['TotalOrders'].notna().any() else np.nan

        for d in future_idx:
            pred = np.nan
            if best_model in ['Ridge','Lasso','XGBoost']:
                row = {f'lag_{k}': (hist[-k] if len(hist) >= k else np.nan) for k in range(1, MAX_LAGS+1)}
                row.update({
                    'month': d.month,
                    'sin_month': np.sin(2*np.pi*d.month/12),
                    'cos_month': np.cos(2*np.pi*d.month/12),
                    'TotalOrders': last_total_orders
                })
                xrow_df = pd.DataFrame([row])
                if best_model in ['Ridge','Lasso']:
                    # Use only columns seen by model
                    need_cols = [c for c in good_cols if c in xrow_df.columns]
                    pred = safe_expm1(models[best_model].predict(xrow_df[need_cols])[0])
                else:  # XGBoost — align to training columns strictly
                    need_cols = list(X_tr_6.columns)
                    pred = safe_expm1(models['XGBoost'].predict(xrow_df[need_cols])[0])

            elif best_model == 'OLS_Simple':
                xrow_df = pd.DataFrame([{'TotalOrders': last_total_orders, 'month': d.month}])[simple_ols_features]
                xrow = sm.add_constant(xrow_df)
                pred = safe_expm1(models['OLS_Simple'].predict(xrow).iloc[0])

            elif best_model == 'WeightedLagBlend':
                p = extras['WeightedLagBlend']
                y3  = hist[-3]  if len(hist) >= 3  else 0.0
                y6  = hist[-6]  if len(hist) >= 6  else 0.0
                y12 = hist[-12] if len(hist) >= 12 else (hist[-1] if len(hist)>0 else 0.0)
                pred = p['w_lag3']*y3 + p['w_lag6']*y6 + p['w_lag12']*y12

            elif best_model == 'SeasonalNaive3mGR':
                base = hist[-12] if len(hist) >= 12 else (hist[-1] if len(hist)>0 else 0.0)
                pred = base * (1.0 + (r_tgt if np.isfinite(r_tgt) else 0.0))

            elif best_model == 'WeightedLagBlendDrv':
                p = extras['WeightedLagBlend']
                y3  = hist[-3]  if len(hist) >= 3  else 0.0
                y6  = hist[-6]  if len(hist) >= 6  else 0.0
                y12 = hist[-12] if len(hist) >= 12 else (hist[-1] if len(hist)>0 else 0.0)
                wlb = p['w_lag3']*y3 + p['w_lag6']*y6 + p['w_lag12']*y12
                sdrv_base = hist[-12] if len(hist) >= 12 else (hist[-1] if len(hist)>0 else 0.0)
                sdrv = sdrv_base * (1.0 + (extras['WeightedLagBlendDrv'].get('r_drv', 0.0) if 'WeightedLagBlendDrv' in extras else 0.0))
                lam  = extras['WeightedLagBlendDrv']['lambda']
                pred = lam*wlb + (1.0 - lam)*sdrv

            hist.append(pred)
            append_fx(d, pred)

    # ------------------- Write outputs -------------------
    fx_df    = pd.DataFrame(forecasts).sort_values(["client_id","fx_date"])
    audit_df = pd.DataFrame(audit_rows).sort_values(["client_id","model"]) if audit_rows else pd.DataFrame()

    logger.info(f"Pushing {len(fx_df):,} rows to {DEST_TABLE} …")
    if not bigquery_manager.import_data_to_bigquery(
        fx_df, DEST_TABLE, gbq_insert_action="append", auto_convert_df=True
    ):
        raise RuntimeError("GBQ import failed")

    dedup_sql = f"""
    UPDATE `{DEST_TABLE}` t
    SET t.fx_status = 'I'
    WHERE t.fx_status = 'A'
      AND EXISTS (
        SELECT 1
        FROM `{DEST_TABLE}` s
        WHERE s.client_id = t.client_id
          AND s.vol_type  = t.vol_type
          AND s.fx_date   = t.fx_date
          AND s.load_ts   > t.load_ts
      )
    """
    bigquery_manager.run_gbq_sql(dedup_sql, return_dataframe=False)
    logger.info("✓ Older forecasts set to inactive")

    fx_df.to_csv(LOCAL_CSV, index=False)
    audit_df.to_csv(AUDIT_CSV, index=False)
    pd.DataFrame(xgb_imp_rows).to_csv(XGB_VIZ_CSV, index=False)
    logger.info(f"✓ CSVs saved to\n    {LOCAL_CSV}\n    {AUDIT_CSV}\n    {XGB_VIZ_CSV}")

    with open(SUMMARIES_FILE, 'w') as f:
        for summary in all_model_summaries:
            f.write(summary + "="*80 + "\n\n")
    logger.info(f"✓ Model summaries saved to\n    {SUMMARIES_FILE}")

    pd.DataFrame(stat_test_results).to_csv(STATS_CSV, index=False)
    logger.info(f"✓ Statistical tests saved to\n    {STATS_CSV}")

    if all_modeling_data:
        modeling_df = pd.concat(all_modeling_data)
        modeling_df.to_csv(MODELING_DATA_CSV, index=True)
        logger.info(f"✓ Modeling data saved to\n    {MODELING_DATA_CSV}")

    # Keep if your current scripthelper expects it (your July ‘25 flow does)
    bigquery_manager.update_log_in_bigquery()
    logger.info("✓ Ticket forecasting (PROD) completed successfully")

except Exception as exc:
    email_manager.handle_error("Ticket Forecasting Script Failure", exc, is_test=True)
