"""
Credit Tickets - Monthly Volume Forecast — v1.17.3 (Final Hotfix++)

What’s new vs v1.17.2:
  • Target hygiene: treat `total_cases_opened` as the only target; prevent leakage
  • OLS guardrails:
      - Lasso pre-select → prune multicollinearity (|corr|>0.95) with priority-keep
      - Feature cap to preserve degrees of freedom (DF)
      - Always include seasonal anchor if present (lag_12), and force `trend` if only static market vars selected
  • Robust inference:
      - Build BOTH target lags and driver lags during recursion (k=1..12)
      - Use driver FX when available; hold market vars flat at last known values
      - Pure float matrices for statsmodels to avoid dtype errors
  • SeasonalNaive branches build driver each step and update histories safely

Benchmarked models (6-month validation):
  - XGBoost (log target)
  - OLS_Auto_Market (Lasso feature select + guardrails)
  - SeasonalNaive3mGR, SeasonalNaive3mDiffGR

Local outputs: forecasts, audit, modeling, stats, XGB feature importances
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

MARKET_VARS = ['UNRATE', 'HSN1F', 'FEDFUNDS', 'MORTGAGE30US']

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

# includes mape() definition (fix from v1.17)
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
    adf_series = pd.to_numeric(group_data[target_col], errors='coerce').dropna()
    if len(adf_series) > 3:
        adf_p = adfuller(adf_series)[1]
        results['adf_p_value'], results['is_stationary'] = float(adf_p), bool(adf_p < 0.05)
    return results

# ---------- collinearity pruning (priority-keep) ----------
def prune_collinearity(X_df: pd.DataFrame, keep_priority: list, thr: float = 0.95) -> list:
    """
    Greedy pruning: keep features in priority order; drop any new feature whose
    absolute correlation with any kept feature exceeds `thr`.
    """
    if X_df.shape[1] <= 1:
        return X_df.columns.tolist()

    # Order columns by (1) explicit priority, then (2) remaining columns by variance (proxy for info)
    cols = X_df.columns.tolist()
    prio = [c for c in keep_priority if c in cols]
    rest = [c for c in cols if c not in prio]
    # variance sort descending to prefer informative features
    rest_sorted = sorted(rest, key=lambda c: np.nanvar(X_df[c].values), reverse=True)
    ordered = prio + rest_sorted

    kept = []
    # Precompute correlation matrix once on rows w/o NA
    Xc = X_df.dropna()
    if Xc.shape[0] < 3:
        return ordered[: min(6, len(ordered))]  # minimal fallback

    corrM = Xc.corr(numeric_only=True).fillna(0.0)

    for col in ordered:
        if not kept:
            kept.append(col)
            continue
        ok = True
        for k in kept:
            c = corrM.loc[k, col] if (k in corrM.index and col in corrM.columns) else 0.0
            if np.abs(c) > thr:
                ok = False
                break
        if ok:
            kept.append(col)
    return kept

# ------------------- data fetching -------------------------
def fetch_market_data(bq: BigQueryManager) -> pd.DataFrame:
    sql = """
    SELECT DATE_TRUNC(Date, MONTH) as date,
           AVG(UNRATE) as UNRATE,
           AVG(HSN1F) as HSN1F,
           AVG(FEDFUNDS) as FEDFUNDS,
           AVG(MORTGAGE30US) as MORTGAGE30US
    FROM tax_clnt_svcs.fred 
    WHERE Date >= '2023-01-01'
    GROUP BY 1 ORDER BY 1
    """
    df = bq.run_gbq_sql(sql, return_dataframe=True)
    if df.empty: return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').ffill()
    for c in MARKET_VARS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def fetch_credit_workload_drivers(bq: BigQueryManager) -> pd.DataFrame:
    sql = """
    SELECT MonthOfOrder AS date, TotalOrders
    FROM tax_clnt_svcs.view_cx_nontax_platforms_workload_drivers
    WHERE client_id = 'Credit'
    """
    df = bigquery_manager.run_gbq_sql(sql, return_dataframe=True)
    if df.empty: return pd.DataFrame(columns=["date", "TotalOrders"])
    df['date'] = pd.to_datetime(df['date'])
    df['TotalOrders'] = pd.to_numeric(df['TotalOrders'], errors='coerce')
    return df.groupby('date', as_index=False)['TotalOrders'].sum()

def fetch_credit_driver_forecasts(bq: BigQueryManager) -> pd.DataFrame:
    sql = """
    SELECT fx_date AS date, fx_vol AS TotalOrders_fx
    FROM tax_clnt_svcs.cx_nontax_platforms_workload_drivers_fx
    WHERE fx_status = 'A' AND client_id = 'Credit'
    """
    df = bigquery_manager.run_gbq_sql(sql, return_dataframe=True)
    if df.empty: return pd.DataFrame(columns=['date','TotalOrders_fx'])
    df['date'] = pd.to_datetime(df['date'])
    df['TotalOrders_fx'] = pd.to_numeric(df['TotalOrders_fx'], errors='coerce')
    return df

# ------------------- MAIN EXECUTION -------------------------
try:
    logger.info("Starting Credit Ticket Forecasting Script (v1.17.3)...")

    # Base tickets (monthly)
    TICKET_QUERY = """
    SELECT DATE_TRUNC(dt, MONTH) AS date,
           client_id,
           'tickets' AS vol_type,
           SUM(cases_opened) AS total_cases_opened
    FROM tax_clnt_svcs.view_nontax_agg
    WHERE client_id IN ('Credit - Customer Support', 'Credit - Tech Support')
    GROUP BY 1, 2, 3
    ORDER BY 2, 1;
    """
    df = bigquery_manager.run_gbq_sql(TICKET_QUERY, return_dataframe=True)
    if df.empty:
        raise RuntimeError("BQ returned zero rows for tickets")
    df['date'] = pd.to_datetime(df['date'])
    df['total_cases_opened'] = pd.to_numeric(df['total_cases_opened'], errors='coerce')

    # Market & drivers
    market_df = fetch_market_data(bigquery_manager)
    drv_df    = fetch_credit_workload_drivers(bigquery_manager)
    drv_fx_df = fetch_credit_driver_forecasts(bigquery_manager)

    if not market_df.empty: df = pd.merge(df, market_df, on='date', how='left')
    if not drv_df.empty:    df = pd.merge(df, drv_df,    on='date', how='left')

    # Sort and add calendar
    df = df.sort_values(['client_id','date'])
    df['month'] = df['date'].dt.month.astype(int)

    # Forward-fill ONLY market vars by client (avoid leaking the target)
    for c in MARKET_VARS:
        if c in df.columns:
            df[c] = df.groupby('client_id')[c].ffill()

    # Build lags (target + drivers)
    for lag in range(1, MAX_LAGS + 1):
        df[f'lag_{lag}']     = df.groupby('client_id')['total_cases_opened'].shift(lag)
        if 'TotalOrders' in df.columns:
            df[f'drv_lag_{lag}'] = df.groupby('client_id')['TotalOrders'].shift(lag)

    forecasts, audit_rows, modeling_data, stat_test_results, all_model_summaries, xgb_imp_rows = [], [], [], [], [], []

    # ----------------- per client loop -----------------
    for cid, client_df_orig in df.groupby('client_id'):
        logger.info(f"\n{'='*36}\nProcessing Client: {cid}\n{'='*36}")

        # Stats snapshot (ADF)
        stat_test_results.append(perform_statistical_tests(client_df_orig, cid))

        client_df = client_df_orig.copy()
        client_df['trend'] = np.arange(len(client_df), dtype=float)

        ts = client_df.set_index('date')['total_cases_opened'].asfreq('MS')

        # Build modeling table (NOTE: exclude total_cases_opened from features to avoid leakage)
        feat_all = pd.DataFrame({'y': ts}).join(
            client_df.set_index('date').drop(columns=['total_cases_opened'], errors='ignore')
        )

        # Clean to numeric + drop any NA rows
        cols_to_clean = [c for c in feat_all.columns if c not in ['client_id', 'vol_type', 'date']]
        for col in cols_to_clean:
            feat_all[col] = pd.to_numeric(feat_all[col], errors='coerce')
        feat_all.dropna(inplace=True)

        if len(feat_all) < VAL_LEN_6M + 2:
            logger.warning(f"Skipping {cid} - insufficient data for validation.")
            continue

        modeling_data.append(feat_all.assign(client_id=cid))

        # Split train/validation (last 6 months)
        train_df, val_df = feat_all.iloc[:-VAL_LEN_6M], feat_all.iloc[-VAL_LEN_6M:]
        y_tr_log = np.log1p(np.asarray(train_df['y'].values, dtype=float))

        # Candidate features (target removed)
        candidate_cols = [c for c in feat_all.columns if c not in ['y', 'client_id', 'vol_type', 'date']]

        X_tr = train_df[candidate_cols].astype(float)
        X_val = val_df[candidate_cols].astype(float)

        model_preds_val, model_meta = {}, {}

        # 1) XGBoost (log target)
        try:
            xgb = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
            ).fit(X_tr.values, y_tr_log)
            model_preds_val['XGBoost'] = safe_expm1(xgb.predict(X_val.values))
            model_meta['XGBoost'] = {'model': xgb, 'columns': candidate_cols}
            gain = xgb.get_booster().get_score(importance_type='gain')
            xgb_imp_rows.append({
                'client_id': cid,
                'feature_importance': json.dumps(sorted(gain.items(), key=lambda i: i[1], reverse=True))
            })
        except Exception as e:
            logger.warning(f"{cid}: XGBoost failed: {e}")

        # 2) OLS_Auto_Market (Lasso selection -> prune -> cap -> OLS refit) with guardrails
        try:
            # Scale for Lasso (selection only)
            scaler = StandardScaler(with_mean=True, with_std=True).fit(X_tr.values)
            X_tr_scaled = scaler.transform(X_tr.values)

            # Cross-validated Lasso to pick a compact set
            lasso = LassoCV(cv=min(3, max(2, len(train_df)//4)),
                            random_state=42, n_alphas=100).fit(X_tr_scaled, y_tr_log)
            sel_idx = np.where(np.abs(lasso.coef_) > 1e-8)[0].tolist()
            selected_feats = [candidate_cols[i] for i in sel_idx]

            # Guardrail: if only market vars selected, force 'trend'
            only_market = (len(selected_feats) > 0) and all(f in MARKET_VARS for f in selected_feats)
            if only_market and 'trend' in candidate_cols and 'trend' not in selected_feats:
                logger.info(f"{cid}: Only market vars selected — forcing 'trend' into OLS.")
                selected_feats.append('trend')

            # If Lasso selected nothing, fallback to strong seasonal/lag signal
            if not selected_feats:
                fallback = [c for c in ['lag_12', 'lag_1', 'lag_2', 'lag_3', 'TotalOrders', 'drv_lag_1', 'trend', 'month'] if c in candidate_cols]
                selected_feats = fallback[:4] if fallback else candidate_cols[:3]

            # Prune multicollinearity with priority-keep
            keep_priority = [c for c in ['lag_12', 'lag_1', 'lag_2', 'lag_3', 'TotalOrders', 'drv_lag_1', 'trend', 'month'] if c in candidate_cols]
            pruned = prune_collinearity(train_df[selected_feats].astype(float), keep_priority, thr=0.95)

            # Cap features to preserve DF: df_resid >= 5
            n_obs = len(train_df)
            max_feats = max(1, min(6, n_obs - 6))  # aim for at least ~6 residual DF
            if len(pruned) > max_feats:
                # Rank remaining by absolute correlation with y (log-space target for consistency)
                ytmp = y_tr_log
                feat_scores = []
                for c in pruned:
                    xv = train_df[c].values
                    cor = np.corrcoef(ytmp, xv)[0,1] if np.isfinite(xv).sum() >= 3 else 0.0
                    feat_scores.append((c, abs(float(cor)) if np.isfinite(cor) else 0.0))
                pruned = [c for c, _ in sorted(feat_scores, key=lambda t: t[1], reverse=True)[:max_feats]]

            # Make sure seasonal anchor is present if available
            if 'lag_12' in candidate_cols and 'lag_12' not in pruned:
                # If adding it would exceed max, replace the weakest
                if len(pruned) >= max_feats:
                    # drop weakest by abs corr
                    ytmp = y_tr_log
                    corrs = [(c, abs(np.corrcoef(ytmp, train_df[c].values)[0,1]) if np.isfinite(train_df[c].values).sum()>=3 else 0.0) for c in pruned]
                    weakest = sorted(corrs, key=lambda t: t[1])[0][0] if corrs else pruned[-1]
                    pruned = [c for c in pruned if c != weakest]
                pruned.append('lag_12')

            selected_feats = pruned

            # Refit OLS on selected_feats (pure float arrays)
            Xtr_sel_df = train_df[selected_feats].astype(float)
            Xval_sel_df = val_df[selected_feats].astype(float)

            Xtr_mat = sm.add_constant(Xtr_sel_df, has_constant='add').to_numpy(dtype=float)
            Xval_mat = sm.add_constant(Xval_sel_df, has_constant='add').to_numpy(dtype=float)

            ols_auto = sm.OLS(y_tr_log, Xtr_mat).fit()
            raw_val = ols_auto.predict(Xval_mat)
            model_preds_val['OLS_Auto_Market'] = safe_expm1(raw_val)
            model_meta['OLS_Auto_Market'] = {'model': ols_auto, 'features': selected_feats}

            try:
                all_model_summaries.append(
                    f"--- Client:{cid}, Model:OLS_Auto_Market ---\n{str(ols_auto.summary())}\n"
                )
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"{cid}: OLS_Auto_Market failed: {e}")

        # 3) SeasonalNaive3mGR & 4) SeasonalNaive3mDiffGR
        try:
            snaive_gr, snaive_diff = [], []
            full_ts_for_pos = df[df['client_id'] == cid].set_index('date')['total_cases_opened'].asfreq('MS')
            for d in val_df.index:
                base = ts.get(d - pd.DateOffset(months=12), np.nan)
                pos = full_ts_for_pos.index.get_loc(d)
                r  = recent_3m_growth(full_ts_for_pos, pos)
                rp = prior_year_3m_growth(full_ts_for_pos, pos)
                snaive_gr.append(base * (1 + r) if pd.notna(base) else np.nan)
                snaive_diff.append(base * (1 + (r - rp)) if pd.notna(base) else np.nan)
            model_preds_val['SeasonalNaive3mGR'] = np.asarray(snaive_gr, dtype=float)
            model_preds_val['SeasonalNaive3mDiffGR'] = np.asarray(snaive_diff, dtype=float)
        except Exception as e:
            logger.warning(f"{cid}: SeasonalNaive models failed: {e}")

        # Winner by 6m SMAPE
        if not model_preds_val:
            logger.warning(f"{cid}: No models produced validation predictions; skipping.")
            continue
        scores = {name: safe_reg_metrics(val_df['y'].values, pred)['smape'] for name, pred in model_preds_val.items()}
        winner = min(scores, key=scores.get)
        logger.info(f"✓ WINNER for {cid}: {winner} (SMAPE: {scores[winner]:.2f})")
        for name in model_preds_val:
            audit_rows.append({'client_id': cid, 'model': name, 'winner_model': winner, 'val_smape_6m': float(scores.get(name, np.nan))})

        # ---------------- Recursive Forecast (15 months) ----------------
        hist = deque(ts.tolist(), maxlen=60)
        drv_hist = deque(client_df['TotalOrders'].dropna().tolist(), maxlen=60)

        # last known market values (per-column) and trend
        mk_df = client_df[MARKET_VARS].copy()
        if not mk_df.empty:
            last_known_market = {mv: float(mk_df[mv].dropna().iloc[-1]) for mv in MARKET_VARS if mv in mk_df.columns and mk_df[mv].notna().any()}
        else:
            last_known_market = {}
        last_trend = float(client_df['trend'].iloc[-1]) if 'trend' in client_df.columns else float(len(client_df)-1)

        driver_fx_lookup = ({pd.Timestamp(d): float(v) for d, v in drv_fx_df[['date','TotalOrders_fx']].values}
                            if not drv_fx_df.empty else {})

        future_idx = pd.date_range(ts.index[-1], periods=FORECAST_HORIZON+1, freq='MS')[1:]

        xgb_model   = model_meta.get('XGBoost', {}).get('model')
        xgb_columns = model_meta.get('XGBoost', {}).get('columns')
        ols_meta    = model_meta.get('OLS_Auto_Market', {})

        for i, d in enumerate(future_idx, 1):
            # Driver for this month (available for ALL branches)
            last_driver = (drv_hist[-1] if len(drv_hist) else 0.0)
            current_total_orders = driver_fx_lookup.get(d, last_driver)

            pred = np.nan

            if winner in ['XGBoost', 'OLS_Auto_Market']:
                # Build feature row with BOTH target lags and driver lags
                row = {}
                for k in range(1, MAX_LAGS + 1):
                    row[f'lag_{k}']     = (hist[-k] if len(hist) >= k else (hist[-1] if len(hist) else 0.0))
                    row[f'drv_lag_{k}'] = (drv_hist[-k] if len(drv_hist) >= k else current_total_orders)

                row['month']       = int(d.month)
                row['TotalOrders'] = float(current_total_orders)
                row['trend']       = float(last_trend + i)
                for mv in MARKET_VARS:
                    if mv in df.columns:
                        row[mv] = float(last_known_market.get(mv, last_known_market.get(mv, 0.0)))

                if winner == 'XGBoost' and xgb_model is not None and xgb_columns is not None:
                    x_row_full = pd.DataFrame([{c: row.get(c, 0.0) for c in xgb_columns}])  # fill missing with 0.0
                    raw = xgb_model.predict(x_row_full.values)[0]
                    pred = float(safe_expm1(raw))
                elif winner == 'OLS_Auto_Market' and 'model' in ols_meta and 'features' in ols_meta:
                    feats = ols_meta['features']
                    x_df = pd.DataFrame([{c: row.get(c, 0.0) for c in feats}]).astype(float)  # fill missing with 0.0
                    X_row = sm.add_constant(x_df, has_constant='add').to_numpy(dtype=float)  # pure float
                    raw = float(ols_meta['model'].predict(X_row)[0])
                    pred = float(safe_expm1(raw))

            else:  # SeasonalNaive* winners
                base = hist[-12] if len(hist) >= 12 else (hist[-1] if len(hist) else 0.0)
                temp_series = pd.Series(list(hist))
                pos = len(temp_series)
                r = recent_3m_growth(temp_series, pos)
                if winner == 'SeasonalNaive3mGR':
                    pred = float(base * (1.0 + r))
                else:  # SeasonalNaive3mDiffGR
                    rp = prior_year_3m_growth(temp_series, pos)
                    pred = float(base * (1.0 + (r - rp)))

            # update histories for next step
            hist.append(pred)
            drv_hist.append(current_total_orders)  # ALWAYS push the driver, regardless of winner

            forecasts.append({
                'fx_date': d,
                'client_id': cid,
                'vol_type': 'tickets',
                'fx_vol': safe_round(pred),
                'fx_id': f"{FX_ID_PREFIX[winner]}_{STAMP:%Y%m%d}",
                'fx_status': "A",
                'load_ts': STAMP
            })

    # --- Output (local only) ---
    if forecasts:
        fx_df = pd.DataFrame(forecasts).sort_values(['client_id','fx_date'])
        fx_df.to_csv(LOCAL_CSV, index=False)
        pd.DataFrame(audit_rows).sort_values(['client_id']).to_csv(AUDIT_CSV, index=False)
        if modeling_data:
            pd.concat(modeling_data).to_csv(MODELING_DATA_CSV, index=False)
        if stat_test_results:
            pd.DataFrame(stat_test_results).to_csv(STATS_CSV, index=False)
        if all_model_summaries:
            with open(SUMMARIES_FILE, 'w') as f:
                f.write("\n\n".join(all_model_summaries))
        if xgb_imp_rows:
            pd.DataFrame(xgb_imp_rows).to_csv(XGB_VIZ_CSV, index=False)

        logger.info("✓ All local audit files saved.")
        logger.info("✓ Credit ticket forecasting completed successfully (local only).")
    else:
        logger.warning("No forecasts were generated.")

except Exception as exc:
    email_manager.handle_error("Credit Ticket Forecasting Script Failure", exc, is_test=True)
