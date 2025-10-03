"""
Credit Tickets - Monthly Volume Forecast — v1.13 (Lasso NaN Fix + Guardrails + 2024 Start)

Key features:
  • Train window: rows >= 2024-01-01 with lag_12 available.
  • 6-month validation per client.
  • Models:
      - XGBoost (log target).
      - OLS_Auto_Market: LassoCV selection (with median imputer + scaler) -> OLS refit on selected
        features (with its own median imputer). Guardrail: if selection is only market vars, add 'trend'.
      - SeasonalNaive3mGR baseline.
  • Full driver & target lag coverage (1..12) in train and inference.
  • Robust dtype/NaN handling; consistent feature assembly at forecast time.
"""

# ------------------- standard imports -----------------------
import os
import sys
import warnings
import pytz
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor

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

# ------------------- forecast parameters --------------------
FORECAST_HORIZON   = 15
MAX_LAGS           = 12
VAL_LEN_6M         = 6
TRAIN_START_DATE   = pd.Timestamp("2024-01-01")
STAMP              = datetime.now(pytz.timezone("America/Chicago"))

FX_ID_PREFIX = {
    "XGBoost": "xgb_ticket",
    "OLS_Auto_Market": "ols_auto_mkt_ticket",
    "SeasonalNaive3mGR": "seasonal_naive3mgr_ticket"
}

# ------------------- metrics & helpers -------------------------
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

def safe_reg_metrics(y_true, y_pred):
    if y_pred is None: return dict(smape=np.nan)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any(): return dict(smape=np.nan)
    return dict(smape=smape(y_true[mask], y_pred[mask]))

def recent_3m_growth(ts: pd.Series) -> float:
    """YoY-like growth proxy using last 3 vs prior 3; shift(1) to avoid peeking."""
    if ts is None or len(ts) < 6: return 0.0
    ts_shifted = ts.shift(1)
    if ts_shifted.isna().sum() > len(ts_shifted) - 6: return 0.0
    cur3  = ts_shifted.iloc[-3:].values
    prev3 = ts_shifted.iloc[-6:-3].values
    prev_mean = float(np.mean(prev3)) if len(prev3) else 0.0
    return (float(np.mean(cur3)) / prev_mean - 1.0) if prev_mean > 0 else 0.0

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
    df = bigquery_manager.run_gbq_sql(sql, return_dataframe=True)
    if df.empty: return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').ffill()
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
    logger.info("Starting Credit Ticket Forecasting Script (v1.13)...")

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
    if df.empty: raise RuntimeError("BQ returned zero rows for tickets")
    df['date'] = pd.to_datetime(df['date'])
    df['total_cases_opened'] = pd.to_numeric(df['total_cases_opened'], errors='coerce')

    # Market & drivers
    market_df = fetch_market_data(bigquery_manager)
    drv_df    = fetch_credit_workload_drivers(bigquery_manager)
    drv_fx_df = fetch_credit_driver_forecasts(bigquery_manager)

    if not market_df.empty: df = pd.merge(df, market_df, on='date', how='left')
    if not drv_df.empty:    df = pd.merge(df, drv_df,    on='date', how='left')

    # Calendar features
    df['month']     = df['date'].dt.month.astype(int)
    df['sin_month'] = np.sin(2*np.pi*df['month']/12.0)
    df['cos_month'] = np.cos(2*np.pi*df['month']/12.0)

    # Sort rows; forward-fill ONLY market columns by date order
    df = df.sort_values(['client_id','date'])
    for col in ['UNRATE','HSN1F','FEDFUNDS','MORTGAGE30US']:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Build lags (target + drivers)
    for lag in range(1, MAX_LAGS + 1):
        df[f'lag_{lag}']     = df.groupby('client_id')['total_cases_opened'].shift(lag)
        if 'TotalOrders' in df.columns:
            df[f'drv_lag_{lag}'] = df.groupby('client_id')['TotalOrders'].shift(lag)

    # Require training rows >= 2024-01-01 AND lag_12 available
    df = df[(df['date'] >= TRAIN_START_DATE) & (df['lag_12'].notna())].copy()

    forecasts, audit_rows, modeling_rows = [], [], []
    market_vars = ['UNRATE','HSN1F','FEDFUNDS','MORTGAGE30US']

    # ----------------- per client loop -----------------
    for cid, client_df in df.groupby('client_id'):
        logger.info(f"\n{'='*36}\nProcessing Client: {cid}\n{'='*36}")

        client_df = client_df.sort_values('date').copy()
        client_df['trend'] = np.arange(len(client_df), dtype=float)

        # enforce numeric dtypes
        num_cols = [c for c in client_df.columns if c not in ['client_id','vol_type','date']]
        for c in num_cols:
            client_df[c] = pd.to_numeric(client_df[c], errors='coerce')

        ts = client_df.set_index('date')['total_cases_opened']

        # Candidate features for learning
        candidate_cols = [c for c in client_df.columns
                          if c not in ['client_id','vol_type','date','total_cases_opened']]
        # full frame with target + candidates
        feat_all = pd.DataFrame({'y': ts}).join(client_df.set_index('date')[candidate_cols])

        # Ensure we still keep only rows with all target lags available
        needed = ['y'] + [f'lag_{k}' for k in range(1, MAX_LAGS+1)]
        feat_all = feat_all.dropna(subset=[c for c in needed if c in feat_all.columns])

        if len(feat_all) < VAL_LEN_6M + 2:
            logger.warning(f"{cid}: insufficient rows after filtering; skipping.")
            continue

        # Split: last 6 months for validation
        train_df = feat_all.iloc[:-VAL_LEN_6M]
        val_df   = feat_all.iloc[-VAL_LEN_6M:]

        y_tr_log = np.log1p(np.asarray(train_df['y'].values, dtype=float))

        # Assemble training/validation X matrices (float), allow NaNs (to be imputed where needed)
        X_tr_df = train_df[candidate_cols].astype(float)
        X_val_df = val_df[candidate_cols].astype(float)

        model_preds_val, model_meta = {}, {}

        # 1) XGBoost (log-target); XGBoost tolerates NaN
        try:
            xgb = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=100, learning_rate=0.05, max_depth=3,
                random_state=42
            ).fit(X_tr_df.values, y_tr_log)
            preds = safe_expm1(xgb.predict(X_val_df.values))
            model_preds_val['XGBoost'] = preds
            model_meta['XGBoost'] = {'model': xgb, 'columns': candidate_cols}
        except Exception as e:
            logger.warning(f"{cid}: XGBoost failed: {e}")

        # 2) OLS_Auto_Market (Lasso selection -> OLS refit), with imputers
        try:
            # Impute (median) then scale for LassoCV selection
            imp_all   = SimpleImputer(strategy='median')
            X_tr_imp  = imp_all.fit_transform(X_tr_df)
            X_val_imp = imp_all.transform(X_val_df)

            scaler = StandardScaler(with_mean=True, with_std=True).fit(X_tr_imp)
            X_tr_scaled = scaler.transform(X_tr_imp)

            # LassoCV selection on scaled data
            lasso = LassoCV(cv=min(3, max(2, len(train_df)//4)),
                            random_state=42, n_alphas=100).fit(X_tr_scaled, y_tr_log)
            sel_idx = np.where(np.abs(lasso.coef_) > 1e-8)[0].tolist()
            selected_feats = [candidate_cols[i] for i in sel_idx]

            # Guardrail: if only market variables, force 'trend'
            only_market = (len(selected_feats) > 0) and all(f in market_vars for f in selected_feats)
            if only_market and 'trend' in candidate_cols and 'trend' not in selected_feats:
                logger.info(f"{cid}: Only market features selected — forcing 'trend' into OLS.")
                selected_feats.append('trend')

            # Fallback if nothing selected: use ['lag_12','trend'] if present
            if not selected_feats:
                fallback = [c for c in ['lag_12','trend'] if c in candidate_cols]
                if not fallback:
                    # last resort: strongest single feature by correlation with y
                    corrs = {}
                    ytmp = y_tr_log
                    for c in candidate_cols:
                        xv = X_tr_df[c].values
                        if np.isfinite(xv).sum() >= 3:
                            try:
                                cor = np.corrcoef(ytmp, np.nan_to_num(xv, nan=np.nanmedian(xv)))[0,1]
                            except Exception:
                                cor = 0.0
                            corrs[c] = abs(float(cor)) if np.isfinite(cor) else 0.0
                    fallback = [max(corrs, key=corrs.get)] if corrs else []
                selected_feats = fallback

            # Refit OLS on selected features with its own median imputer
            imp_sel   = SimpleImputer(strategy='median')
            Xtr_sel   = imp_sel.fit_transform(train_df[selected_feats].astype(float))
            Xval_sel  = imp_sel.transform(  val_df[selected_feats].astype(float))

            Xtr_ols = sm.add_constant(pd.DataFrame(Xtr_sel, columns=selected_feats), has_constant='add').to_numpy(dtype=float)
            Xval_ols = sm.add_constant(pd.DataFrame(Xval_sel, columns=selected_feats), has_constant='add').to_numpy(dtype=float)

            ols_auto = sm.OLS(y_tr_log, Xtr_ols).fit()
            preds = safe_expm1(ols_auto.predict(Xval_ols))

            model_preds_val['OLS_Auto_Market'] = preds
            model_meta['OLS_Auto_Market'] = {
                'model': ols_auto,
                'features': selected_feats,
                'imputer': imp_sel,           # save imputer for inference
            }
        except Exception as e:
            logger.warning(f"{cid}: OLS_Auto_Market failed: {e}")

        # 3) SeasonalNaive3mGR baseline
        try:
            snaive = []
            for d in val_df.index:
                base = ts.get(d - pd.DateOffset(months=12), np.nan)
                r = recent_3m_growth(ts.loc[:d])
                snaive.append(base * (1 + r) if np.isfinite(base) else np.nan)
            model_preds_val['SeasonalNaive3mGR'] = np.asarray(snaive, dtype=float)
        except Exception as e:
            logger.warning(f"{cid}: SeasonalNaive3mGR failed: {e}")

        if not model_preds_val:
            logger.warning(f"{cid}: No models produced validation predictions; skipping.")
            continue

        # Select winner by 6m SMAPE
        yv = np.asarray(val_df['y'].values, dtype=float)
        scores = {name: safe_reg_metrics(yv, np.asarray(p, dtype=float))['smape']
                  for name, p in model_preds_val.items()}
        valid_scores = {k:v for k,v in scores.items() if np.isfinite(v)}
        if not valid_scores:
            logger.warning(f"{cid}: No finite validation scores; skipping.")
            continue

        winner = min(valid_scores, key=valid_scores.get)
        logger.info(f"✓ WINNER for {cid}: {winner} (6m SMAPE: {valid_scores[winner]:.2f})")
        audit_rows.append({'client_id': cid, 'winner_model': winner, 'smape_6m': float(valid_scores[winner])})

        # ---------------- Recursive Forecast (15 months) ----------------
        hist = deque(ts.tolist(), maxlen=60)
        drv_hist = deque(client_df['TotalOrders'].dropna().tolist(), maxlen=60)

        market_hist_df = client_df[market_vars].dropna(axis=1, how='all')[market_vars] if market_vars else pd.DataFrame()
        last_known_market = (market_hist_df.iloc[-1].to_dict()
                             if not market_hist_df.empty else {})

        last_trend = float(client_df['trend'].iloc[-1]) if 'trend' in client_df.columns else float(len(client_df)-1)
        driver_fx_lookup = ({pd.Timestamp(d): float(v) for d, v in drv_fx_df[['date','TotalOrders_fx']].values}
                            if not drv_fx_df.empty else {})

        future_idx = pd.date_range(ts.index[-1], periods=FORECAST_HORIZON+1, freq='MS')[1:]

        # handy model metadata
        xgb_model   = model_meta.get('XGBoost', {}).get('model')
        xgb_columns = model_meta.get('XGBoost', {}).get('columns')
        ols_meta    = model_meta.get('OLS_Auto_Market', {})

        for step, d in enumerate(future_idx, start=1):
            # assemble future feature row skeleton
            current_total_orders = driver_fx_lookup.get(d, (drv_hist[-1] if len(drv_hist) else 0.0))

            # target & driver lags for 1..12
            row = {}
            for k in range(1, MAX_LAGS+1):
                row[f'lag_{k}']     = (hist[-k] if len(hist) >= k else (hist[-1] if len(hist) else 0.0))
                row[f'drv_lag_{k}'] = (drv_hist[-k] if len(drv_hist) >= k else current_total_orders)

            # calendar + drivers + trend
            row['month']       = int(d.month)
            row['sin_month']   = float(np.sin(2*np.pi*row['month']/12.0))
            row['cos_month']   = float(np.cos(2*np.pi*row['month']/12.0))
            row['TotalOrders'] = float(current_total_orders)
            row['trend']       = float(last_trend + step)

            # market variables: hold last known values
            for mv in market_vars:
                if mv in df.columns:
                    row[mv] = float(last_known_market.get(mv, np.nan))

            # rolling means (prior-only style) if they exist in training columns (not strictly required here)
            # (kept minimal since selection handles this)

            # Predict per winner
            pred = np.nan

            if winner == 'XGBoost' and xgb_model is not None and xgb_columns is not None:
                # Build full feature vector in training order; allow NaN
                x_row_full = pd.DataFrame([{c: row.get(c, np.nan) for c in xgb_columns}])
                raw = xgb_model.predict(x_row_full.values)[0]
                pred = float(safe_expm1(raw))

            elif winner == 'OLS_Auto_Market' and 'model' in ols_meta and 'features' in ols_meta and 'imputer' in ols_meta:
                feats = ols_meta['features']
                imp   = ols_meta['imputer']
                x_df  = pd.DataFrame([{c: row.get(c, np.nan) for c in feats}]).astype(float)
                X_row_imp = imp.transform(x_df.values)
                X_row = sm.add_constant(pd.DataFrame(X_row_imp, columns=feats), has_constant='add').to_numpy(dtype=float)
                raw = float(ols_meta['model'].predict(X_row)[0])
                pred = float(safe_expm1(raw))

            elif winner == 'SeasonalNaive3mGR':
                base = (hist[-12] if len(hist) >= 12 else (hist[-1] if len(hist) else 0.0))
                r = recent_3m_growth(pd.Series(list(hist)))
                pred = float(base * (1.0 + r))

            # update deques
            hist.append(pred)
            drv_hist.append(current_total_orders)

            forecasts.append({
                'fx_date': d,
                'client_id': cid,
                'vol_type': 'tickets',
                'fx_vol': safe_round(pred),
                'fx_id': f"{FX_ID_PREFIX[winner]}_{STAMP:%Y%m%d}",
                'fx_status': "A",
                'load_ts': STAMP
            })

        # modeling snapshot for audit
        snap_cols = ['total_cases_opened','TotalOrders','month','sin_month','cos_month','trend'] \
                    + [f'lag_{i}' for i in (1,3,6,9,12)] + [f'drv_lag_{i}' for i in (1,3,6,9,12)] \
                    + [c for c in market_vars if c in client_df.columns]
        snap_cols = [c for c in snap_cols if c in client_df.columns]
        snap = client_df.set_index('date')[snap_cols].copy()
        snap['client_id'] = cid
        modeling_rows.append(snap.reset_index())

    # --- Output (local only) ---
    if forecasts:
        fx_df = pd.DataFrame(forecasts).sort_values(['client_id','fx_date'])
        fx_df.to_csv(LOCAL_CSV, index=False)
        pd.DataFrame(audit_rows).sort_values(['client_id']).to_csv(AUDIT_CSV, index=False)
        pd.concat(modeling_rows, ignore_index=True).to_csv(MODELING_DATA_CSV, index=False)
        logger.info(f"✓ Forecast files saved:\n  {LOCAL_CSV}\n  {AUDIT_CSV}\n  {MODELING_DATA_CSV}")
        logger.info("✓ Credit ticket forecasting completed successfully (local only).")
    else:
        logger.warning("No forecasts were generated.")

except Exception as exc:
    email_manager.handle_error("Credit Ticket Forecasting Script Failure", exc, is_test=True)
