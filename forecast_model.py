"""
Credit Tickets - Monthly Volume Forecast — v1.16 (Object-Dtype OLS Fix + 2024 Start + NaN Guard)

What’s fixed / added vs v1.15:
  • **Hard OLS dtype fix**: always pass pure float **NumPy arrays** (endog & exog) into statsmodels OLS
    for both training and prediction, eliminating the "Pandas data cast to numpy dtype of object" error.
  • **Train window**: only train on rows **>= 2024-01-01** and require **lag_12** to exist.
  • **6-month validation** per client.
  • **NaN guard**: robust cleaning before modeling; XGBoost trained on float arrays; OLS_Auto_Market uses
    Lasso-based selection and OLS refit on floats. Guardrail ensures 'trend' gets added if only market vars selected.
  • **Forecast loop cleanups**: consistent feature assembly, safe TotalOrders handling, and no reliance on
    scope-leaked variables.
  • Outputs: forecasts, audit, modeling snapshot, stats tests, XGB importance, and model summaries.
"""

# ------------------- standard imports -----------------------
import os
import sys
import warnings
import pytz
from datetime import datetime
from collections import deque
import json

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
TRAIN_START_DATE   = pd.Timestamp("2024-01-01")  # <-- enforce 2024 start (and lag_12 availability)
STAMP              = datetime.now(pytz.timezone("America/Chicago"))

FX_ID_PREFIX = {
    "XGBoost": "xgb_ticket",
    "OLS_Auto_Market": "ols_auto_mkt_ticket",
    "SeasonalNaive3mGR": "seasonal_naive3mgr_ticket"
}

MARKET_VARS = ['UNRATE', 'HSN1F', 'FEDFUNDS', 'MORTGAGE30US']

# ------------------- metric & helpers -------------------------
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
    """Growth proxy using last 3 vs prior 3 (shifted to avoid peeking)."""
    if ts is None or len(ts) < 6: return 0.0
    ts_shifted = ts.shift(1)
    if ts_shifted.isna().sum() > len(ts_shifted) - 6: return 0.0
    cur3  = ts_shifted.iloc[-3:].values
    prev3 = ts_shifted.iloc[-6:-3].values
    prev_mean = float(np.mean(prev3)) if len(prev3) else 0.0
    return (float(np.mean(cur3)) / prev_mean - 1.0) if prev_mean > 0 else 0.0

def perform_statistical_tests(group_data: pd.DataFrame, client_id: str):
    results = {'client_id': client_id}
    target_col = 'total_cases_opened'
    adf_series = pd.to_numeric(group_data[target_col], errors='coerce').dropna()
    if len(adf_series) > 3:
        adf_p = adfuller(adf_series)[1]
        results['adf_p_value'], results['is_stationary'] = float(adf_p), bool(adf_p < 0.05)
    return results

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
    logger.info("Starting Credit Ticket Forecasting Script (v1.16)...")

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

    # Calendar features
    df = df.sort_values(['client_id','date'])
    df['month']     = df['date'].dt.month.astype(int)
    df['sin_month'] = np.sin(2*np.pi*df['month']/12.0)
    df['cos_month'] = np.cos(2*np.pi*df['month']/12.0)

    # Forward-fill ONLY market vars by client/date (avoid leaking target)
    for c in MARKET_VARS:
        if c in df.columns:
            df[c] = df.groupby('client_id')[c].ffill()

    # Build lags (target + drivers)
    for lag in range(1, MAX_LAGS + 1):
        df[f'lag_{lag}']     = df.groupby('client_id')['total_cases_opened'].shift(lag)
        if 'TotalOrders' in df.columns:
            df[f'drv_lag_{lag}'] = df.groupby('client_id')['TotalOrders'].shift(lag)

    # Enforce train window and lag_12 availability
    df = df[(df['date'] >= TRAIN_START_DATE) & (df['lag_12'].notna())].copy()

    forecasts, audit_rows, modeling_data, stat_test_results, all_model_summaries, xgb_imp_rows = [], [], [], [], [], []

    # ----------------- per client loop -----------------
    for cid, client_df in df.groupby('client_id'):
        logger.info(f"\n{'='*36}\nProcessing Client: {cid}\n{'='*36}")

        client_df = client_df.sort_values('date').copy()
        client_df['trend'] = np.arange(len(client_df), dtype=float)

        # Stats snapshot (ADF)
        stat_test_results.append(perform_statistical_tests(client_df, cid))

        # Candidate features (everything numeric except identifiers)
        candidate_cols = [c for c in client_df.columns
                          if c not in ['client_id','vol_type','date','total_cases_opened']]

        # Build feature frame (index = date)
        feat_all = pd.DataFrame({'y': client_df.set_index('date')['total_cases_opened']}) \
                    .join(client_df.set_index('date')[candidate_cols])

        # Robust cleaning: coerce numerics + drop rows with ANY NaN
        for c in feat_all.columns:
            if c not in ['client_id','vol_type']:  # all are numeric columns here
                feat_all[c] = pd.to_numeric(feat_all[c], errors='coerce')
        feat_all.dropna(inplace=True)

        # Require at least 6 months for validation
        if len(feat_all) < VAL_LEN_6M + 2:
            logger.warning(f"Skipping {cid} - insufficient data for validation after cleaning.")
            continue

        # Save modeling snapshot
        modeling_data.append(feat_all.assign(client_id=cid))

        # Train/Validation split (last 6 months for validation)
        train_df = feat_all.iloc[:-VAL_LEN_6M]
        val_df   = feat_all.iloc[-VAL_LEN_6M:]

        # Ensure we have some predictors left
        candidate_cols = [c for c in candidate_cols if c in feat_all.columns]
        if not candidate_cols:
            logger.warning(f"{cid}: No candidate features after cleaning; skipping.")
            continue

        # Response (log-space) as pure float array
        y_tr_log = np.log1p(np.asarray(train_df['y'].values, dtype=float))

        # Design matrices (float DataFrames -> arrays as needed)
        X_tr_df = train_df[candidate_cols].astype(float)
        X_val_df = val_df[candidate_cols].astype(float)

        model_preds_val, model_meta = {}, {}

        # 1) XGBoost (log-target); tolerate NaN inherently, but our cleaning removed NaN anyway
        try:
            xgb = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
            )
            xgb.fit(X_tr_df.values, y_tr_log)
            pred_val = safe_expm1(xgb.predict(X_val_df.values))
            model_preds_val['XGBoost'] = pred_val
            model_meta['XGBoost'] = {'model': xgb, 'features': candidate_cols}
            gain = xgb.get_booster().get_score(importance_type='gain')
            xgb_imp_rows.append({
                'client_id': cid,
                'feature_importance': json.dumps(sorted(gain.items(), key=lambda i: i[1], reverse=True))
            })
        except Exception as e:
            logger.warning(f"{cid}: XGBoost failed: {e}")

        # 2) OLS_Auto_Market (Lasso selection -> OLS refit) with guardrail
        try:
            # Scale for Lasso selection (no NaNs at this point)
            scaler = StandardScaler(with_mean=True, with_std=True).fit(X_tr_df.values)
            X_tr_scaled = scaler.transform(X_tr_df.values)

            lasso = LassoCV(cv=min(3, max(2, len(train_df)//4)),
                            random_state=42, n_alphas=100).fit(X_tr_scaled, y_tr_log)
            sel_idx = np.where(np.abs(lasso.coef_) > 1e-8)[0].tolist()
            selected_feats = [candidate_cols[i] for i in sel_idx]

            # Guardrail: if only market variables selected, force-in 'trend'
            only_market = (len(selected_feats) > 0) and all(f in MARKET_VARS for f in selected_feats)
            if only_market and 'trend' in candidate_cols and 'trend' not in selected_feats:
                logger.info(f"{cid}: Only market vars selected — forcing 'trend' into OLS.")
                selected_feats.append('trend')

            # Fallback if nothing selected
            if not selected_feats:
                fallback = [c for c in ['lag_12', 'trend'] if c in candidate_cols]
                if not fallback:
                    # strongest single corr with y_tr_log (very small data fallback)
                    corrs = {}
                    for c in candidate_cols:
                        xv = X_tr_df[c].values
                        if np.isfinite(xv).sum() >= 3:
                            try:
                                cor = np.corrcoef(y_tr_log, xv)[0,1]
                            except Exception:
                                cor = 0.0
                            corrs[c] = abs(float(cor)) if np.isfinite(cor) else 0.0
                    fallback = [max(corrs, key=corrs.get)] if corrs else []
                selected_feats = fallback

            if selected_feats:
                Xtr_sel_df = train_df[selected_feats].astype(float)
                Xval_sel_df = val_df[selected_feats].astype(float)

                # --- CRITICAL: convert to pure float NumPy arrays for statsmodels ---
                Xtr_mat = sm.add_constant(Xtr_sel_df, has_constant='add').to_numpy(dtype=float)
                Xval_mat = sm.add_constant(Xval_sel_df, has_constant='add').to_numpy(dtype=float)

                ols_auto = sm.OLS(y_tr_log, Xtr_mat).fit()
                raw_val = ols_auto.predict(Xval_mat)
                pred_val = safe_expm1(raw_val)

                model_preds_val['OLS_Auto_Market'] = pred_val
                model_meta['OLS_Auto_Market'] = {
                    'model': ols_auto,
                    'features': selected_feats
                }
                # Save text summary
                try:
                    all_model_summaries.append(
                        f"--- Client:{cid}, Model:OLS_Auto_Market ---\n{str(ols_auto.summary())}\n"
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"{cid}: OLS_Auto_Market failed: {e}")

        # 3) SeasonalNaive3mGR baseline
        try:
            ts = client_df.set_index('date')['total_cases_opened']
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
        valid_scores = {k: v for k, v in scores.items() if np.isfinite(v)}
        if not valid_scores:
            logger.warning(f"{cid}: No finite validation scores; skipping.")
            continue

        winner = min(valid_scores, key=valid_scores.get)
        logger.info(f"✓ WINNER for {cid}: {winner} (6m SMAPE: {valid_scores[winner]:.2f})")
        audit_rows.append({'client_id': cid, 'winner_model': winner, 'smape_6m': float(valid_scores[winner])})

        # ---------------- Recursive Forecast (15 months) ----------------
        # histories
        ts_hist = deque(client_df['total_cases_opened'].tolist(), maxlen=60)
        drv_hist = deque(client_df['TotalOrders'].dropna().tolist(), maxlen=60)

        # last known markets (held flat)
        mk_hist_df = client_df[MARKET_VARS].dropna(axis=1, how='all')[MARKET_VARS] if MARKET_VARS else pd.DataFrame()
        last_mkt = (mk_hist_df.iloc[-1].to_dict() if not mk_hist_df.empty else {})

        last_trend = float(client_df['trend'].iloc[-1]) if 'trend' in client_df.columns else float(len(client_df)-1)
        driver_fx_lookup = ({pd.Timestamp(d): float(v) for d, v in drv_fx_df[['date','TotalOrders_fx']].values}
                            if not drv_fx_df.empty else {})

        future_idx = pd.date_range(client_df['date'].iloc[-1], periods=FORECAST_HORIZON+1, freq='MS')[1:]

        # model assets for inference
        xgb_model   = model_meta.get('XGBoost', {}).get('model')
        xgb_cols    = model_meta.get('XGBoost', {}).get('features')
        ols_meta    = model_meta.get('OLS_Auto_Market', None)

        for step, d in enumerate(future_idx, start=1):
            # Compute this month’s driver level
            last_driver = drv_hist[-1] if len(drv_hist) else 0.0
            current_total_orders = driver_fx_lookup.get(d, last_driver)

            # Assemble row dict of features (target/driver lags 1..12, month, trend, TotalOrders, market vars)
            row = {}
            # target & driver lags
            for k in range(1, MAX_LAGS + 1):
                row[f'lag_{k}']     = (ts_hist[-k] if len(ts_hist) >= k else (ts_hist[-1] if len(ts_hist) else 0.0))
                row[f'drv_lag_{k}'] = (drv_hist[-k] if len(drv_hist) >= k else current_total_orders)

            # calendar/driver/market/trend
            m = int(d.month)
            row['month']       = m
            row['sin_month']   = float(np.sin(2*np.pi*m/12.0))
            row['cos_month']   = float(np.cos(2*np.pi*m/12.0))
            row['TotalOrders'] = float(current_total_orders)
            row['trend']       = float(last_trend + step)
            for mv in MARKET_VARS:
                if mv in df.columns:
                    row[mv] = float(last_mkt.get(mv, np.nan))

            pred = np.nan

            if winner == 'XGBoost' and xgb_model is not None and xgb_cols is not None:
                x_row_df = pd.DataFrame([{c: row.get(c, np.nan) for c in xgb_cols}])
                raw = xgb_model.predict(x_row_df.values)[0]
                pred = float(safe_expm1(raw))

            elif winner == 'OLS_Auto_Market' and ols_meta is not None:
                feats = ols_meta['features']
                x_df = pd.DataFrame([{c: row.get(c, np.nan) for c in feats}]).astype(float)
                X_row = sm.add_constant(x_df, has_constant='add').to_numpy(dtype=float)  # <-- pure float array
                raw = float(ols_meta['model'].predict(X_row)[0])
                pred = float(safe_expm1(raw))

            elif winner == 'SeasonalNaive3mGR':
                base = (ts_hist[-12] if len(ts_hist) >= 12 else (ts_hist[-1] if len(ts_hist) else 0.0))
                r = recent_3m_growth(pd.Series(list(ts_hist)))
                pred = float(base * (1.0 + r))

            # update histories
            ts_hist.append(pred)
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
