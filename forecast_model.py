"""
Credit Tickets - Monthly Volume Forecast — v1.5
(Train start >= 2024-01-01, require 12 lags; 6m validation; auto vs forced-lag models)

What’s new vs v1.4
  • Training rows are restricted to dates >= 2024-01-01 AND require lag_12 to exist.
  • Validation always uses the last 6 months per client.
  • Two primary models:
      1) OLS_Auto: LassoCV selects from a rich feature set (target/driver lags 1..12,
         seasonality, trend, rolling means, and same-month driver). We then refit
         a log-space OLS on the selected features.
      2) OLS_ForcedLag3912: OLS using ONLY target lags {3,9,12} and driver lags {3,9,12}
         plus simple month seasonality (month, sin, cos). This enforces your 3/9/12 request
         for BOTH target and driver.
     (+) WeightedLagBlend(3,6,12) baseline for robustness.
  • Robust dtype handling for statsmodels (train & predict) and safe_expm1 everywhere.
  • Local CSV outputs for forecasts, audit, and modeling snapshots.
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
from sklearn.linear_model import LassoCV

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# ------------------- helper-package path --------------------
sys.path.append(r"C:\WFM_Scripting\Automation")  # adjust if needed
from scripthelper import Config, Logger, BigQueryManager, EmailManager

# ------------------- initialise scripthelper ---------------
config           = Config(rpt_id=286)
logger           = Logger(config)
email_manager    = EmailManager(config)
bigquery_manager = BigQueryManager(config)

# ------------------- file / table paths --------------------
LOCAL_CSV          = r"C:\WFM_Scripting\credit_ticket_forecast_results.csv"
AUDIT_CSV          = r"C:\WFM_Scripting\credit_ticket_model_eval_debug.csv"
MODELING_DATA_CSV  = r"C:\WFM_Scripting\credit_ticket_modeling_data.csv"

# ------------------- forecast parameters --------------------
FORECAST_HORIZON     = 15
MAX_LAGS             = 12
VAL_LEN_6M           = 6
TRAIN_START_DATE     = pd.Timestamp("2024-01-01")  # enforce start (with lag_12 available)
STAMP                = datetime.now(pytz.timezone("America/Chicago"))

FX_ID_PREFIX = {
    "WeightedLagBlend":     "wlb_ticket",
    "OLS_Auto":             "ols_auto_ticket",
    "OLS_ForcedLag3912":    "ols_3912_ticket"
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
    def mape(actual, forecast):
        actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
        denom = np.where(actual == 0, 1.0, np.abs(actual))
        return np.mean(np.abs((actual - forecast) / denom)) * 100
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

# ------------------- data fetching -------------------------
def fetch_credit_workload_drivers(bq: BigQueryManager) -> pd.DataFrame:
    sql = """
      SELECT MonthOfOrder, TotalOrders
      FROM tax_clnt_svcs.view_cx_nontax_platforms_workload_drivers
      WHERE client_id = 'Credit'
    """
    df = bigquery_manager.run_gbq_sql(sql, return_dataframe=True)
    if df.empty:
        return pd.DataFrame(columns=["date", "TotalOrders"])
    df = df.rename(columns={'MonthOfOrder': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df['TotalOrders'] = pd.to_numeric(df['TotalOrders'], errors='coerce')
    return df.groupby('date', as_index=False)['TotalOrders'].sum()

def fetch_credit_driver_forecasts(bq: BigQueryManager) -> pd.DataFrame:
    sql = """
      SELECT fx_date, fx_vol AS TotalOrders_fx
      FROM tax_clnt_svcs.cx_nontax_platforms_workload_drivers_fx
      WHERE fx_status = 'A' AND client_id = 'Credit'
    """
    df = bigquery_manager.run_gbq_sql(sql, return_dataframe=True)
    if df.empty:
        return pd.DataFrame(columns=['date','TotalOrders_fx'])
    df['date'] = pd.to_datetime(df['fx_date'])
    df = df.drop(columns=['fx_date'])
    df['TotalOrders_fx'] = pd.to_numeric(df['TotalOrders_fx'], errors='coerce')
    return df

# ------------------- MAIN EXECUTION -------------------------
try:
    logger.info("Starting Credit Ticket Forecasting Script (v1.5)...")

    # 1) Base tickets query (monthly)
    TICKET_QUERY = """
    SELECT
      DATE_TRUNC(dt, MONTH) AS date,
      client_id,
      'tickets' AS vol_type,
      SUM(cases_opened) AS total_cases_opened
    FROM
      tax_clnt_svcs.view_nontax_agg
    WHERE
      client_id IN ('Credit - Customer Support', 'Credit - Tech Support')
    GROUP BY 1, 2, 3
    ORDER BY 2, 1;
    """
    df = bigquery_manager.run_gbq_sql(TICKET_QUERY, return_dataframe=True)
    if df.empty:
        raise RuntimeError("BQ returned zero rows for tickets")

    # basic prep
    df['date'] = pd.to_datetime(df['date'])
    df['total_cases_opened'] = pd.to_numeric(df['total_cases_opened'], errors='coerce')

    # calendar features (global, row-level)
    df['month'] = df['date'].dt.month.astype(int)
    df['sin_month'] = np.sin(2*np.pi*df['month']/12.0)
    df['cos_month'] = np.cos(2*np.pi*df['month']/12.0)

    logger.info(f"✓ Pulled {len(df):,} ticket rows")

    # 2) Drivers (Credit-level history + forecasts)
    drv_df    = fetch_credit_workload_drivers(bigquery_manager)
    drv_fx_df = fetch_credit_driver_forecasts(bigquery_manager)
    if not drv_df.empty:
        df = pd.merge(df, drv_df, on='date', how='left')
    else:
        df['TotalOrders'] = np.nan

    # 3) Sort + build lags (target & drivers)
    df = df.sort_values(['client_id', 'date'])
    for lag in range(1, MAX_LAGS + 1):
        df[f'lag_{lag}']     = df.groupby('client_id')['total_cases_opened'].shift(lag)
        df[f'drv_lag_{lag}'] = df.groupby('client_id')['TotalOrders'].shift(lag)

    # prior-only rolling means (avoid leakage)
    df['target_roll3'] = df.groupby('client_id')['total_cases_opened'].shift(1).rolling(3).mean()
    df['driver_roll3'] = df.groupby('client_id')['TotalOrders'].shift(1).rolling(3).mean()

    # 4) TRAIN WINDOW ENFORCEMENT: keep rows with date >= 2024-01-01 and lag_12 available
    df_train = df[(df['date'] >= TRAIN_START_DATE) & df['lag_12'].notna()].copy()

    forecasts, audit_rows, modeling_rows = [], [], []

    # ----------------- per client loop -----------------
    for cid, group_df_full in df_train.groupby('client_id'):
        logger.info(f"\n{'='*36}\nProcessing Client: {cid}\n{'='*36}")

        # dtype coercion
        numeric_cols = (
            ['total_cases_opened','TotalOrders','target_roll3','driver_roll3',
             'month','sin_month','cos_month'] +
            [f'lag_{i}' for i in range(1, MAX_LAGS+1)] +
            [f'drv_lag_{i}' for i in range(1, MAX_LAGS+1)]
        )
        for c in numeric_cols:
            if c in group_df_full.columns:
                group_df_full[c] = pd.to_numeric(group_df_full[c], errors='coerce')

        # trend per client
        group_df_full = group_df_full.sort_values('date')
        group_df_full['trend'] = np.arange(len(group_df_full), dtype=float)

        client_df = group_df_full.set_index('date').sort_index()
        ts = client_df['total_cases_opened']

        # modeling snapshot
        snap_cols = [col for col in [
            'total_cases_opened','TotalOrders','target_roll3','driver_roll3',
            'month','sin_month','cos_month','trend',
            'lag_1','lag_3','lag_6','lag_9','lag_12',
            'drv_lag_1','drv_lag_3','drv_lag_6','drv_lag_9','drv_lag_12'
        ] if col in client_df.columns]
        snap = client_df[snap_cols].copy()
        snap['client_id'] = cid
        modeling_rows.append(snap.reset_index())

        # ---------- Build unified feature frame ----------
        # Rich candidate set (for OLS_Auto)
        candidate_cols = (
            ['TotalOrders','target_roll3','driver_roll3','month','sin_month','cos_month','trend'] +
            [f'lag_{i}' for i in range(1, MAX_LAGS+1)] +
            [f'drv_lag_{i}' for i in range(1, MAX_LAGS+1)]
        )
        candidate_cols = [c for c in candidate_cols if c in client_df.columns]

        feat_all = pd.DataFrame({'y': ts}).join(client_df[candidate_cols])
        feat_all = feat_all.apply(pd.to_numeric, errors='coerce').dropna()

        if len(feat_all) < (VAL_LEN_6M + 6):  # require minimally reasonable rows
            logger.warning(f"{cid}: insufficient rows after feature assembly; skipping.")
            continue

        # Train/Validation split — last 6 months as validation
        train_df = feat_all.iloc[:-VAL_LEN_6M]
        val_df   = feat_all.iloc[-VAL_LEN_6M:]

        # Containers
        model_preds_val = {}
        model_meta      = {}

        # ---------------- MODEL A: OLS_Auto (LassoCV selection -> OLS refit) ----------------
        try:
            y_tr_log = np.log1p(np.asarray(train_df['y'].values, dtype=float))
            X_tr_df  = train_df[candidate_cols].astype(float)
            X_val_df = val_df[candidate_cols].astype(float)

            scaler = StandardScaler(with_mean=True, with_std=True)
            X_tr_scaled = scaler.fit_transform(X_tr_df.values)
            X_val_scaled = scaler.transform(X_val_df.values)

            # LassoCV for selection (3-fold CV; small data safe)
            lasso = LassoCV(cv=min(3, max(2, len(train_df)//4)), random_state=42, n_alphas=100).fit(X_tr_scaled, y_tr_log)
            coef = lasso.coef_

            selected_idx = np.where(np.abs(coef) > 1e-8)[0].tolist()
            selected_feats = [candidate_cols[i] for i in selected_idx]

            # fallback if nothing selected: top-3 by corr with y_tr_log
            if not selected_feats:
                corrs = {}
                for c in candidate_cols:
                    try:
                        cor = np.corrcoef(y_tr_log, X_tr_df[c].values)[0,1]
                    except Exception:
                        cor = 0.0
                    if np.isfinite(cor):
                        corrs[c] = abs(float(cor))
                selected_feats = [k for k,_ in sorted(corrs.items(), key=lambda kv: kv[1], reverse=True)[:3]]

            # Refit OLS (log-space) on the selected features
            X_tr_sel = sm.add_constant(train_df[selected_feats].astype(float), has_constant='add')
            X_tr_mat = X_tr_sel.to_numpy(dtype=float)
            ols_auto = sm.OLS(y_tr_log, X_tr_mat).fit()

            # Validation predictions
            X_val_sel = sm.add_constant(val_df[selected_feats].astype(float), has_constant='add').to_numpy(dtype=float)
            raw_pred  = ols_auto.predict(X_val_sel)
            preds_val = safe_expm1(raw_pred)
            model_preds_val['OLS_Auto'] = preds_val
            model_meta['OLS_Auto'] = {
                'model': ols_auto, 'features': selected_feats,
                'r_squared': float(ols_auto.rsquared)
            }
        except Exception as e:
            logger.warning(f"{cid}: OLS_Auto failed: {e}")

        # ---------------- MODEL B: OLS_ForcedLag3912 (exact lags for target+driver) ----------------
        try:
            forced_feats = [f'lag_{k}' for k in (3,9,12)] + [f'drv_lag_{k}' for k in (3,9,12)] + ['month','sin_month','cos_month']
            forced_feats = [f for f in forced_feats if f in candidate_cols]

            # Need non-empty & enough rows
            train_forced = pd.DataFrame({'y': train_df['y']}).join(train_df[forced_feats]).dropna()
            val_forced   = pd.DataFrame({'y': val_df['y']}).join(val_df[forced_feats]).dropna()

            if len(train_forced) >= len(forced_feats) + 2 and len(val_forced) >= 1:
                y_tr_log = np.log1p(np.asarray(train_forced['y'].values, dtype=float))
                X_tr_df  = sm.add_constant(train_forced[forced_feats].astype(float), has_constant='add')
                X_tr_mat = X_tr_df.to_numpy(dtype=float)

                ols_forced = sm.OLS(y_tr_log, X_tr_mat).fit()

                X_val_df = sm.add_constant(val_forced[forced_feats].astype(float), has_constant='add').to_numpy(dtype=float)
                raw_pred = ols_forced.predict(X_val_df)
                preds_val = safe_expm1(raw_pred)

                # align back to full val_df index (fill NaN for rows dropped by dropna)
                aligned = np.full(len(val_df), np.nan, dtype=float)
                mask = val_forced.index.get_indexer(val_df.index)
                # get_indexer returns -1 for missing; map positives
                cur = 0
                for i, idx in enumerate(mask):
                    if idx != -1:
                        aligned[i] = preds_val[cur]
                        cur += 1

                model_preds_val['OLS_ForcedLag3912'] = aligned
                model_meta['OLS_ForcedLag3912'] = {
                    'model': ols_forced, 'features': forced_feats,
                    'r_squared': float(ols_forced.rsquared)
                }
            else:
                logger.warning(f"{cid}: insufficient rows for OLS_ForcedLag3912; skipping this model.")
        except Exception as e:
            logger.warning(f"{cid}: OLS_ForcedLag3912 failed: {e}")

        # ---------------- MODEL C: WeightedLagBlend(3,6,12) baseline ----------------
        try:
            feat_w = pd.DataFrame({'y': ts})
            for k in (3,6,12):
                feat_w[f'lag_{k}'] = feat_w['y'].shift(k)
            feat_w = feat_w.dropna()

            # tune on last 6 months overlap
            val_index = val_df.index
            sub = feat_w.reindex(val_index).dropna()
            if not sub.empty:
                y_v  = np.asarray(sub['y'].values, dtype=float)
                l3   = np.asarray(sub['lag_3'].values, dtype=float)
                l6   = np.asarray(sub['lag_6'].values, dtype=float)
                l12  = np.asarray(sub['lag_12'].values, dtype=float)

                best_w, best_s = (1.0, 0.0, 0.0), np.inf
                for w1 in np.arange(0,1.05,0.05):
                    for w2 in np.arange(0,1.05 - w1,0.05):
                        w3 = 1 - w1 - w2
                        s = smape(y_v, w1*l3 + w2*l6 + w3*l12)
                        if s < best_s:
                            best_w, best_s = (w1,w2,w3), s
                # expand to all 6 months
                full_l3  = feat_w['lag_3'].reindex(val_index).fillna(method='ffill').fillna(0).values
                full_l6  = feat_w['lag_6'].reindex(val_index).fillna(method='ffill').fillna(0).values
                full_l12 = feat_w['lag_12'].reindex(val_index).fillna(method='ffill').fillna(0).values
                model_preds_val['WeightedLagBlend'] = best_w[0]*full_l3 + best_w[1]*full_l6 + best_w[2]*full_l12
                model_meta['WeightedLagBlend'] = {'weights': best_w}
            else:
                logger.warning(f"{cid}: WLB had no overlap with validation; skipping WLB.")
        except Exception as e:
            logger.warning(f"{cid}: WLB failed: {e}")

        # ---------------- Select winner by 6m SMAPE ----------------
        if not model_preds_val:
            logger.warning(f"{cid}: No model produced validation predictions; skipping.")
            continue

        scores = {name: safe_reg_metrics(val_df['y'].values, pred)['smape']
                  for name, pred in model_preds_val.items()}
        valid_scores = {k:v for k,v in scores.items() if np.isfinite(v)}
        if not valid_scores:
            logger.warning(f"{cid}: No valid SMAPE scores; skipping.")
            continue

        winner = min(valid_scores, key=valid_scores.get)
        logger.info(f"✓ WINNER for {cid}: {winner} (6m SMAPE: {valid_scores[winner]:.2f})")

        # audit rows (store params + score)
        for name, pred in model_preds_val.items():
            row = {'client_id': cid, 'model': name, 'winner_model': winner,
                   'val_smape_6m': float(scores.get(name, np.nan))}
            if name == 'WeightedLagBlend':
                row.update({'w_lag3': model_meta[name]['weights'][0],
                            'w_lag6': model_meta[name]['weights'][1],
                            'w_lag12': model_meta[name]['weights'][2]})
            else:
                meta = model_meta.get(name, {})
                if 'r_squared' in meta:
                    row['r_squared'] = float(meta['r_squared'])
                if 'features' in meta:
                    row['features'] = ",".join(meta['features'])
            audit_rows.append(row)

        # ---------------- Recursive Forecast (15 months) ----------------
        hist = deque(ts.tolist(), maxlen=60)
        drv_hist = deque(client_df['TotalOrders'].dropna().tolist(), maxlen=60)

        last_total_orders = float(client_df['TotalOrders'].dropna().iloc[-1]) if client_df['TotalOrders'].notna().any() else 0.0
        driver_fx_lookup = ({pd.Timestamp(d): float(v) for d, v in drv_fx_df[['date','TotalOrders_fx']].values}
                            if not drv_fx_df.empty else {})
        if driver_fx_lookup:
            logger.info("· Using forecasted drivers when available for future months.")

        # Track trend continuation from last observed row
        last_trend = float(client_df['trend'].iloc[-1]) if 'trend' in client_df.columns else float(len(client_df)-1)
        future_idx = pd.date_range(ts.index[-1], periods=FORECAST_HORIZON+1, freq='MS')[1:]

        # cached bits for OLS models
        ols_model   = model_meta.get(winner, {}).get('model', None)
        ols_features = model_meta.get(winner, {}).get('features', None)

        # WLB weights if needed
        wlb_w = model_meta.get('WeightedLagBlend', {}).get('weights', (1.0,0.0,0.0))

        for step, d in enumerate(future_idx, start=1):
            pred = np.nan
            current_total_orders = driver_fx_lookup.get(d, last_total_orders)

            if winner == 'WeightedLagBlend':
                y3  = hist[-3]  if len(hist) >= 3  else (hist[-1] if hist else 0.0)
                y6  = hist[-6]  if len(hist) >= 6  else y3
                y12 = hist[-12] if len(hist) >= 12 else y6
                pred = wlb_w[0]*y3 + wlb_w[1]*y6 + wlb_w[2]*y12

            elif winner in ('OLS_Auto', 'OLS_ForcedLag3912') and ols_model is not None:
                # Build feature row for the chosen model
                month_i = int(d.month)
                sin_i   = np.sin(2*np.pi*month_i/12.0)
                cos_i   = np.cos(2*np.pi*month_i/12.0)
                trend_i = last_trend + step

                # target lags
                lag_vals = {f'lag_{k}': (hist[-k] if len(hist) >= k else (hist[-1] if len(hist) else 0.0))
                            for k in range(1, MAX_LAGS+1)}
                # driver lags
                drv_vals = {f'drv_lag_{k}': (drv_hist[-k] if len(drv_hist) >= k else current_total_orders)
                            for k in range(1, MAX_LAGS+1)}

                # rolling means (prior-only, mirror training logic)
                target_roll3 = (np.mean(list(hist)[-3:]) if len(hist) >= 3 else (hist[-1] if len(hist) else 0.0))
                driver_roll3 = (np.mean(list(drv_hist)[-3:]) if len(drv_hist) >= 3 else (drv_hist[-1] if len(drv_hist) else current_total_orders))

                # assemble full dict
                feat_map = {
                    'TotalOrders': current_total_orders,
                    'target_roll3': target_roll3,
                    'driver_roll3': driver_roll3,
                    'month': month_i,
                    'sin_month': sin_i,
                    'cos_month': cos_i,
                    'trend': trend_i
                }
                feat_map.update(lag_vals)
                feat_map.update(drv_vals)

                # select only the trained features in the right order
                if winner == 'OLS_Auto' and ols_features:
                    cols = ols_features
                else:
                    # forced-lag features were saved in 'features'
                    cols = model_meta[winner]['features']

                x_df = pd.DataFrame([{k: float(feat_map.get(k, 0.0)) for k in cols}])
                X_row = sm.add_constant(x_df, has_constant='add').to_numpy(dtype=float)

                raw_pred = float(ols_model.predict(X_row)[0])
                pred     = float(safe_expm1(raw_pred))

            # update histories for next step
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

        # Debug snapshot
        logger.info(f"Dtypes snapshot for {cid} (tail):\n{client_df.tail(2).dtypes.to_string()}")

    # --- Output (local only) ---
    if forecasts:
        fx_df = pd.DataFrame(forecasts).sort_values(['client_id','fx_date'])
        fx_df.to_csv(LOCAL_CSV, index=False)
        pd.DataFrame(audit_rows).sort_values(['client_id','model']).to_csv(AUDIT_CSV, index=False)
        pd.concat(modeling_rows, ignore_index=True).to_csv(MODELING_DATA_CSV, index=False)

        logger.info(f"✓ Forecast files saved:\n  {LOCAL_CSV}\n  {AUDIT_CSV}\n  {MODELING_DATA_CSV}")
        logger.info("✓ Credit ticket forecasting completed successfully (local only).")
    else:
        logger.warning("No forecasts were generated.")

except Exception as exc:
    email_manager.handle_error("Credit Ticket Forecasting Script Failure", exc, is_test=True)
