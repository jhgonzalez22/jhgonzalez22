"""
CX Credit - Call Volume Forecast - Production Hybrid v2.7

- vol_type strictly 'phone' or 'chat'
- PHONE: CWLB vs XGB vs Native3mGR; winner by 6m SMAPE; recursive forecast
- CHAT : WLB with lags {1,3} only; weights learned on LAST 3 MONTHS (3m SMAPE);
         recursive forecast using those fixed weights

Hardening:
 - Pearson numeric coercion + guards
 - Safe JSON serialization (NumPy -> Python native)
 - Safe logging for SMAPE values
 - GBQ push enabled with dedup of older rows
"""

# ------------------- standard imports -----------------------
import os, sys, warnings, pytz, json, itertools
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ------------------- helper-package path --------------------
sys.path.append(r"C:\WFM_Scripting\Automation")
from scripthelper import Config, Logger, BigQueryManager, EmailManager

# ------------------- initialise scripthelper ---------------
config           = Config(rpt_id=299)
logger           = Logger(config)
email_manager    = EmailManager(config)
bigquery_manager = BigQueryManager(config)

# ------------------- file / table paths & queries --------------------
SQL_QUERY_PATH = (r"C:\WFM_Scripting\Forecasting"
                  r"\GBQ - Credit Historicals Phone - Chat Volumes.sql")

FRED_GBQ_QUERY = """
    SELECT DATE_TRUNC(Date, MONTH) as date,
           AVG(UNRATE) as UNRATE,
           AVG(HSN1F) as HSN1F,
           AVG(FEDFUNDS) as FEDFUNDS,
           AVG(MORTGAGE30US) as MORTGAGE30US
    FROM tax_clnt_svcs.fred 
    WHERE DATE_TRUNC(Date, MONTH) >= '2022-01-01' 
      AND DATE_TRUNC(Date, MONTH) <= DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 DAY)
    GROUP BY 1 
    ORDER BY 1
"""

DEST_TABLE           = "tax_clnt_svcs.cx_nontax_platforms_forecast"
FORECAST_RESULTS_CSV = r"C:\WFM_Scripting\forecast_results_credit.csv"
MODEL_EVAL_CSV       = r"C:\WFM_Scripting\model_eval_credit.csv"
MODELING_DATA_CSV    = r"C:\WFM_Scripting\modeling_data_credit.csv"
STATS_CSV            = r"C:\WFM_Scripting\statistical_tests_credit.csv"
XGB_VIZ_CSV          = r"C:\WFM_Scripting\xgb_feature_importance_credit.csv"
MODEL_SUMMARIES_FILE = r"C:\WFM_Scripting\model_summaries_credit.txt"

# ------------------- forecast parameters --------------------
FORECAST_HORIZON   = 12
VAL_LEN_PHONE_6M   = 6
VAL_LEN_CHAT_3M    = 3
STAMP              = datetime.now(pytz.timezone("America/Chicago"))
GRID_STEP          = 0.05
ECONOMIC_FEATURES  = ['UNRATE', 'HSN1F', 'FEDFUNDS', 'MORTGAGE30US']

# ------------------- helpers -------------------------
def to_native(obj):
    if isinstance(obj, dict):   return {to_native(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_native(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)): return obj.item()
    if isinstance(obj, (np.bool_,)): return bool(obj)
    return obj

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

def find_best_weights(ts: pd.Series, lags: list, constraints: dict, val_window_len: int, step: float):
    """Grid-search convex weights on specified lags over the last val_window_len months."""
    val_window_len = min(val_window_len, max(3, len(ts) // 2))
    if len(ts) < val_window_len + max(lags):
        return None, np.inf, None

    validation_data = ts.iloc[-val_window_len:]
    best_weights, best_smape, best_preds = None, np.inf, None

    num_lags = len(lags)
    weight_iters = [np.arange(0, 1.0 + step, step) for _ in range(num_lags - 1)]
    for W in itertools.product(*weight_iters):
        if sum(W) > 1.0 + 1e-9:
            continue
        all_w = list(W) + [1.0 - sum(W)]
        weights = {int(l): float(w) for l, w in zip(lags, all_w)}

        # Per-lag constraints (optional)
        valid = True
        for lag, limits in constraints.items():
            if isinstance(lag, int) and lag in weights:
                if 'max' in limits and weights[lag] > limits['max'] + 1e-9:
                    valid = False; break
                if 'min' in limits and weights[lag] < limits['min'] - 1e-9:
                    valid = False; break
        if not valid:
            continue

        # Combined constraints (optional)
        if 'combined' in constraints:
            for combo in constraints['combined']:
                combo_sum = sum(weights.get(l, 0.0) for l in combo['lags'])
                if 'min' in combo and combo_sum < combo['min'] - 1e-9:
                    valid = False; break
            if not valid:
                continue

        forecast_vals = 0.0
        for lag, w in weights.items():
            forecast_vals += ts.shift(lag).reindex(validation_data.index) * w

        mask = forecast_vals.notna() & validation_data.notna()
        if not mask.any():
            continue

        current_smape = smape(validation_data[mask], forecast_vals[mask])
        if current_smape < best_smape - 0.1:
            best_smape, best_weights, best_preds = current_smape, weights, forecast_vals

    return best_weights, best_smape, best_preds

def perform_statistical_tests(bu_data: pd.DataFrame, bu_name: str, features: list):
    results = {'bu': bu_name}
    target_col = 'TotalCallsOffered'
    adf_series = pd.to_numeric(bu_data[target_col], errors='coerce').dropna()
    if len(adf_series) > 12:
        try:
            adf_p = adfuller(adf_series)[1]
            results['adf_p_value'] = float(adf_p)
            results['is_stationary'] = bool(adf_p < 0.05)
        except Exception:
            results['adf_p_value'] = None
            results['is_stationary'] = None
    correlations = {}
    for col in features:
        if col not in bu_data.columns: continue
        target = pd.to_numeric(bu_data[target_col], errors='coerce')
        predictor = pd.to_numeric(bu_data[col], errors='coerce')
        mask = target.notna() & predictor.notna()
        if mask.sum() < 3: continue
        x = target[mask].to_numpy(dtype=np.float64)
        y = predictor[mask].to_numpy(dtype=np.float64)
        if not (np.isfinite(x).all() and np.isfinite(y).all()): continue
        if np.std(x) == 0.0 or np.std(y) == 0.0: continue
        try:
            corr, p_val = pearsonr(x, y)
            correlations[col] = {'correlation': float(corr), 'p_value': float(p_val)}
        except Exception:
            continue
    results['correlations'] = json.dumps(to_native(correlations))
    logger.info(f"  --- Statistical Tests for: {bu_name} ---")
    if 'adf_p_value' in results and results['adf_p_value'] is not None:
        logger.info(f"  ADF P-Value: {results['adf_p_value']:.4f} (Stationary: {results['is_stationary']})")
    return results

# SeasonalNaive3mGR (native3mg) helpers for PHONE
def compute_3m_yoy_factor(series: pd.Series) -> float:
    s = series.dropna()
    ratios = []
    for k in [1, 2, 3]:
        if len(s) > 12 + k:
            num = s.iloc[-k]
            den = s.iloc[-12 - k]
            if den and np.isfinite(num) and np.isfinite(den) and den != 0:
                ratios.append(num / den)
    return float(np.mean(ratios)) if ratios else 1.0

def seasonal_naive_3m_gr_predict(train_series: pd.Series, pred_index: pd.DatetimeIndex) -> pd.Series:
    factor = compute_3m_yoy_factor(train_series)
    preds = []
    s = train_series.copy()
    for d in pred_index:
        t12 = s.get(d - pd.DateOffset(months=12), np.nan)
        preds.append((t12 if np.isfinite(t12) else np.nan) * factor if np.isfinite(factor) else np.nan)
    return pd.Series(preds, index=pred_index, dtype='float')

# ------------------- MAIN EXECUTION -------------------------
try:
    logger.info("Starting Credit Forecasting v2.7 (Phone + Chat)...")

    # 1) Fetch & Prepare Data
    source_df = bigquery_manager.run_gbq_sql(SQL_QUERY_PATH, return_dataframe=True)
    source_df = source_df.rename(columns={'Month': 'date'})
    source_df['date'] = pd.to_datetime(source_df['date'])
    source_df['TotalCallsOffered'] = pd.to_numeric(source_df['TotalCallsOffered'], errors='coerce')
    df = source_df[['date', 'bu', 'VolumeType', 'TotalCallsOffered']].drop_duplicates().sort_values(['bu', 'date'])

    # 2) Economic Data (used by Phone/XGB)
    fred_df = bigquery_manager.run_gbq_sql(FRED_GBQ_QUERY, return_dataframe=True)
    lagged_econ_cols = []
    if not fred_df.empty:
        fred_df['date'] = pd.to_datetime(fred_df['date'])
        for col in ECONOMIC_FEATURES:
            fred_df[col] = pd.to_numeric(fred_df[col], errors='coerce')
        fred_df = fred_df.sort_values('date').reset_index(drop=True)
        for col in ECONOMIC_FEATURES:
            for lag in [3, 6, 12]:
                fred_df[f'{col}_lag_{lag}'] = fred_df[col].shift(lag)
        lagged_econ_cols = [f'{col}_lag_{l}' for col in ECONOMIC_FEATURES for l in [3,6,12]]
        df = pd.merge(df, fred_df[['date'] + ECONOMIC_FEATURES + lagged_econ_cols], on='date', how='left')
        for c in ECONOMIC_FEATURES + lagged_econ_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 3) Feature engineering common
    df['month']     = df['date'].dt.month
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['trend']     = df.groupby(['VolumeType','bu']).cumcount() + 1

    # Target lags
    def add_target_lags(frame, voltype, lags):
        m = frame['VolumeType'] == voltype
        for L in lags:
            frame.loc[m, f'lag_{L}'] = frame[m].groupby('bu')['TotalCallsOffered'].shift(L)

    add_target_lags(df, 'Phone', [1,2,3,6,12])
    add_target_lags(df, 'Chat',  [1,3])  # Chat uses only lag_1 and lag_3

    # Outputs
    forecasts, audit_rows, all_modeling_data, stat_test_results, xgb_imp_rows = [], [], [], [], []

    # -------- PHONE: CWLB vs XGB vs Native3mGR ----------
    logger.info("\n" + "="*60 + "\nProcessing Channel: Phone\n" + "="*60)
    phone_df = df[df['VolumeType'] == 'Phone'].copy()
    req_cols = ['lag_12']
    if lagged_econ_cols:
        req_cols += [c for c in lagged_econ_cols if c.endswith('_lag_12')]
    phone_df.dropna(subset=[c for c in req_cols if c in phone_df.columns], inplace=True)

    for bu_name in phone_df['bu'].unique():
        logger.info(f"\n--- Phone :: BU: {bu_name} ---")
        bu = phone_df[phone_df['bu'] == bu_name].set_index('date').sort_index().copy()
        bu['TotalCallsOffered'] = pd.to_numeric(bu['TotalCallsOffered'], errors='coerce')
        bu_ts = bu['TotalCallsOffered'].asfreq('MS')
        all_modeling_data.append(bu.copy())

        # XGB features (Phone)
        xgb_feats = ['trend','sin_month','cos_month','lag_3','lag_6','lag_12'] + \
                    [c for c in [f'{x}_lag_{l}' for x in ECONOMIC_FEATURES for l in [3,6,12]] if c in bu.columns]

        stat_test_results.append(perform_statistical_tests(bu, f"Phone:{bu_name}", xgb_feats))
        if len(bu) <= VAL_LEN_PHONE_6M:
            logger.warning(f"  Not enough data for 6m validation: Phone:{bu_name}; skipping.")
            continue
        train, valid = bu.iloc[:-VAL_LEN_PHONE_6M], bu.iloc[-VAL_LEN_PHONE_6M:]
        y_val = valid['TotalCallsOffered']

        # CWLB
        cwlb_weights, cwlb_smape, _ = find_best_weights(
            bu_ts, [1,3,6,12], {1:{'max':0.5}, 3:{'min':0.2}, 'combined':[{'lags':[3,6,12], 'min':0.5}]},
            VAL_LEN_PHONE_6M, GRID_STEP
        )
        logger.info(f"  -> CWLB SMAPE={f'{cwlb_smape:.2f}' if np.isfinite(cwlb_smape) else 'N/A'}")

        # XGB
        Xtr, Xv = train[xgb_feats], valid[xgb_feats]
        ytr = train['TotalCallsOffered']
        xgb = XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        xgb.fit(Xtr, np.log1p(ytr))
        xgb_preds = safe_expm1(xgb.predict(Xv))
        xgb_smape = smape(y_val, xgb_preds)
        logger.info(f"  -> XGB  SMAPE={xgb_smape:.2f}")
        imp = {str(f): float(s) for f, s in zip(xgb.feature_names_in_, xgb.feature_importances_)}
        xgb_imp_rows.append({'bu': f"Phone:{bu_name}", 'feature_importance': json.dumps(to_native(imp))})

        # Native3mGR
        native_val_preds = seasonal_naive_3m_gr_predict(train['TotalCallsOffered'], valid.index)
        native_smape = smape(y_val, native_val_preds)
        logger.info(f"  -> Native3mGR SMAPE={native_smape:.2f}")

        # Winner
        smapes = {'CWLB': cwlb_smape, 'XGBoost': xgb_smape, 'Native3mGR': native_smape}
        winner = min(smapes, key=lambda k: smapes[k])
        winner_smape = smapes[winner]
        logger.info(f"✓ WINNER Phone:{bu_name}: {winner} (SMAPE {winner_smape:.2f})")
        audit_rows.append({
            'channel': 'Phone', 'bu': bu_name, 'winner': winner,
            'winner_smape': float(winner_smape) if np.isfinite(winner_smape) else None,
            'cwlb_smape': float(cwlb_smape) if np.isfinite(cwlb_smape) else None,
            'xgb_smape': float(xgb_smape) if np.isfinite(xgb_smape) else None,
            'native3mgr_smape': float(native_smape) if np.isfinite(native_smape) else None,
            'details': json.dumps(to_native({'cwlb_weights': cwlb_weights}))
        })

        # Forecast with the winning model
        if np.isfinite(winner_smape):
            hist = deque(bu_ts.tolist(), maxlen=24)
            future_idx = pd.date_range(bu_ts.index[-1], periods=FORECAST_HORIZON + 1, freq='MS')[1:]

            long_weights = None
            if winner == 'CWLB':
                long_weights, _, _ = find_best_weights(bu_ts, [3,6,12], {3:{'min':0.4}, 6:{'min':0.2}, 12:{'min':0.1}}, 12, GRID_STEP)
            native_factor = compute_3m_yoy_factor(train['TotalCallsOffered']) if winner == 'Native3mGR' else None

            for i, d in enumerate(future_idx, 1):
                if winner == 'CWLB' and cwlb_weights:
                    if long_weights is None:
                        Ls = [1,3,6,12]
                        lags_to_use = {L: hist[-L] for L in Ls if len(hist) >= L}
                        pred = sum(cwlb_weights.get(L, 0.0) * lags_to_use[L] for L in lags_to_use)
                    else:
                        if i == 1:
                            w = cwlb_weights; Ls = [1,3,6,12]
                        elif i in [2,3]:
                            w = cwlb_weights; Ls = [3,6,12]
                        else:
                            w = long_weights;  Ls = sorted(long_weights.keys())
                        lags_to_use = {L: hist[-L] for L in Ls if len(hist) >= L}
                        pred = sum(w.get(L, 0.0) * lags_to_use[L] for L in lags_to_use)
                elif winner == 'XGBoost':
                    row = {'trend': (train['trend'].max() or 0) + len(valid) + i,
                           'sin_month': np.sin(2*np.pi*d.month/12), 'cos_month': np.cos(2*np.pi*d.month/12),
                           'lag_3': hist[-3] if len(hist) >= 3 else hist[-1],
                           'lag_6': hist[-6] if len(hist) >= 6 else hist[-1],
                           'lag_12': hist[-12] if len(hist) >= 12 else hist[-1]}
                    for c in ECONOMIC_FEATURES:
                        for l in [3,6,12]:
                            key = f'{c}_lag_{l}'
                            if key in xgb_feats:
                                row[key] = bu[key].dropna().iloc[-1] if key in bu.columns and not bu[key].dropna().empty else np.nan
                    pred = safe_expm1(xgb.predict(pd.DataFrame([row])[xgb_feats])[0])
                else:  # Native3mGR
                    t12 = hist[-12] if len(hist) >= 12 else hist[-1]
                    factor = native_factor if (native_factor is not None and np.isfinite(native_factor)) else 1.0
                    pred = (t12 or 0.0) * factor

                hist.append(pred)
                forecasts.append({
                    'fx_date': d, 'client_id': bu_name, 'vol_type': 'phone',
                    'fx_vol': safe_round(pred), 'fx_id': f"{winner.lower()}_phone_{STAMP:%Y%m%d}",
                    'fx_status': "A", 'load_ts': STAMP
                })

    # -------- CHAT: WLB (lags {1,3}) ONLY; learn weights on 3m; recursive forecast ----------
    logger.info("\n" + "="*60 + "\nProcessing Channel: Chat (WLB lag1,lag3; 3m SMAPE)\n" + "="*60)
    chat_df = df[df['VolumeType'] == 'Chat'].copy()

    for bu_name in chat_df['bu'].unique():
        logger.info(f"\n--- Chat :: BU: {bu_name} ---")
        bu = chat_df[chat_df['bu'] == bu_name].set_index('date').sort_index().copy()
        bu['TotalCallsOffered'] = pd.to_numeric(bu['TotalCallsOffered'], errors='coerce')
        bu_ts = bu['TotalCallsOffered'].asfreq('MS')
        all_modeling_data.append(bu.copy())

        # Simple stats
        stat_test_results.append(perform_statistical_tests(bu, f"Chat:{bu_name}", [c for c in ['lag_1','lag_3'] if c in bu.columns]))

        if len(bu) <= VAL_LEN_CHAT_3M:
            logger.warning(f"  Not enough data for 3m validation: Chat:{bu_name}; skipping.")
            continue

        # Learn dynamic weights on {1,3} via 3-month validation SMAPE
        train, valid = bu.iloc[:-VAL_LEN_CHAT_3M], bu.iloc[-VAL_LEN_CHAT_3M:]
        wlb_weights, wlb_smape, _ = find_best_weights(
            bu_ts, [1,3], constraints={}, val_window_len=VAL_LEN_CHAT_3M, step=GRID_STEP
        )
        logger.info(f"  -> Chat WLB(1,3) 3m SMAPE={f'{wlb_smape:.2f}' if np.isfinite(wlb_smape) else 'N/A'} weights={wlb_weights}")

        audit_rows.append({
            'channel': 'Chat', 'bu': bu_name, 'winner': 'WLB(1,3)-3m',
            'winner_smape': float(wlb_smape) if np.isfinite(wlb_smape) else None,
            'details': json.dumps(to_native({'weights': wlb_weights}))
        })

        # Recursive forecast with fixed learned weights
        if wlb_weights:
            hist = deque(bu_ts.tolist(), maxlen=24)
            future_idx = pd.date_range(bu_ts.index[-1], periods=FORECAST_HORIZON + 1, freq='MS')[1:]
            for i, d in enumerate(future_idx, 1):
                l1 = hist[-1] if len(hist) >= 1 else np.nan
                l3 = hist[-3] if len(hist) >= 3 else (hist[-1] if len(hist) else np.nan)
                parts = []
                if np.isfinite(l1): parts.append(wlb_weights.get(1, 0.0) * l1)
                if np.isfinite(l3): parts.append(wlb_weights.get(3, 0.0) * l3)
                pred = float(np.sum(parts)) if parts else (hist[-1] if len(hist) else 0.0)
                hist.append(pred)
                forecasts.append({
                    'fx_date': d, 'client_id': bu_name, 'vol_type': 'chat',
                    'fx_vol': safe_round(pred), 'fx_id': f"wlb13_3m_chat_{STAMP:%Y%m%d}",
                    'fx_status': "A", 'load_ts': STAMP
                })
        else:
            logger.warning(f"  No valid weights learned for Chat:{bu_name}; skipping forecast.")

    # 6) Save outputs (local)
    if audit_rows:
        pd.DataFrame(audit_rows).to_csv(MODEL_EVAL_CSV, index=False)
        logger.info(f"✓ Model evaluation results saved to {MODEL_EVAL_CSV}")

    if stat_test_results:
        pd.DataFrame(stat_test_results).to_csv(STATS_CSV, index=False)
        logger.info(f"✓ Statistical tests saved to {STATS_CSV}")

    if xgb_imp_rows:
        pd.DataFrame(xgb_imp_rows).to_csv(XGB_VIZ_CSV, index=False)
        logger.info(f"✓ XGBoost feature importances saved to {XGB_VIZ_CSV}")

    if all_modeling_data:
        pd.concat(all_modeling_data).to_csv(MODELING_DATA_CSV, index=True)
        logger.info(f"✓ Combined modeling data saved to {MODELING_DATA_CSV}")

    with open(MODEL_SUMMARIES_FILE, 'w') as f:
        f.write("No statsmodels summaries generated in this script version.\n")

    # 7) Write forecasts locally and PUSH to GBQ (enabled)
    if forecasts:
        fx_df = pd.DataFrame(forecasts).sort_values(['client_id', 'fx_date'])
        fx_df.to_csv(FORECAST_RESULTS_CSV, index=False)
        logger.info(f"✓ Forecast data saved locally to {FORECAST_RESULTS_CSV}")

        # --- GBQ PUSH ---
        logger.info(f"Pushing {len(fx_df):,} forecast rows to {DEST_TABLE}...")
        bigquery_manager.import_data_to_bigquery(
            fx_df, DEST_TABLE, gbq_insert_action="append", auto_convert_df=True
        )

        # Deactivate older rows where a newer load_ts exists for same (client_id, vol_type, fx_date)
        client_list = fx_df['client_id'].dropna().unique().tolist()
        if client_list:
            client_list_str = ", ".join([f"'{c}'" for c in client_list])
            dedup_sql = f"""
            UPDATE `{DEST_TABLE}` t
            SET t.fx_status = 'I'
            WHERE t.fx_status = 'A'
              AND t.client_id IN ({client_list_str})
              AND EXISTS (
                SELECT 1 FROM `{DEST_TABLE}` s
                WHERE s.client_id = t.client_id
                  AND s.vol_type = t.vol_type
                  AND s.fx_date = t.fx_date
                  AND s.load_ts > t.load_ts
              )
            """
            bigquery_manager.run_gbq_sql(dedup_sql, return_dataframe=False)
            bigquery_manager.update_log_in_bigquery()
            logger.info("✓ Forecasts pushed to GBQ and older active rows deactivated.")
        else:
            logger.warning("Client list empty after forecast build; skipped dedup SQL.")

    else:
        logger.warning("No forecasts were generated; nothing to push.")

except Exception as exc:
    email_manager.handle_error("Credit Forecast Script Failure", exc, is_test=True)
