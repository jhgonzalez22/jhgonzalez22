"""
CX Credit - Call Volume Forecast - Production Hybrid (CWLB/WLB vs. XGB w/ GBQ FRED) v2.4

Adds a Chat path:
 - Phone (existing): target lags [1,3,6,12], econ lags [3,6,12]
 - Chat (new)     : target lags [1,3,6],    econ lags [3,6,12]

Both channels benchmark a constrained Weighted Lag Blend (WLB/CWLB) vs XGBoost and
produce 12-month forecasts. GBQ push remains commented for testing.

v2.x hardening:
 - Pearson numeric coercion + guards
 - Safe json serialization (NumPy -> Python native)
 - Fixed CWLB SMAPE logging
"""

# ------------------- standard imports -----------------------
import os
import sys
import warnings
import pytz
import json
from datetime import datetime
from collections import deque
import itertools

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
VAL_LEN_6M         = 6
STAMP              = datetime.now(pytz.timezone("America/Chicago"))
GRID_STEP          = 0.05
ECONOMIC_FEATURES  = ['UNRATE', 'HSN1F', 'FEDFUNDS', 'MORTGAGE30US']

# ------------------- helpers -------------------------
def to_native(obj):
    """Recursively convert NumPy scalars/bools to Python native types for json.dumps."""
    if isinstance(obj, dict):
        return {to_native(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
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
    # Ensure sufficient history
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

        # Per-lag constraints
        valid = True
        for lag, limits in constraints.items():
            if isinstance(lag, int) and lag in weights:
                if 'max' in limits and weights[lag] > limits['max'] + 1e-9:
                    valid = False; break
                if 'min' in limits and weights[lag] < limits['min'] - 1e-9:
                    valid = False; break
        if not valid:
            continue

        # Combined constraints
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

    # ADF on numeric-only series
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
        if col not in bu_data.columns:
            continue

        target = pd.to_numeric(bu_data[target_col], errors='coerce')
        predictor = pd.to_numeric(bu_data[col], errors='coerce')

        mask = target.notna() & predictor.notna()
        if mask.sum() < 3:
            continue

        x = target[mask].to_numpy(dtype=np.float64)
        y = predictor[mask].to_numpy(dtype=np.float64)

        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            continue
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            continue

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

# ------------------- MAIN EXECUTION -------------------------
try:
    logger.info("Starting Credit Forecasting (Phone + Chat) v2.4...")

    # 1) Fetch & Prepare Data (keep VolumeType in frame so we can split)
    source_df = bigquery_manager.run_gbq_sql(SQL_QUERY_PATH, return_dataframe=True)
    source_df = source_df.rename(columns={'Month': 'date'})
    source_df['date'] = pd.to_datetime(source_df['date'])
    source_df['TotalCallsOffered'] = pd.to_numeric(source_df['TotalCallsOffered'], errors='coerce')

    # Minimal columns we need (keep VolumeType to branch)
    df = source_df[['date', 'bu', 'VolumeType', 'TotalCallsOffered']].drop_duplicates()
    df = df.sort_values(['bu', 'date'])

    # 2) Integrate Economic Data from GBQ (monthly, with 3/6/12 lags)
    logger.info("Fetching economic data from BigQuery...")
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

    # 3) Feature engineering common to both channels
    df['month']     = df['date'].dt.month
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['trend']     = df.groupby(['VolumeType','bu']).cumcount() + 1

    # Build target lags by channel (Phone: 1/2/3/6/12; Chat: 1/3/6)
    def add_target_lags(frame, voltype, lags):
        m = frame['VolumeType'] == voltype
        for L in lags:
            frame.loc[m, f'lag_{L}'] = frame[m].groupby('bu')['TotalCallsOffered'].shift(L)

    add_target_lags(df, 'Phone', [1,2,3,6,12])
    add_target_lags(df, 'Chat',  [1,3,6])

    # Containers for outputs across both channels
    forecasts, audit_rows, all_modeling_data, stat_test_results, xgb_imp_rows = [], [], [], [], []

    # ------------------- per-channel runner -------------------
    def run_channel(channel_name: str, target_lags: list, use_cw_for_phone=True):
        """
        channel_name: 'Phone' or 'Chat'
        target_lags:  list of target lags to include in XGB & WLB search
        use_cw_for_phone: if True, keeps 'CWLB' naming for Phone; otherwise generic 'WLB'
        """
        logger.info(f"\n{'='*60}\nProcessing Channel: {channel_name}\n{'='*60}")
        ch_df = df[df['VolumeType'] == channel_name].copy()

        # Require at least lag max and econ lag 12 present for modeling
        required_cols = [f'lag_{max(target_lags)}']
        if lagged_econ_cols:
            required_cols += [f'{c}_lag_{l}' for c in ECONOMIC_FEATURES for l in [12] if f'{c}_lag_{l}' in ch_df.columns]
        ch_df.dropna(subset=[c for c in required_cols if c in ch_df.columns], inplace=True)

        for bu_name in ch_df['bu'].unique():
            logger.info(f"\n--- {channel_name} :: BU: {bu_name} ---")
            bu_df = ch_df[ch_df['bu'] == bu_name].set_index('date').sort_index().copy()
            bu_df['TotalCallsOffered'] = pd.to_numeric(bu_df['TotalCallsOffered'], errors='coerce')

            # Assemble XGB features: trend, seasonality, requested target lags, econ lags 3/6/12
            tlag_feats = [f'lag_{L}' for L in target_lags if f'lag_{L}' in bu_df.columns]
            econ_feats = [f'{c}_lag_{l}' for c in ECONOMIC_FEATURES for l in [3,6,12] if f'{c}_lag_{l}' in bu_df.columns]
            xgb_features = ['trend', 'sin_month', 'cos_month'] + tlag_feats + econ_feats

            # Stats
            stat_test_results.append(perform_statistical_tests(bu_df, f"{channel_name}:{bu_name}", xgb_features))

            # Train/val split
            if len(bu_df) <= VAL_LEN_6M:
                logger.warning(f"  Not enough data for 6m validation: {channel_name}:{bu_name}; skipping.")
                continue
            train_df, valid_df = bu_df.iloc[:-VAL_LEN_6M], bu_df.iloc[-VAL_LEN_6M:]
            y_val = valid_df['TotalCallsOffered']

            # WLB constraints
            # Phone keeps earlier CWLB settings; Chat uses WLB (no lag12) with similar idea:
            #   - cap lag1 to 0.5
            #   - enforce some minimum share on longer lags
            if channel_name == 'Phone' and use_cw_for_phone:
                lags_for_wlb = [1,3,6,12]
                constraints = {1:{'max':0.5}, 3:{'min':0.2}, 'combined':[{'lags':[3,6,12], 'min':0.5}]}
            else:
                lags_for_wlb = target_lags  # Chat: [1,3,6]
                constraints = {1:{'max':0.5}, 3:{'min':0.25}, 'combined':[{'lags':[3,6], 'min':0.5}]}

            bu_ts = bu_df['TotalCallsOffered'].asfreq('MS')
            wlb_weights, wlb_smape, _ = find_best_weights(
                bu_ts, lags_for_wlb, constraints, VAL_LEN_6M, GRID_STEP
            )
            wlb_smape_str = f"{wlb_smape:.2f}" if np.isfinite(wlb_smape) else "N/A"
            wlb_label = 'CWLB' if (channel_name == 'Phone' and use_cw_for_phone) else 'WLB'
            logger.info(f"  -> {wlb_label} evaluated: SMAPE={wlb_smape_str}")

            # XGB
            X_train = train_df[xgb_features]
            y_train = train_df['TotalCallsOffered']
            X_val   = valid_df[xgb_features]

            xgb = XGBRegressor(objective="reg:squarederror",
                               n_estimators=100,
                               learning_rate=0.05,
                               max_depth=3,
                               random_state=42)
            xgb.fit(X_train, np.log1p(y_train))
            xgb_preds = safe_expm1(xgb.predict(X_val))
            xgb_smape = smape(y_val, xgb_preds)
            logger.info(f"  -> XGBoost evaluated: SMAPE={xgb_smape:.2f}")

            # Feature importances
            imp = {str(feat): float(score) for feat, score in zip(xgb.feature_names_in_, xgb.feature_importances_)}
            xgb_imp_rows.append({'bu': f"{channel_name}:{bu_name}", 'feature_importance': json.dumps(to_native(imp))})

            # Winner selection + long-horizon weights for WLB winner
            if wlb_smape <= xgb_smape:
                winner, winner_smape = wlb_label, wlb_smape
                winner_details = {int(k): float(v) for k, v in (wlb_weights or {}).items()}
                if channel_name == 'Phone' and use_cw_for_phone:
                    long_lags = [3,6,12]
                    long_constraints = {3:{'min':0.4}, 6:{'min':0.2}, 12:{'min':0.1}}
                else:
                    long_lags = [3,6]
                    long_constraints = {3:{'min':0.4}, 6:{'min':0.2}}
                long_weights, _, _ = find_best_weights(bu_ts, long_lags, long_constraints, 12, GRID_STEP)
            else:
                winner, winner_smape = 'XGBoost', xgb_smape
                winner_details = {'n_estimators': 100, 'lr': 0.05, 'max_depth': 3}
                long_weights = None

            logger.info(f"✓ WINNER for {channel_name}:{bu_name}: {winner} (SMAPE: {winner_smape:.2f})")
            audit_rows.append({
                'channel': channel_name,
                'bu': bu_name,
                'winner': winner,
                'winner_smape': float(winner_smape) if np.isfinite(winner_smape) else None,
                f'{wlb_label.lower()}_smape': float(wlb_smape) if np.isfinite(wlb_smape) else None,
                'xgb_smape': float(xgb_smape) if np.isfinite(xgb_smape) else None,
                'details': json.dumps(to_native(winner_details))
            })

            # 5) Forecast generation
            if np.isfinite(winner_smape):
                logger.info(f"Generating {FORECAST_HORIZON}-month forecast with {winner}...")
                hist = deque(bu_ts.tolist(), maxlen=24)
                future_idx = pd.date_range(bu_ts.index[-1], periods=FORECAST_HORIZON + 1, freq='MS')[1:]

                for i, d in enumerate(future_idx, 1):
                    pred = np.nan
                    if winner in ('WLB', 'CWLB') and wlb_weights:
                        # early steps may use the short-horizon weights; later, long_weights if available
                        if long_weights is None:
                            # fallback: keep short weights
                            # map requested lags to available history safely
                            lags_to_use = {}
                            for L in sorted(set(lags_for_wlb)):
                                if len(hist) >= L:
                                    lags_to_use[L] = hist[-L]
                            pred = sum((wlb_weights.get(L, 0.0) * lags_to_use[L]) for L in lags_to_use)
                        else:
                            if i in [1,2] and 1 in lags_for_wlb:
                                # allow lag1 only in very first step if available
                                lags_to_use = {}
                                for L in sorted(set(lags_for_wlb)):
                                    if len(hist) >= L:
                                        lags_to_use[L] = hist[-L]
                                use_w = wlb_weights
                            else:
                                # switch to long-horizon set
                                lags_to_use = {}
                                for L in sorted(set(long_weights.keys())):
                                    if len(hist) >= L:
                                        lags_to_use[L] = hist[-L]
                                use_w = long_weights
                            pred = sum((use_w.get(L, 0.0) * lags_to_use[L]) for L in lags_to_use)

                    elif winner == 'XGBoost':
                        # Single-row feature vector
                        row = {
                            'trend': (train_df['trend'].max() or 0) + len(valid_df) + i,
                            'sin_month': np.sin(2*np.pi*d.month/12),
                            'cos_month': np.cos(2*np.pi*d.month/12),
                        }
                        # target lags
                        for L in target_lags:
                            row[f'lag_{L}'] = hist[-L] if len(hist) >= L else hist[-1]
                        # econ lags (reuse last known lags)
                        for c in ECONOMIC_FEATURES:
                            for l in [3,6,12]:
                                key = f'{c}_lag_{l}'
                                if key in xgb_features:
                                    # Use last known level for that lag; if not available, NaN (XGB will still output)
                                    if key in bu_df.columns and not bu_df[key].dropna().empty:
                                        row[key] = bu_df[key].dropna().iloc[-1]
                                    else:
                                        row[key] = np.nan

                        x_row_df = pd.DataFrame([row])[xgb_features]
                        pred = safe_expm1(xgb.predict(x_row_df)[0])
                    else:
                        pred = hist[-1]

                    hist.append(pred)
                    forecasts.append({
                        'fx_date': d,
                        'client_id': bu_name,
                        'vol_type': f"{channel_name.lower()}_{winner.lower()}",
                        'fx_vol': safe_round(pred),
                        'fx_id': f"{winner.lower()}_{channel_name.lower()}_{STAMP:%Y%m%d}",
                        'fx_status': "A",
                        'load_ts': STAMP
                    })

    # ------------------- run both channels -------------------
    run_channel(channel_name='Phone', target_lags=[1,3,6,12], use_cw_for_phone=True)
    run_channel(channel_name='Chat',  target_lags=[1,3,6],     use_cw_for_phone=False)

    # 6) Save All Local Deliverables and (optionally) Push to GBQ
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

    if forecasts:
        fx_df = pd.DataFrame(forecasts).sort_values(['client_id', 'fx_date'])
        fx_df.to_csv(FORECAST_RESULTS_CSV, index=False)
        logger.info(f"✓ Forecast data saved locally to {FORECAST_RESULTS_CSV}")

        logger.warning("GBQ PUSH IS DISABLED FOR TESTING. No data was written to the database.")
        # -------------------------------------------------------------------------
        # --- GBQ PUSH BLOCK ---
        # --- THIS SECTION IS DISABLED FOR TESTING. UNCOMMENT TO ENABLE MONTHLY PUSH.
        # -------------------------------------------------------------------------
        # logger.info(f"Pushing {len(fx_df):,} forecast rows to {DEST_TABLE}...")
        # bigquery_manager.import_data_to_bigquery(fx_df, DEST_TABLE, gbq_insert_action="append", auto_convert_df=True)
        #
        # client_list_str = ", ".join([f"'{c}'" for c in fx_df['client_id'].unique()])
        # dedup_sql = f"""
        # UPDATE `{DEST_TABLE}` t
        # SET t.fx_status = 'I'
        # WHERE t.fx_status = 'A' AND t.vol_type IN ('phone_wlb','phone_cwlb','phone_xgboost',
        #                                            'chat_wlb','chat_xgboost')
        #   AND t.client_id IN ({client_list_str})
        #   AND EXISTS (
        #     SELECT 1 FROM `{DEST_TABLE}` s
        #     WHERE s.client_id = t.client_id
        #       AND s.vol_type = t.vol_type
        #       AND s.fx_date = t.fx_date
        #       AND s.load_ts > t.load_ts
        #   )
        # """
        # bigquery_manager.run_gbq_sql(dedup_sql, return_dataframe=False)
        # logger.info("✓ Older forecasts deactivated in GBQ for overlapping rows.")
        #
        # bigquery_manager.update_log_in_bigquery()
        # logger.info("✓ Forecasting complete and results pushed to BigQuery.")
        # -------------------------------------------------------------------------

    else:
        logger.warning("No forecasts were generated.")

except Exception as exc:
    email_manager.handle_error("Credit Forecast Script Failure", exc, is_test=True)
