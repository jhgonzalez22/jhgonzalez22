"""
CX Credit - Call Volume Forecast - Production Hybrid (CWLB vs. XGB w/ GBQ FRED) v2.3

This script forecasts monthly phone call volumes by benchmarking a Constrained
Weighted Lag Blend (CWLB) model against an XGBoost model with economic features.
It is designed to generate a new forecast and push it to BigQuery every month it runs.

v2.3 Fix: Corrected a data type issue in the perform_statistical_tests function
by explicitly coercing data to numeric before correlation calculation.
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

# ------------------- utilities -------------------------
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
        weights = dict(zip(lags, all_w))

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
                combo_sum = sum(weights.get(l, 0) for l in combo['lags'])
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

    results['correlations'] = json.dumps(correlations)
    logger.info(f"  --- Statistical Tests for: {bu_name} ---")
    if 'adf_p_value' in results and results['adf_p_value'] is not None:
        logger.info(f"  ADF P-Value: {results['adf_p_value']:.4f} (Stationary: {results['is_stationary']})")
    return results

# ------------------- MAIN EXECUTION -------------------------
try:
    logger.info("Starting Credit Phone Forecasting Script (v2.3 Monthly Push Test)...")

    # 1) Fetch & Prepare Data
    source_df = bigquery_manager.run_gbq_sql(SQL_QUERY_PATH, return_dataframe=True)
    source_df = source_df.rename(columns={'Month': 'date'})
    source_df['date'] = pd.to_datetime(source_df['date'])

    # Filter to phone and minimal columns
    df = source_df[source_df['VolumeType'] == 'Phone'].sort_values(['bu', 'date']).copy()
    df = df[['date', 'bu', 'TotalCallsOffered']].drop_duplicates()

    # Normalize numeric early
    df['TotalCallsOffered'] = pd.to_numeric(df['TotalCallsOffered'], errors='coerce')

    # 2) Integrate Economic Data from GBQ
    logger.info("Fetching economic data from BigQuery...")
    fred_df = bigquery_manager.run_gbq_sql(FRED_GBQ_QUERY, return_dataframe=True)
    if not fred_df.empty:
        fred_df['date'] = pd.to_datetime(fred_df['date'])
        # Ensure FRED numerics are numeric
        for col in ECONOMIC_FEATURES:
            fred_df[col] = pd.to_numeric(fred_df[col], errors='coerce')

        # Build econ lags on the monthly series
        fred_df = fred_df.sort_values('date').reset_index(drop=True)
        for col in ECONOMIC_FEATURES:
            for lag in [3, 6, 12]:
                fred_df[f'{col}_lag_{lag}'] = fred_df[col].shift(lag)

        # Merge on date
        df = pd.merge(df, fred_df, on='date', how='left')

        # Coerce lag columns to numeric and require enough history
        lagged_econ_cols = [f'{col}_lag_12' for col in ECONOMIC_FEATURES]
        for c in lagged_econ_cols + ECONOMIC_FEATURES:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

    else:
        lagged_econ_cols = []

    # 3) Engineer Other Features
    df['month'] = df['date'].dt.month
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['trend'] = df.groupby('bu').cumcount() + 1

    for lag in [1, 2, 3, 6, 12]:
        df[f'lag_{lag}'] = df.groupby('bu')['TotalCallsOffered'].shift(lag)

    # Require core lags + econ lags (if present)
    required_cols = ['lag_12']
    if lagged_econ_cols:
        required_cols += lagged_econ_cols
    df.dropna(subset=required_cols, inplace=True)

    forecasts, audit_rows, all_modeling_data, stat_test_results, xgb_imp_rows = [], [], [], [], []

    # 4) Main loop for per-BU bake-off
    for bu_name in df['bu'].unique():
        logger.info(f"\n{'='*40}\nProcessing BU: {bu_name}\n{'='*40}")
        bu_df = df[df['bu'] == bu_name].set_index('date').sort_index()

        # Keep numeric hygiene for modeling columns
        bu_df['TotalCallsOffered'] = pd.to_numeric(bu_df['TotalCallsOffered'], errors='coerce')

        bu_ts = bu_df['TotalCallsOffered'].asfreq('MS')
        all_modeling_data.append(bu_df.copy())

        # XGB features: seasonality, trend, target lags, econ lags (3/6/12)
        xgb_features = (['trend', 'sin_month', 'cos_month', 'lag_3', 'lag_6', 'lag_12'] +
                        [f'{c}_lag_{l}' for c in ECONOMIC_FEATURES for l in [3, 6, 12] if f'{c}_lag_{l}' in bu_df.columns])

        # Statistical tests (robust to types)
        stat_test_results.append(perform_statistical_tests(bu_df, bu_name, xgb_features))

        # Train/validation split
        if len(bu_df) <= VAL_LEN_6M:
            logger.warning(f"  Not enough data for 6-month validation in {bu_name}; skipping.")
            continue
        train_df, valid_df = bu_df.iloc[:-VAL_LEN_6M], bu_df.iloc[-VAL_LEN_6M:]
        y_val = valid_df['TotalCallsOffered']

        # CWLB (short-horizon constraints; 1 can be used but capped; 3/6/12 must carry at least half combined)
        cwlb_constraints = {
            1: {'max': 0.5},
            3: {'min': 0.2},
            'combined': [{'lags': [3, 6, 12], 'min': 0.5}]
        }
        cwlb_weights, cwlb_smape, _ = find_best_weights(
            bu_ts, [1, 3, 6, 12], cwlb_constraints, VAL_LEN_6M, GRID_STEP
        )
        cwlb_smape_str = f"{cwlb_smape:.2f}" if np.isfinite(cwlb_smape) else "N/A"
        logger.info(f"  -> CWLB evaluated: SMAPE={cwlb_smape_str}")

        # XGBoost training (log-space target)
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

        # Store feature importances
        imp = {feat: score for feat, score in zip(xgb.feature_names_in_, xgb.feature_importances_)}
        xgb_imp_rows.append({'bu': bu_name, 'feature_importance': json.dumps(imp)})

        # Winner selection; compute long-horizon CWLB weights if CWLB wins
        if cwlb_smape <= xgb_smape:
            winner, winner_smape, winner_details = 'CWLB', cwlb_smape, cwlb_weights
            long_constraints = {3: {'min': 0.4}, 6: {'min': 0.2}, 12: {'min': 0.1}}
            long_weights, _, _ = find_best_weights(bu_ts, [3, 6, 12], long_constraints, 12, GRID_STEP)
        else:
            winner, winner_smape, winner_details = 'XGBoost', xgb_smape, {'n_estimators': 100, 'lr': 0.05, 'max_depth': 3}
            long_weights = None

        logger.info(f"✓ WINNER for {bu_name}: {winner} (SMAPE: {winner_smape:.2f})")
        audit_rows.append({
            'bu': bu_name,
            'winner': winner,
            'winner_smape': winner_smape,
            'cwlb_smape': cwlb_smape,
            'xgb_smape': xgb_smape,
            'details': json.dumps(winner_details)
        })

        # 5) Forecast generation
        if np.isfinite(winner_smape):
            logger.info(f"Generating {FORECAST_HORIZON}-month forecast with {winner}...")
            hist = deque(bu_ts.tolist(), maxlen=24)

            # Econ history used to form "future" lag features by re-using last known lags
            fred_hist_df = bu_df[[c for c in ECONOMIC_FEATURES if c in bu_df.columns]].copy()
            future_idx = pd.date_range(bu_ts.index[-1], periods=FORECAST_HORIZON + 1, freq='MS')[1:]

            for i, d in enumerate(future_idx, 1):
                pred = np.nan
                if winner == 'CWLB' and cwlb_weights:
                    if long_weights is None:
                        # Fallback: use CWLB weights throughout if long-weights missing
                        if i == 1:
                            lags_to_use = {1: hist[-1], 3: hist[-3], 6: hist[-6], 12: hist[-12]}
                        elif i in [2, 3]:
                            # approximate early steps
                            lags_to_use = {3: hist[-3], 6: hist[-6], 12: hist[-12]}
                        else:
                            lags_to_use = {3: hist[-3], 6: hist[-6], 12: hist[-12]}
                        pred = sum(cwlb_weights.get(l, 0) * v for l, v in lags_to_use.items())
                    else:
                        if i == 1:
                            w, lags_to_use = cwlb_weights, {1: hist[-1], 3: hist[-3], 6: hist[-6], 12: hist[-12]}
                        elif i in [2, 3]:
                            w, lags_to_use = cwlb_weights, {3: hist[-3], 6: hist[-6], 12: hist[-12]}
                        else:
                            w, lags_to_use = long_weights, {3: hist[-3], 6: hist[-6], 12: hist[-12]}
                        pred = sum(w.get(lag, 0) * value for lag, value in lags_to_use.items())

                elif winner == 'XGBoost':
                    # Build one-row feature frame for time d
                    row = {
                        'trend': (train_df['trend'].max() or 0) + len(valid_df) + i,
                        'sin_month': np.sin(2*np.pi*d.month/12),
                        'cos_month': np.cos(2*np.pi*d.month/12),
                        'lag_3': hist[-3] if len(hist) >= 3 else hist[-1],
                        'lag_6': hist[-6] if len(hist) >= 6 else hist[-1],
                        'lag_12': hist[-12] if len(hist) >= 12 else hist[-1],
                    }
                    for col in ECONOMIC_FEATURES:
                        for lag in [3, 6, 12]:
                            key = f'{col}_lag_{lag}'
                            if key in xgb_features:
                                # Reuse last known lagged econ level
                                row[key] = bu_df[col].iloc[-lag] if col in bu_df.columns and len(bu_df[col]) >= lag else np.nan

                    x_row_df = pd.DataFrame([row])[xgb_features]
                    pred = safe_expm1(xgb.predict(x_row_df)[0])
                else:
                    # Fallback to last value (should rarely happen)
                    pred = hist[-1]

                hist.append(pred)
                forecasts.append({
                    'fx_date': d,
                    'client_id': bu_name,
                    'vol_type': f'phone_{winner.lower()}',
                    'fx_vol': safe_round(pred),
                    'fx_id': f"{winner.lower()}_phone_{STAMP:%Y%m%d}",
                    'fx_status': "A",
                    'load_ts': STAMP
                })

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
        # WHERE t.fx_status = 'A' AND t.vol_type LIKE 'phone_%'
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
        # logger.info("✓ Older phone forecasts deactivated in GBQ.")
        #
        # bigquery_manager.update_log_in_bigquery()
        # logger.info("✓ Forecasting complete and results pushed to BigQuery.")
        # -------------------------------------------------------------------------

    else:
        logger.warning("No forecasts were generated.")

except Exception as exc:
    email_manager.handle_error("Credit Phone Production Script Failure", exc, is_test=True)
