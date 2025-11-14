"""
Answerlink - Call Volume Forecast - Production Hybrid v12.4 (Local Save)

WHAT THIS SCRIPT DOES
---------------------
Generates 12-month forecasts using a 5-model bake-off:
[OLS, WBL, Native3mGR, SARIMAX, XGBoost]

FEATURE LOGIC:
- Models are forced to use specific, user-defined features.
- OLS/WBL/Native run on "DUMB" (imputed) data.
- SARIMAX/XGBoost run on "SMART" (real spikes) data.

FIX v12.4:
- Solves the "Chat forecast is 0" bug caused by feature mismatch.
- DELETES the complex 'build_future_X' helper function.
- REPLACES it with a simpler, more robust ffill() and lag creation
  process inside the main forecast generation loop. This is far less
  prone to error and ensures the model gets the exact features it expects.
- This script also retains the WBL constraint logic from v12.3.

ADDITIONAL FIX:
- For XGBoost winner, we no longer force n_estimators = best_iteration
  when refitting on full data (which could collapse the model to 0 trees
  and give all-zero chat forecasts). Instead we reuse the same params
  (minus early_stopping_rounds) and refit cleanly on full SMART data.
"""

import os
import sys
import warnings
import pytz
import json
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
from scipy.stats import pearsonr

# --- Model Imports ---
import statsmodels.api as sm
import pmdarima as pm
import xgboost as xgb

warnings.filterwarnings("ignore")

# ------------------- HELPER-PACKAGE PATH --------------------
sys.path.append(r"C:\WFM_Scripting\Automation")
try:
    from scripthelper import Config, Logger, BigQueryManager, EmailManager
except ImportError:
    print("CRITICAL: 'scripthelper' not found. Ensure C:\\WFM_Scripting\\Automation is in your path.")
    sys.exit(1)

# ------------------- INITIALISE SCRIPTHELPER ----------------
config = Config(rpt_id=301) 
logger = Logger(config)
email_manager = EmailManager(config)
bigquery_manager = BigQueryManager(config)

# ------------------- CONFIGURATION --------------------
# SQL Paths
MAIN_DATA_SQL_PATH = r"C:\WFM_Scripting\Forecasting\GBQ - Non-Tax Answerlink Phone Chat Timeseries by Month.sql"
CALENDAR_SQL_PATH = r"C:\WFM_Scripting\Forecasting\GBQ - Calendar days.sql"

# Output Paths
BASE_PATH = r"C:\WFM_Scripting"
FORECAST_RESULTS_CSV = os.path.join(BASE_PATH, "forecast_results_answerlink_fy26.csv")
MODEL_EVAL_CSV = os.path.join(BASE_PATH, "model_eval_answerlink_fy26.csv")
MODELING_DATA_SMART_CSV = os.path.join(BASE_PATH, "modeling_data_smart.csv")
MODELING_DATA_DUMB_CSV = os.path.join(BASE_PATH, "modeling_data_dumb.csv")
CORRELATION_ANALYSIS_CSV = os.path.join(BASE_PATH, "correlation_analysis_answerlink.csv")
FUTURE_EXOG_CSV = os.path.join(BASE_PATH, "future_exog_answerlink.csv")

# Settings
FORECAST_HORIZON = 12
VAL_LEN_3M = 3
GRID_STEP = 0.05
TARGET_COL = 'CallsOffered'

# ------------------- FEATURE DEFINITIONS (PER USER) --------------------
# Features for SMART models (SARIMAX, XGBoost)
PHONE_FEATURES = ['MORTGAGE30US_lag3', 'CallsOffered_lag12', 'holiday_count', 'special_event']
CHAT_FEATURES = ['FEDFUNDS_lag12', 'CallsOffered_lag12', 'UNRATE_lag6', 'special_event']

# Features for DUMB models (OLS) - Event flag removed
OLS_PHONE_FEATURES = ['MORTGAGE30US_lag3', 'CallsOffered_lag12', 'holiday_count']
OLS_CHAT_FEATURES = ['FEDFUNDS_lag12', 'CallsOffered_lag12', 'UNRATE_lag6']

# All unique lags needed to create
ALL_LAGS_NEEDED = {
    'CallsOffered': [1, 2, 3, 12], # 1,2,3 for WBL, 12 for models
    'MORTGAGE30US': [3],
    'FEDFUNDS': [12],
    'UNRATE': [6]
}
# Base features needed from SQL
BASE_FEATURES = ['holiday_count', 'business_day_count', 'special_event', 'MORTGAGE30US', 'FEDFUNDS', 'UNRATE', 'HSN1F']

# ------------------- HARDCODED FUTURE DATA (Fannie Mae) --------------------
FUTURE_ECON_DATA = {
    'Month': pd.to_datetime([
        '2025-11-01', '2025-12-01', '2026-01-01', '2026-02-01', 
        '2026-03-01', '2026-04-01', '2026-05-01', '2026-06-01', 
        '2026-07-01', '2026-08-01', '2026-09-01', '2026-10-01'
    ]),
    'UNRATE': [4.40, 4.43, 4.47, 4.50, 4.50, 4.50, 4.50, 4.50, 4.50, 4.50, 4.50, 4.50],
    'HSN1F': [907, 908, 913, 920, 927, 934, 941, 951, 958, 964, 968, 971],
    'FEDFUNDS': [3.77, 3.70, 3.60, 3.53, 3.47, 3.40, 3.40, 3.40, 3.33, 3.27, 3.20, 3.13],
    'MORTGAGE30US': [6.37, 6.30, 6.23, 6.17, 6.10, 6.03, 6.00, 5.97, 5.93, 5.90, 5.90, 5.90],
    'special_event': [0] * 12
}

# ------------------- UTILITIES -------------------------
def smape(actual, forecast):
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    denom = (np.abs(actual) + np.abs(forecast)) / 2.0
    denom[denom == 0] = 1.0
    return np.mean(np.abs(actual - forecast) / denom) * 100

def safe_round(x):
    try:
        return int(max(0, round(float(x) if x is not None and np.isfinite(x) else 0.0)))
    except:
        return 0

def compute_3m_yoy_factor(series):
    s, ratios = series.dropna(), []
    for k in [1, 2, 3]:
        if len(s) > 12 + k:
            num, den = s.iloc[-k], s.iloc[-12 - k]
            if den and den != 0:
                ratios.append(num / den)
    return float(np.mean(ratios)) if ratios else 1.0

def find_best_weights(ts, lags, val_len, step, constraints=None):
    if not lags or len(ts) < val_len + max(lags):
        return None, np.inf
    validation_data, best_weights, best_smape = ts.iloc[-val_len:], None, np.inf
    weight_iters = [np.arange(0, 1.0 + step, step) for _ in range(len(lags) - 1)] if len(lags) > 1 else [[1.0]]
    for W in itertools.product(*weight_iters):
        if sum(W) > 1.0 + 1e-9:
            continue
        all_w = list(W) if len(lags) == 1 else list(W) + [1.0 - sum(W)]
        weights = {int(l): float(w) for l, w in zip(lags, all_w)}
        if constraints:
            valid = True
            for lag, limits in constraints.items():
                if lag in weights:
                    if 'min' in limits and weights[lag] < (limits['min'] - 1e-9):
                        valid = False
            if not valid:
                continue
        
        forecast_vals = 0
        for lag, w in weights.items():
            forecast_vals += ts.shift(lag).reindex(validation_data.index) * w
        
        if forecast_vals.isna().any():
            continue
        current_smape = smape(validation_data, forecast_vals)
        if current_smape < best_smape:
            best_smape, best_weights = current_smape, weights
    return best_weights, best_smape

def perform_correlation_analysis(full_data_long: pd.DataFrame, lag_list: list):
    logger.info("Running correlation analysis...")
    correlation_results = []
    series_definitions = [{'type': 'Phone'}, {'type': 'Chat'}]
    
    features_to_lag = [TARGET_COL] + ['UNRATE', 'HSN1F', 'FEDFUNDS', 'MORTGAGE30US', 'business_day_count', 'holiday_count']

    for series in series_definitions:
        series_data_raw = full_data_long[
            (full_data_long['VolumeType'] == series['type'])
        ].set_index('Month').sort_index().asfreq('MS')
        
        if series_data_raw.empty:
            continue
        
        analysis_df = series_data_raw.copy()
        for col in features_to_lag:
            if col in analysis_df.columns:
                for lag in lag_list:
                    analysis_df[f'{col}_lag{lag}'] = analysis_df[col].shift(lag)
        
        all_cols = analysis_df.select_dtypes(include=np.number).columns
        feature_cols = [col for col in all_cols if col not in ['Month', TARGET_COL, 'HandledCallsTime', 'Avg_Handle_Time']]
        
        analysis_df = analysis_df.dropna(subset=[TARGET_COL] + feature_cols)
        if analysis_df.empty:
            continue
            
        target_series = analysis_df[TARGET_COL]
        
        for col in feature_cols:
            predictor_series = analysis_df[col]
            if target_series.empty or predictor_series.empty:
                continue
            if np.std(target_series) == 0 or np.std(predictor_series) == 0:
                continue
            
            try:
                corr, p_val = pearsonr(target_series, predictor_series)
                correlation_results.append({
                    'series': f"Answerlink_{series['type']}", 
                    'variable': col, 
                    'pearson_correlation': float(corr), 
                    'p_value': float(p_val)
                })
            except Exception as e:
                logger.warning(f"  Corr failed for {col}: {e}")
                continue
    return correlation_results

def create_series_lags(df_series, lag_config):
    df_out = df_series.copy()
    for col, lags in lag_config.items():
        if col in df_out.columns:
            for lag in lags:
                df_out[f'{col}_lag{lag}'] = df_out[col].shift(lag)
    return df_out

# ------------------- MAIN PIPELINE -------------------------
def run_pipeline():
    try:
        logger.info("Starting Answerlink Production Hybrid v12.4...")

        # 1. LOAD MAIN DATA
        logger.info(f"Reading Main SQL: {MAIN_DATA_SQL_PATH}")
        df = bigquery_manager.run_gbq_sql(MAIN_DATA_SQL_PATH, return_dataframe=True)
        if df is None or df.empty:
            raise ValueError("BigQuery returned no data.")
        df['Month'] = pd.to_datetime(df['Month'])
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce').fillna(0).astype('float64')
        for col in ['UNRATE', 'HSN1F', 'FEDFUNDS', 'MORTGAGE30US']:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan).ffill().bfill()
        
        # 2. LOAD CALENDAR DATA
        logger.info(f"Reading Calendar SQL: {CALENDAR_SQL_PATH}")
        df_cal = bigquery_manager.run_gbq_sql(CALENDAR_SQL_PATH, return_dataframe=True)
        if df_cal is None or df_cal.empty:
            raise ValueError("BigQuery returned no Calendar data.")
        cal_date_col = next((c for c in df_cal.columns if 'date' in c.lower() or 'month' in c.lower()), None)
        df_cal[cal_date_col] = pd.to_datetime(df_cal[cal_date_col])
        df_cal_monthly = df_cal.set_index(cal_date_col).resample('MS')[['business_day_count', 'holiday_count']].sum().reset_index()
        df_cal_monthly.rename(columns={cal_date_col: 'Month'}, inplace=True)

        # 3. PREPARE FUTURE EXOG
        df_future_econ = pd.DataFrame(FUTURE_ECON_DATA)
        df_future_exog = pd.merge(df_future_econ, df_cal_monthly, on='Month', how='left')
        df_future_exog['business_day_count'].fillna(21, inplace=True)
        df_future_exog['holiday_count'].fillna(0, inplace=True)
        df_future_exog = df_future_exog.set_index('Month')
        
        df_future_exog.to_csv(FUTURE_EXOG_CSV)
        logger.info(f"Future Exog saved to {FUTURE_EXOG_CSV}")

        # 4. MERGE CALENDAR INTO MAIN DF
        cols_to_drop = [c for c in ['business_day_count', 'holiday_count'] if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
        df = pd.merge(df, df_cal_monthly, on='Month', how='left')
        df[['business_day_count', 'holiday_count']] = df[['business_day_count', 'holiday_count']].fillna(method='ffill').fillna(20)

        # 5. FEATURE ENGINEERING & BACKFILLING
        df['special_event'] = 0
        event_mask = (df['Month'].isin(['2024-12-01', '2025-01-01'])) & (df['VolumeType'] == 'Phone')
        df.loc[event_mask, 'special_event'] = 1

        # Chat Backfill
        chat_mask = (df['VolumeType'] == 'Chat')
        jan_24_mask = chat_mask & (df['Month'] == '2024-01-01')
        feb_24_mask = chat_mask & (df['Month'] == '2024-02-01')
        try:
            feb_val = df.loc[feb_24_mask, TARGET_COL].values[0]
            if not df.loc[jan_24_mask, TARGET_COL].empty and df.loc[jan_24_mask, TARGET_COL].values[0] < 10:
                df.loc[jan_24_mask, TARGET_COL] = feb_val
        except IndexError:
            pass

        months_2023 = pd.date_range('2023-01-01', '2023-12-01', freq='MS')
        for m in months_2023:
            m_future = m + pd.DateOffset(years=1)
            try:
                val = df.loc[chat_mask & (df['Month'] == m_future), TARGET_COL].values[0]
                mask_23 = chat_mask & (df['Month'] == m)
                df.loc[mask_23, TARGET_COL] = val
            except IndexError:
                pass
        
        # --- Create SMART and DUMB dataframes (long-format) ---
        df_smart = df.copy()

        df_dumb = df.copy()
        phone_median = df_dumb.loc[(df_dumb['VolumeType'] == 'Phone') & (df_dumb['special_event'] == 0), TARGET_COL].median()
        df_dumb.loc[(df_dumb['VolumeType'] == 'Phone') & (df_dumb['special_event'] == 1), TARGET_COL] = phone_median
        
        df_smart.to_csv(MODELING_DATA_SMART_CSV, index=False)
        df_dumb.to_csv(MODELING_DATA_DUMB_CSV, index=False)

        # 6. CORRELATION ANALYSIS
        corr_results = perform_correlation_analysis(df_smart, [1, 3, 6, 12]) 
        if corr_results:
            pd.DataFrame(corr_results).to_csv(CORRELATION_ANALYSIS_CSV, index=False)

        # 7. MODEL BAKE-OFF
        forecasts, audit_rows = [], []
        series_defs = [
            {'name': 'Answerlink_Phone', 'type': 'Phone', 'features': PHONE_FEATURES, 'ols_features': OLS_PHONE_FEATURES},
            {'name': 'Answerlink_Chat', 'type': 'Chat', 'features': CHAT_FEATURES, 'ols_features': OLS_CHAT_FEATURES}
        ]

        for ser in series_defs:
            logger.info(f"\nProcessing {ser['name']}...")
            
            # --- Get SMART Data (for SARIMAX, XGB) ---
            ts_df_smart_raw = df_smart[df_smart['VolumeType'] == ser['type']].set_index('Month').sort_index().asfreq('MS')
            ts_df_smart = create_series_lags(ts_df_smart_raw, ALL_LAGS_NEEDED)
            train_smart = ts_df_smart.iloc[:-VAL_LEN_3M]
            valid_smart = ts_df_smart.iloc[-VAL_LEN_3M:]
            y_true = valid_smart[TARGET_COL] 
            
            # --- Get DUMB Data (for OLS, WBL, Native) ---
            ts_df_dumb_raw = df_dumb[df_dumb['VolumeType'] == ser['type']].set_index('Month').sort_index().asfreq('MS')
            ts_df_dumb = create_series_lags(ts_df_dumb_raw, ALL_LAGS_NEEDED)
            train_dumb = ts_df_dumb.iloc[:-VAL_LEN_3M]
            valid_dumb = ts_df_dumb.iloc[-VAL_LEN_3M:]

            smapes = {}
            models = {}
            
            # --- Define conditional WBL params ---
            if ser['type'] == 'Phone':
                wbl_lags = [1, 3, 6, 12]
                wbl_constraints = {12: {'min': 0.15}} # Force seasonality
            else: # Chat
                wbl_lags = [1, 2, 3]
                wbl_constraints = None # Pure trend
            
            # --- Model 1: OLS (On DUMB Data) ---
            try:
                X_train = train_dumb[ser['ols_features']].dropna()
                y_train = train_dumb.loc[X_train.index, TARGET_COL]
                X_train_const = sm.add_constant(X_train)
                
                ols_model = sm.OLS(y_train, X_train_const).fit()
                
                X_valid = valid_dumb[ser['ols_features']].dropna()
                y_valid = y_true.loc[X_valid.index] 
                X_valid_const = sm.add_constant(X_valid, has_constant='add')
                
                preds = ols_model.predict(X_valid_const)
                smapes['OLS'] = smape(y_valid, preds)
                models['OLS'] = ols_model
                logger.info(f"  OLS SMAPE: {smapes['OLS']:.2f}")
            except Exception as e:
                smapes['OLS'] = np.inf
                logger.warning(f"  OLS failed: {e}")

            # --- Model 2: WBL (On DUMB Data) ---
            try:
                wts, score = find_best_weights(ts_df_dumb[TARGET_COL], wbl_lags, VAL_LEN_3M, GRID_STEP, constraints=wbl_constraints)
                smapes['WBL'] = score
                models['WBL'] = wts
                logger.info(f"  WBL SMAPE: {smapes['WBL']:.2f} (Lags: {wbl_lags}, Constraints: {wbl_constraints})")
            except Exception as e:
                smapes['WBL'] = np.inf
                logger.warning(f"  WBL failed: {e}")

            # --- Model 3: Native3mGR (On DUMB Data) ---
            try:
                factor = compute_3m_yoy_factor(train_dumb[TARGET_COL])
                preds = [ts_df_dumb[TARGET_COL].get(d - pd.DateOffset(years=1), 0) * factor for d in valid_dumb.index]
                smapes['Native3mGR'] = smape(y_true, preds)
                models['Native3mGR'] = factor
                logger.info(f"  Native3mGR SMAPE: {smapes['Native3mGR']:.2f}")
            except Exception as e:
                smapes['Native3mGR'] = np.inf
                logger.warning(f"  Native3mGR failed: {e}")

            # --- Model 4: SARIMAX (On SMART Data) ---
            try:
                X_train = train_smart[ser['features']].dropna()
                y_train = train_smart.loc[X_train.index, TARGET_COL]

                arima = pm.auto_arima(
                    y_train,
                    exogenous=X_train,
                    m=12,
                    seasonal=True,
                    suppress_warnings=True,
                    stepwise=True,
                    error_action='ignore'
                )
                
                X_valid = valid_smart[ser['features']].dropna()
                y_valid = y_true.loc[X_valid.index] 

                preds = arima.predict(n_periods=len(X_valid), exogenous=X_valid)
                smapes['SARIMAX'] = smape(y_valid, preds)
                models['SARIMAX'] = arima
                logger.info(f"  SARIMAX SMAPE: {smapes['SARIMAX']:.2f}")
            except Exception as e:
                smapes['SARIMAX'] = np.inf
                logger.warning(f"  SARIMAX failed: {e}")

            # --- Model 5: XGBoost (On SMART Data) ---
            try:
                X_train = train_smart[ser['features']].dropna()
                y_train = train_smart.loc[X_train.index, TARGET_COL]
                
                X_valid = valid_smart[ser['features']].dropna()
                y_valid = y_true.loc[X_valid.index] 

                if X_valid.empty:
                    raise ValueError("XGB Validation set is empty after dropna")

                reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=20)
                reg.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
                
                preds = reg.predict(X_valid)
                smapes['XGBoost'] = smape(y_valid, preds)
                models['XGBoost'] = reg
                logger.info(f"  XGBoost SMAPE: {smapes['XGBoost']:.2f}")
            except Exception as e:
                smapes['XGBoost'] = np.inf
                logger.warning(f"  XGBoost failed: {e}")

            # --- Winner ---
            valid_scores = {k: v for k, v in smapes.items() if v < np.inf}
            best_model_name = min(valid_scores, key=valid_scores.get) if valid_scores else "None"
            logger.info(f"  WINNER: {best_model_name} (SMAPE: {valid_scores.get(best_model_name):.2f})")
            
            audit_rows.append({
                'series': ser['name'],
                'winner': best_model_name,
                'winner_smape': valid_scores.get(best_model_name, np.inf),
                'smape_OLS': smapes.get('OLS'),
                'smape_WBL': smapes.get('WBL'),
                'smape_Native3mGR': smapes.get('Native3mGR'),
                'smape_SARIMAX': smapes.get('SARIMAX'),
                'smape_XGBoost': smapes.get('XGBoost')
            })

            # --- 8. FORECAST GENERATION ---
            if best_model_name == "None":
                logger.error(f"  No valid models for {ser['name']}. Skipping forecast.")
                continue

            future_preds = []
            
            # --- ** NEW, SIMPLIFIED FUTURE DATAFRAME CREATION ** ---
            
            # 1. Get the correct history (Dumb for WBL/Native/OLS, Smart for SARIMAX/XGB)
            if best_model_name in ['OLS', 'WBL', 'Native3mGR']:
                hist_df = ts_df_dumb_raw.copy()
            else:
                hist_df = ts_df_smart_raw.copy()
                
            # 2. Combine history with future exog, ffill() to fill missing econ values
            # This carries the last known econ values (Oct 2025) into the future
            future_df = pd.concat([hist_df, df_future_exog])
            future_df[BASE_FEATURES] = future_df[BASE_FEATURES].fillna(method='ffill')
            
            # 3. Create lags on this combined dataframe
            future_df_with_lags = create_series_lags(future_df, ALL_LAGS_NEEDED)
            
            # 4. Select only the future 12 months for prediction
            X_future = future_df_with_lags.loc[df_future_exog.index]
            
            # --- End of new logic ---

            if best_model_name == 'OLS':
                model = models['OLS']
                # Refit on full dumb data
                X_full = ts_df_dumb[ser['ols_features']].dropna()
                y_full = ts_df_dumb.loc[X_full.index, TARGET_COL]
                model = sm.OLS(y_full, sm.add_constant(X_full)).fit()
                
                # Select the exact columns for prediction
                X_future_const = sm.add_constant(X_future[ser['ols_features']], has_constant='add')
                future_preds = model.predict(X_future_const)
            
            elif best_model_name == 'WBL':
                wts = models['WBL']
                hist_deque = deque(ts_df_dumb_raw[TARGET_COL].dropna().tolist(), maxlen=max(wts.keys()) + 1)
                for _ in range(FORECAST_HORIZON):
                    lags_to_use = {L: hist_deque[-L] for L in wts if len(hist_deque) >= L}
                    pred = sum(wts.get(L, 0.0) * val for L, val in lags_to_use.items())
                    future_preds.append(pred)
                    hist_deque.append(pred)

            elif best_model_name == 'Native3mGR':
                factor = models['Native3mGR']
                hist_data = ts_df_dumb_raw[TARGET_COL].copy()
                for d in df_future_exog.index:
                    last_year_val = hist_data.get(d - pd.DateOffset(years=1), 0)
                    pred = last_year_val * factor
                    future_preds.append(pred)
                    hist_data[d] = pred

            elif best_model_name == 'SARIMAX':
                model = models['SARIMAX']
                # Refit on full smart data
                X_full = ts_df_smart[ser['features']].dropna()
                y_full = ts_df_smart.loc[X_full.index, TARGET_COL]
                model.fit(y_full, exogenous=X_full)
                
                # Select the exact columns for prediction
                future_preds = model.predict(n_periods=FORECAST_HORIZON, exogenous=X_future[ser['features']])

            elif best_model_name == 'XGBoost':
                # Reuse the same hyperparameters as the validation model,
                # but DO NOT force n_estimators to best_iteration.
                validation_model = models['XGBoost']
                params = validation_model.get_params()

                # Remove early stopping from the final full-data refit
                params.pop('early_stopping_rounds', None)

                final_model = xgb.XGBRegressor(**params)

                # Refit on full smart data
                X_full = ts_df_smart[ser['features']].dropna()
                y_full = ts_df_smart.loc[X_full.index, TARGET_COL]
                final_model.fit(X_full, y_full)

                # Select the exact columns for prediction
                future_preds = final_model.predict(X_future[ser['features']])

            # --- Save Forecasts ---
            for i, d in enumerate(df_future_exog.index):
                forecasts.append({
                    'Date': d, 
                    'Channel': ser['type'], 
                    'Model': best_model_name, 
                    'Forecast': safe_round(future_preds[i] if isinstance(future_preds, list) else future_preds[i])
                })

        # 9. SAVE ALL RESULTS
        if audit_rows:
            pd.DataFrame(audit_rows).to_csv(MODEL_EVAL_CSV, index=False)
        if forecasts:
            pd.DataFrame(forecasts).to_csv(FORECAST_RESULTS_CSV, index=False)
        
        logger.info("\nScript finished successfully.")
        logger.info(f"Forecasts saved to: {FORECAST_RESULTS_CSV}")
        logger.info(f"Model Eval saved to: {MODEL_EVAL_CSV}")

    except Exception as exc:
        logger.error(f"Script failed: {exc}")
        email_manager.handle_error("Answerlink Forecast Failure", exc, is_test=True)

if __name__ == "__main__":
    run_pipeline()
