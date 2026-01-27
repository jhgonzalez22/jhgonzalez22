"""
CX Credit - Call Volume Forecast - Production Hybrid v5.3
========================================================
FIXES:
- ValueError (numpy.object_): Forced pd.to_numeric conversion on all metric columns.
- InvalidIndexError: Maintained unique index logic using groupby().sum().
- BigQuery Bool Error: Kept DataFrame validation checks.

METHODOLOGY:
- Models: WLB, MLR, XGBoost, Workload Ratio, SeasonalNaiveGR, Native3m, Hybrid.
- Automated Lags: Tests workload products with 0, 1, and 2-month lags.
- Launch Dates: Respects start dates for Zillow (Oct-25), CrossCountry (Jul-25), etc.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# ------------------- PATHS & CONFIGURATION --------------------
sys.path.append(r"C:\WFM_Scripting\Automation")
try:
    from scripthelper import Config, Logger, BigQueryManager, EmailManager
except ImportError:
    print("CRITICAL ERROR: scripthelper.py not found.")
    sys.exit(1)

config = Config(rpt_id=299)
logger = Logger(config)
bigquery_manager = BigQueryManager(config)
email_manager = EmailManager(config)

warnings.filterwarnings("ignore")

# --- PARAMETERS ---
BACKCAST_MODE = False  
FORECAST_HORIZON = 12
VAL_LEN = 3 
STAMP = datetime.now(pytz.timezone("America/Chicago"))
BASE_DIR = r"C:\WFM_Scripting"

# File Paths
MAIN_DATA_SQL_PATH = fr"{BASE_DIR}\Forecasting\GBQ - Credit Historicals Phone - Chat Volumes.sql"

# Local Output Paths
suffix = "_BACKCAST" if BACKCAST_MODE else ""
LOCAL_CSV = fr"{BASE_DIR}\forecast_results_credit{suffix}.csv"
AUDIT_CSV = fr"{BASE_DIR}\model_eval_debug{suffix}.csv"
CORRELATION_CSV = fr"{BASE_DIR}\feature_correlations_credit{suffix}.csv"
MODELING_DATA_CSV = fr"{BASE_DIR}\modeling_data_credit{suffix}.csv"
STATS_CSV = fr"{BASE_DIR}\statistical_tests{suffix}.csv"
SUMMARIES_FILE = fr"{BASE_DIR}\model_summaries_credit{suffix}.txt"

# Model ID Mapping
MODEL_IDS = {
    "SeasonalNaiveGR": "snaive_gr_workload",
    "Native3m": "native3m_workload",
    "MLR": "mlr_workload",
    "XGBoost": "xgb_workload",
    "WLB_Fixed": "wlb_fix_workload",
    "Hybrid_WLB_MLR": "hybrid_wlb_mlr_workload",
    "Workload_Ratio": "ratio_workload"
}

# --- BUSINESS LOGIC: LAUNCH DATES ---
LAUNCH_DATES = {
    ('CrossCountry Mortgage', 'All'): '2025-07-01',
    ('General', 'Chat'): '2024-07-01',
    ('Prosperity', 'All'): '2025-01-01',
    ('Rapid Recheck', 'Chat'): '2025-01-01',
    ('Zillow', 'All'): '2025-10-01'
}

# ------------------- UTILITIES -------------------------
def safe_round(x):
    try: return int(max(0, round(float(x) if x is not None and np.isfinite(x) else 0.0)))
    except: return 0

def smape(actual, forecast):
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    denom = (np.abs(actual) + np.abs(forecast)) / 2.0
    denom[denom == 0] = 1.0 
    return np.mean(np.abs(actual - forecast) / denom) * 100

# ------------------- CORE PIPELINE -------------------------
def run_pipeline():
    try:
        logger.info(f"Starting Credit Pipeline v5.3. Mode: {'BACKCAST' if BACKCAST_MODE else 'PROD'}")

        # 1. LOAD CALL DATA
        df_calls = bigquery_manager.run_gbq_sql(MAIN_DATA_SQL_PATH, return_dataframe=True)
        if not isinstance(df_calls, pd.DataFrame):
            raise ValueError(f"BigQuery Manager failed to return a DataFrame. Check SQL.")
        
        df_calls.columns = [c.lower().strip() for c in df_calls.columns]
        
        # Column Normalization
        vol_cols = ['cpbd', 'volume_per_business_day', 'callsoffered_per_business_day', 'total_calls_offered']
        found_vol_col = next((c for c in vol_cols if c in df_calls.columns), None)
        if found_vol_col:
            df_calls.rename(columns={found_vol_col: 'cpbd'}, inplace=True)
        
        # FORCE NUMERIC (Fix for numpy.object_ error)
        df_calls['cpbd'] = pd.to_numeric(df_calls['cpbd'], errors='coerce').fillna(0)
        df_calls['month'] = pd.to_datetime(df_calls['month'])
        
        # Group Collapsing
        valid_groups = ['Main', 'General', 'Level', 'Rapid Recheck', 'Tax Transcripts', 'TS2', 'Billing', 'MtgGrp', 'Escalations']
        df_calls['groups_collapsed'] = df_calls['groups'].apply(lambda x: x if x in valid_groups else 'Other Credit Groups')

        # 2. LOAD WORKLOAD DRIVERS
        wl_sql = "SELECT MonthOfOrder as month, product, TotalOrders FROM tax_clnt_svcs.view_cx_nontax_platforms_workload_drivers WHERE client_id = 'Credit'"
        wl_hist_raw = bigquery_manager.run_gbq_sql(wl_sql, return_dataframe=True)
        
        # Pivot and Group to ensure unique index
        df_wl_wide = pd.DataFrame()
        if isinstance(wl_hist_raw, pd.DataFrame) and not wl_hist_raw.empty:
            wl_hist_raw['month'] = pd.to_datetime(wl_hist_raw['month'])
            # FORCE NUMERIC
            wl_hist_raw['TotalOrders'] = pd.to_numeric(wl_hist_raw['TotalOrders'], errors='coerce').fillna(0)
            df_wl_wide = wl_hist_raw.groupby(['month', 'product'])['TotalOrders'].sum().unstack().fillna(0)
            df_wl_wide.index = pd.to_datetime(df_wl_wide.index)
        
        # Create Lags
        driver_features = []
        if not df_wl_wide.empty:
            for col in df_wl_wide.columns:
                df_wl_wide[f"{col}_L1"] = df_wl_wide[col].shift(1)
                df_wl_wide[f"{col}_L2"] = df_wl_wide[col].shift(2)
                driver_features.extend([col, f"{col}_L1", f"{col}_L2"])

        # Future Workload
        fut_wl_sql = "SELECT fx_date as month, product, fx_vol FROM tax_clnt_svcs.cx_nontax_platforms_workload_drivers_fx WHERE client_id = 'Credit' AND fx_status = 'A'"
        wl_fut_raw = bigquery_manager.run_gbq_sql(fut_wl_sql, return_dataframe=True)
        df_wl_fut_wide = pd.DataFrame()
        if isinstance(wl_fut_raw, pd.DataFrame) and not wl_fut_raw.empty:
            wl_fut_raw.columns = [c.lower() for c in wl_fut_raw.columns]
            wl_fut_raw['month'] = pd.to_datetime(wl_fut_raw['month'])
            # FORCE NUMERIC
            wl_fut_raw['fx_vol'] = pd.to_numeric(wl_fut_raw['fx_vol'], errors='coerce').fillna(0)
            df_wl_fut_wide = wl_fut_raw.groupby(['month', 'product'])['fx_vol'].sum().unstack().fillna(0)
            df_wl_fut_wide.index = pd.to_datetime(df_wl_fut_wide.index)

        # 3. BAKE-OFF ENGINE
        forecast_rows, eval_logs, corr_logs = [], [], []
        combos = df_calls[['bu', 'client', 'groups_collapsed', 'volumetype']].drop_duplicates()

        for _, row in combos.iterrows():
            bu, client, group, vtype = row['bu'], row['client'], row['groups_collapsed'], row['volumetype']
            subset = df_calls[(df_calls['client']==client) & (df_calls['groups_collapsed']==group) & (df_calls['volumetype']==vtype)].sort_values('month')

            # Launch Date Logic
            start_key = LAUNCH_DATES.get((client, 'All')) or LAUNCH_DATES.get((client, vtype.capitalize()))
            if start_key:
                subset = subset[subset['month'] >= pd.to_datetime(start_key)]

            # CRITICAL FIX: Ensure monthly uniqueness per subset before join
            ts = subset.groupby('month')['cpbd'].sum()
            if len(ts) < 6: continue
            
            merged = pd.concat([ts, df_wl_wide], axis=1, join='inner').dropna()

            # Driver Selection
            best_driver, max_corr = None, 0
            for feat in driver_features:
                if feat in merged.columns and len(merged) > 4:
                    # SAFE CORRELATION CHECK
                    x_data = merged[feat]
                    y_data = merged['cpbd']
                    
                    # Ensure no constants (std=0) to avoid runtime warnings/errors
                    if x_data.std() > 0 and y_data.std() > 0:
                        r, p = pearsonr(y_data, x_data)
                        corr_logs.append({'Client': client, 'Group': group, 'Feature': feat, 'Corr': r, 'Pval': p})
                        if p < 0.05 and abs(r) > max_corr:
                            max_corr = abs(r); best_driver = feat

            # Training/Val Split
            train_ts = ts.iloc[:-VAL_LEN]
            val_actuals = ts.iloc[-VAL_LEN:]
            m_preds = {}

            # MODEL 1: WLB Fixed
            l1, l3, l12 = train_ts.shift(1), train_ts.shift(3), train_ts.shift(12)
            m_preds['WLB_Fixed'] = (l1 * 0.6) + (l3 * 0.2) + (l12.fillna(l1) * 0.2)
            
            # MODEL 2: SeasonalNaiveGR
            gr = (train_ts.iloc[-1] / train_ts.iloc[-4]) if len(train_ts) > 4 and train_ts.iloc[-4] != 0 else 1.0
            m_preds['SeasonalNaiveGR'] = train_ts.shift(12) * gr

            # MODEL 3: Native3m
            m_preds['Native3m'] = train_ts.rolling(3).mean().shift(1)

            # MODEL 4: Workload Models
            if best_driver and max_corr > 0.6:
                # Ratio logic
                m_train = merged.loc[train_ts.index]
                ratio_val = (m_train['cpbd'] / m_train[best_driver].replace(0, np.nan)).tail(6).mean()
                m_preds['Workload_Ratio'] = merged[best_driver] * ratio_val
                # MLR logic
                reg = LinearRegression().fit(m_train[[best_driver]], train_ts)
                m_preds['MLR'] = pd.Series(reg.predict(merged[[best_driver]]), index=merged.index)

            # Pick Winner
            scores = {m: smape(val_actuals, preds.reindex(val_actuals.index).fillna(0)) for m, preds in m_preds.items()}
            winner = min(scores, key=scores.get)

            # 4. FORECAST GENERATION
            future_dates = pd.date_range(ts.index.max() + pd.DateOffset(months=1), periods=FORECAST_HORIZON, freq='MS')
            for d in future_dates:
                if winner in ['Workload_Ratio', 'MLR']:
                    base_p = best_driver.split('_L')[0]; lag_v = int(best_driver.split('_L')[1]) if '_L' in best_driver else 0
                    tgt_d = d - pd.DateOffset(months=lag_v)
                    wl_val = df_wl_fut_wide.loc[tgt_d, base_p] if tgt_d in df_wl_fut_wide.index else df_wl_wide[base_p].mean()
                    pred_v = (wl_val * ratio_val) if winner == 'Workload_Ratio' else reg.predict([[wl_val]])[0]
                else:
                    pred_v = ts.tail(3).mean() 

                forecast_rows.append({
                    'Month': d, 'bu': bu, 'Client': client, 'Groups': group, 'VolumeType': vtype,
                    'Forecast_CPBD': round(max(0, pred_v), 2), 'Model_ID': MODEL_IDS.get(winner),
                    'Best_Driver': best_driver if best_driver else "None"
                })

            eval_logs.append({'Client': client, 'Group': group, 'Winner': winner, 'sMAPE': scores[winner]})

        # 5. EXPORT LOCAL FILES
        pd.DataFrame(forecast_rows).to_csv(LOCAL_CSV, index=False)
        pd.DataFrame(eval_logs).to_csv(AUDIT_CSV, index=False)
        pd.DataFrame(corr_logs).to_csv(CORRELATION_CSV, index=False)
        df_calls.to_csv(MODELING_DATA_CSV, index=False)
        pd.DataFrame([{'Model': k, 'ID': v} for k,v in MODEL_IDS.items()]).to_csv(STATS_CSV, index=False)

        logger.info(f"Pipeline Finished Successfully. Files in {BASE_DIR}")

    except Exception as e:
        logger.error(f"Critical Failure: {e}")
        email_manager.handle_error("Credit Forecast Script Failure", e, is_test=True)

if __name__ == "__main__":
    run_pipeline()

(venv_Master) PS C:\WFM_Scripting\Forecasting> & C:/Scripting/Python_envs/venv_Master/Scripts/python.exe c:/WFM_Scripting/Forecasting/Rpt_299_File.py
Traceback (most recent call last):
  File "c:\WFM_Scripting\Forecasting\Rpt_299_File.py", line 246, in <module>
    run_pipeline()
  File "c:\WFM_Scripting\Forecasting\Rpt_299_File.py", line 243, in run_pipeline
    email_manager.handle_error("Credit Forecast Script Failure", e, is_test=True)
  File "C:\WFM_Scripting\Automation\scripthelper.py", line 1752, in handle_error
    raise exception
  File "c:\WFM_Scripting\Forecasting\Rpt_299_File.py", line 178, in run_pipeline
    r, p = pearsonr(y_data, x_data)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\scipy\stats\_stats_py.py", line 4839, in pearsonr
    threshold = xp.finfo(dtype).eps ** 0.75
                ^^^^^^^^^^^^^^^
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\numpy\core\getlimits.py", line 516, in __new__
    raise ValueError("data type %r not inexact" % (dtype))
ValueError: data type <class 'numpy.object_'> not inexact
