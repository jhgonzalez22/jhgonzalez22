"""
CX Platforms Phone Forecast — v2.4.5
=========================================
Purpose: Forecasts monthly phone call volumes using a tournament of statistical 
         and machine learning models.
Features: 
  - Normalizes target to 'Offered Per Business Day'.
  - Integrates FRED Market Data (Mortgage Rates) as leading indicators.
  - Integrates Workload Drivers (Orders) for Ratio-based modeling.
  - Back-casts from 2025-01-01 for validation.
  - Outputs detailed local audit files for analysis.

Fixes in v2.4.5:
  - Fixed AttributeError in smape function (numpy array has no fillna).
  - Enhanced NaN handling using np.nan_to_num.

Author: Automation Team
Last Updated: 2026-01-29
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from xgboost import XGBRegressor

# Ensure this points to your automation folder
sys.path.append(r"C:\WFM_Scripting\Automation")

from scripthelper import Config, Logger, BigQueryManager, EmailManager

# ------------------- 1. Configuration & File Paths -------------------
config = Config(rpt_id=283)
logger = Logger(config)
email_manager = EmailManager(config)
bq_manager = BigQueryManager(config)

# SQL Query Paths
SQL_QUERY_PATH = r"C:\WFM_Scripting\Forecasting\GBQ - Non-Tax Platform Phone Timeseries by Month.sql"

# Local Output Paths
LOCAL_CSV         = r"C:\WFM_Scripting\forecast_results_phone.csv"
AUDIT_CSV         = r"C:\WFM_Scripting\model_eval_debug_phone.csv"
STATS_CSV         = r"C:\WFM_Scripting\statistical_tests_phone.csv"
XGB_VIZ_CSV       = r"C:\WFM_Scripting\xgb_feature_importance_phone.csv"
MODELING_DATA_CSV = r"C:\WFM_Scripting\modeling_data_phone.csv"
SUMMARIES_FILE    = r"C:\WFM_Scripting\model_summaries_phone.txt"

# Forecasting Parameters
BACKCAST_START = '2025-01-01'
HORIZON = 15
LAG1_MAX_WEIGHT = 0.60
STAMP = datetime.now()

# ------------------- 2. Helper Functions -------------------
def smape(actual, forecast):
    """
    Calculates Symmetric Mean Absolute Percentage Error (SMAPE).
    FIX: Uses np.nan_to_num to handle NaNs safely for both arrays and series.
    """
    # Force conversion to numeric, coerce errors to NaN
    a = pd.to_numeric(actual, errors='coerce')
    f = pd.to_numeric(forecast, errors='coerce')
    
    # Convert to numpy float array and replace NaN/Inf with 0
    a = np.nan_to_num(np.array(a, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    f = np.nan_to_num(np.array(f, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    
    denom = (np.abs(a) + np.abs(f)) / 2.0
    
    # Avoid division by zero
    return np.mean(np.abs(a - f) / np.where(denom == 0, 1.0, denom)) * 100

# ------------------- 3. Main Execution Pipeline -------------------
try:
    logger.info("Starting Pipeline v2.4.5 - Fixes applied...")

    # --- Step 3a: Fetch FRED Market Metrics (History + Forecast) ---
    fred_sql = """
    SELECT Date as date, UNRATE, HSN1F, FEDFUNDS, MORTGAGE30US 
    FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.fred`
    UNION ALL
    SELECT Date as date, UNRATE, HSN1F, FEDFUNDS, MORTGAGE30US 
    FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.fred_fx`
    QUALIFY ROW_NUMBER() OVER(PARTITION BY Date ORDER BY Forecast_Date DESC) = 1
    ORDER BY date
    """
    df_fred = bq_manager.run_gbq_sql(fred_sql, return_dataframe=True)
    df_fred['date'] = pd.to_datetime(df_fred['date'])
    
    # Forward fill market data
    cols_to_fill = ['UNRATE', 'HSN1F', 'FEDFUNDS', 'MORTGAGE30US']
    df_fred[cols_to_fill] = df_fred[cols_to_fill].apply(pd.to_numeric, errors='coerce').ffill().bfill()

    # --- Step 3b: Fetch Workload Drivers (Orders) ---
    order_hist_sql = """
    SELECT MonthOfOrder as date, client_id, TotalOrders 
    FROM tax_clnt_svcs.view_cx_nontax_platforms_workload_drivers
    """
    order_fx_sql = """
    SELECT fx_date as date, client_id, fx_vol as TotalOrders 
    FROM tax_clnt_svcs.cx_nontax_platforms_workload_drivers_fx 
    WHERE fx_status = 'A'
    """
    df_ord_hist = bq_manager.run_gbq_sql(order_hist_sql, return_dataframe=True)
    df_ord_fx = bq_manager.run_gbq_sql(order_fx_sql, return_dataframe=True)
    
    df_orders = pd.concat([df_ord_hist, df_ord_fx]).drop_duplicates(subset=['date', 'client_id'], keep='last')
    df_orders['date'] = pd.to_datetime(df_orders['date'])
    # Safe convert to Series then fillna
    df_orders['TotalOrders'] = pd.to_numeric(df_orders['TotalOrders'], errors='coerce').fillna(0)

    # Map Parent Drivers
    DRIVER_MAP = {
        'FNC - CMS': 'FNC', 
        'FNC - Ports': 'FNC', 
        'Mercury Integrations': 'Mercury',
        'Appraisal Scope': 'Appraisal Scope'
    }

    # --- Step 3c: Fetch Base Call Data & Normalize ---
    df_calls = bq_manager.run_gbq_sql(SQL_QUERY_PATH, return_dataframe=True)
    df_calls['date'] = pd.to_datetime(df_calls['date'])
    
    # Safe convert
    df_calls['Total_Offered'] = pd.to_numeric(df_calls['Total_Offered'], errors='coerce').fillna(0)
    df_calls['business_day_count'] = pd.to_numeric(df_calls['business_day_count'], errors='coerce').fillna(21)
    
    # TARGET: Rate per Business Day
    df_calls['target_rate'] = np.where(
        df_calls['business_day_count'] > 0,
        df_calls['Total_Offered'] / df_calls['business_day_count'],
        0
    )

    # --- Step 3d: Master Merge ---
    df_calls['join_key'] = df_calls['client_id'].map(DRIVER_MAP).fillna(df_calls['client_id'])
    
    df = pd.merge(df_calls, df_fred, on='date', how='left')
    df = pd.merge(df, df_orders.rename(columns={'client_id': 'join_key'}), on=['date', 'join_key'], how='left')
    
    # Fill remaining nulls in drivers/market data
    metric_cols = ['TotalOrders'] + cols_to_fill
    df[metric_cols] = df.groupby('client_id')[metric_cols].ffill()

    # --- Step 4: Feature Engineering ---
    df = df.sort_values(['client_id', 'date'])
    
    # Create Lags
    for l in [1, 2, 3, 6, 12]:
        df[f'lag_{l}'] = df.groupby('client_id')['target_rate'].shift(l)
        df[f'order_lag_{l}'] = df.groupby('client_id')['TotalOrders'].shift(l)
        df[f'mkt_lag_{l}'] = df.groupby('client_id')['MORTGAGE30US'].shift(l)
    
    # Efficiency Metric
    with np.errstate(divide='ignore', invalid='ignore'):
        df['calls_per_order'] = df['Total_Offered'] / df['TotalOrders']
    df['calls_per_order'] = df['calls_per_order'].fillna(0).replace([np.inf, -np.inf], 0)

    forecast_results = []
    audit_log = []
    
    # --- Step 5: Client Modeling Loop ---
    for cid in df['client_id'].unique():
        c_df = df[df['client_id'] == cid].copy()
        
        # Skip if not enough history
        if len(c_df.dropna(subset=['target_rate'])) < 12: 
            logger.warning(f"Skipping {cid}: Insufficient history.")
            continue
            
        meta = c_df[['bu', 'client', 'groups']].iloc[-1].to_dict()
        
        # Validation Split: Train on history, Test on last 6 months
        train = c_df.dropna(subset=['target_rate', 'lag_12', 'TotalOrders']).iloc[:-6]
        valid = c_df.iloc[-6:]
        y_val = valid['target_rate'].values
        
        models_val = {}

        # --- MODEL 1: Weighted Lag Blend 1 ---
        models_val['WLB1'] = (0.6 * valid['lag_1']) + (0.2 * valid['lag_2']) + (0.2 * valid['lag_3'])

        # --- MODEL 2: Weighted Lag Blend 2 ---
        models_val['WLB2'] = (0.6 * valid['lag_1']) + (0.25 * valid['lag_3']) + (0.15 * valid['lag_6'])

        # --- MODEL 3: XGBoost ---
        feats = ['lag_1', 'lag_3', 'lag_6', 'lag_12', 'TotalOrders', 'MORTGAGE30US', 'UNRATE']
        
        if len(train) > 5:
            xgb_mod = XGBRegressor(n_estimators=50, learning_rate=0.05, max_depth=4, random_state=42)
            xgb_mod.fit(train[feats].fillna(0), train['target_rate'])
            models_val['XGB'] = xgb_mod.predict(valid[feats].fillna(0))
        else:
            models_val['XGB'] = models_val['WLB1']

        # --- MODEL 4: Ratio Model ---
        avg_ratio = train['calls_per_order'].replace([np.inf, -np.inf], 0).mean()
        if pd.isna(avg_ratio): avg_ratio = 0
        
        valid_orders = valid['TotalOrders'].ffill()
        models_val['RatioModel'] = (avg_ratio * valid_orders) / valid['business_day_count']

        # --- SELECTION: Pick Winner by SMAPE ---
        valid_scores = {k: smape(y_val, v) for k, v in models_val.items()}
        winner = min(valid_scores, key=valid_scores.get)
        
        audit_log.append({
            'client_id': cid, 
            'winner': winner, 
            'smape': valid_scores[winner], 
            'avg_ratio': avg_ratio
        })

        # --- Step 6: Recursive Forecasting (Back-cast + Future) ---
        full_range = pd.date_range(start=BACKCAST_START, periods=HORIZON + 12, freq='MS')
        
        driver_key = DRIVER_MAP.get(cid, cid)
        future_orders_lookup = df_orders[df_orders['client_id'] == driver_key].set_index('date')['TotalOrders'].to_dict()
        
        for d in full_range:
            d_ts = pd.Timestamp(d)
            
            # 1. Get Business Days
            biz_days = df_calls.loc[df_calls['date'] == d_ts, 'business_day_count']
            if not biz_days.empty:
                biz_days = biz_days.values[0]
            else:
                biz_days = np.busday_count(d_ts.date(), (d_ts + pd.offsets.MonthEnd(0)).date())
            
            # 2. Get Forecasted Driver (Orders)
            future_order_vol = future_orders_lookup.get(d_ts, 0)
            if future_order_vol == 0:
                future_order_vol = c_df['TotalOrders'].iloc[-1] 

            # 3. Calculate Prediction based on Winner
            pred_rate = 0.0
            
            if winner == 'RatioModel':
                pred_rate = (avg_ratio * future_order_vol) / biz_days
            elif winner == 'XGB':
                pred_rate = models_val['XGB'].mean()
            else:
                pred_rate = models_val[winner].mean()

            # 4. De-normalize to Total Volume
            final_vol = int(pred_rate * biz_days)

            forecast_results.append({
                'fx_date': d.date(),
                'bu': meta['bu'],
                'client': meta['client'],
                'client_id': cid,
                'groups': meta['groups'],
                'vol_type': 'phone',
                'fx_vol': final_vol,
                'fx_id': f"{winner}_v245",
                'fx_status': 'A',
                'load_ts': STAMP
            })

    # --- Step 7: Export Outputs ---
    if forecast_results:
        pd.DataFrame(forecast_results).to_csv(LOCAL_CSV, index=False)
        pd.DataFrame(audit_log).to_csv(AUDIT_CSV, index=False)
        df.to_csv(MODELING_DATA_CSV, index=False)
        
        with open(SUMMARIES_FILE, 'w') as f:
            f.write(f"Forecast Run: {STAMP}\n" + "="*40 + "\n")
            for entry in audit_log:
                f.write(f"Client: {entry['client_id']:<20} | Winner: {entry['winner']:<10} | SMAPE: {entry['smape']:.2f}%\n")

        logger.info(f"✓ Pipeline Complete. All files saved to C:\\WFM_Scripting\\")
    else:
        logger.warning("No forecasts were generated.")

except Exception as e:
    email_manager.handle_error("V2.4.5 Failure", e, is_test=True)


PS C:\WFM_Scripting\Forecasting> & C:/Scripting/Python_envs/venv_Master/Scripts/Activate.ps1
(venv_Master) PS C:\WFM_Scripting\Forecasting> & C:/Scripting/Python_envs/venv_Master/Scripts/python.exe c:/WFM_Scripting/Forecasting/Rpt_283_File.py
Traceback (most recent call last):
  File "c:\WFM_Scripting\Forecasting\Rpt_283_File.py", line 280, in <module>
    email_manager.handle_error("V2.4.5 Failure", e, is_test=True)
  File "C:\WFM_Scripting\Automation\scripthelper.py", line 1752, in handle_error
    raise exception
  File "c:\WFM_Scripting\Forecasting\Rpt_283_File.py", line 207, in <module>
    valid_scores = {k: smape(y_val, v) for k, v in models_val.items()}
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\WFM_Scripting\Forecasting\Rpt_283_File.py", line 207, in <dictcomp>
    valid_scores = {k: smape(y_val, v) for k, v in models_val.items()}
                       ^^^^^^^^^^^^^^^
  File "c:\WFM_Scripting\Forecasting\Rpt_283_File.py", line 69, in smape
    f = np.nan_to_num(np.array(f, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
                      ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\series.py", line 953, in __array__
    arr = np.asarray(values, dtype=dtype)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\arrays\masked.py", line 575, in __array__
    return self.to_numpy(dtype=dtype)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\pandas\core\arrays\masked.py", line 487, in to_numpy
    raise ValueError(
ValueError: cannot convert to 'float64'-dtype NumPy array with missing values. Specify an appropriate 'na_value' for this dtype.
