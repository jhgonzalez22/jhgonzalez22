"""
Credit Tickets - Daily Volume Forecast — v1.2 (DType Hotfix)

This script forecasts monthly ticket volumes for specific Credit business units.
This version includes a hotfix to prevent a data type error when training the OLS model.
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
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")  # keep logs tidy

# ------------------- helper-package path --------------------
sys.path.append(r"C:\WFM_Scripting\Automation")  # adjust if needed
from scripthelper import Config, Logger, BigQueryManager, EmailManager

# ------------------- initialise scripthelper ---------------
config           = Config(rpt_id=286)
logger           = Logger(config)
email_manager    = EmailManager(config)
bigquery_manager = BigQueryManager(config)

# ------------------- file / table paths --------------------
LOCAL_CSV        = r"C:\WFM_Scripting\credit_ticket_forecast_results.csv"
AUDIT_CSV        = r"C:\WFM_Scripting\credit_ticket_model_eval_debug.csv"
STATS_CSV        = r"C:\WFM_Scripting\credit_ticket_statistical_tests.csv"
XGB_VIZ_CSV      = r"C:\WFM_Scripting\credit_ticket_xgb_feature_importance.csv"
MODELING_DATA_CSV = r"C:\WFM_Scripting\credit_ticket_modeling_data.csv"
SUMMARIES_FILE   = r"C:\WFM_Scripting\credit_ticket_model_summaries.txt"

# ------------------- forecast parameters --------------------
FORECAST_HORIZON   = 15
MAX_LAGS           = 12
VAL_LEN_6M         = 6
STAMP              = datetime.now(pytz.timezone("America/Chicago"))

FX_ID_PREFIX = {
    "WeightedLagBlend": "wlb_ticket_custom",
    "OLS_DriverAndLag1": "ols_drvlag1_ticket"
}

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

# ------------------- data fetching & processing -------------------------
def fetch_credit_workload_drivers(bq: BigQueryManager) -> pd.DataFrame:
    sql = "SELECT MonthOfOrder, TotalOrders FROM tax_clnt_svcs.view_cx_nontax_platforms_workload_drivers WHERE client_id = 'Credit'"
    df = bq.run_gbq_sql(sql, return_dataframe=True)
    if df.empty: return pd.DataFrame(columns=["date", "TotalOrders"])
    df = df.rename(columns={'MonthOfOrder': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    return df.groupby('date', as_index=False)['TotalOrders'].sum()

def fetch_credit_driver_forecasts(bq: BigQueryManager) -> pd.DataFrame:
    sql = "SELECT fx_date, fx_vol AS TotalOrders_fx FROM tax_clnt_svcs.cx_nontax_platforms_workload_drivers_fx WHERE fx_status = 'A' AND client_id = 'Credit'"
    df = bq.run_gbq_sql(sql, return_dataframe=True)
    if df.empty: return pd.DataFrame()
    df['date'] = pd.to_datetime(df['fx_date'])
    return df.drop(columns=['fx_date'])

# ------------------- MAIN EXECUTION -------------------------
try:
    logger.info("Starting Credit Ticket Forecasting Script (v1.2)...")
    
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
    GROUP BY 1, 2, 3 ORDER BY 2, 1;
    """
    
    df = bigquery_manager.run_gbq_sql(TICKET_QUERY, return_dataframe=True)
    if df.empty: raise RuntimeError("BQ returned zero rows for tickets")
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"✓ Pulled {len(df):,} ticket rows")

    drv_df = fetch_credit_workload_drivers(bigquery_manager)
    drv_fx_df = fetch_credit_driver_forecasts(bigquery_manager)
    if not drv_df.empty:
        df = pd.merge(df, drv_df, on='date', how='left')
    else:
        df['TotalOrders'] = np.nan

    df = df.sort_values(['client_id', 'date'])
    for lag in range(1, MAX_LAGS + 1):
        df[f'lag_{lag}'] = df.groupby(['client_id'])['total_cases_opened'].shift(lag)

    forecasts, audit_rows = [], []

    for cid, group_df_full in df.groupby('client_id'):
        logger.info(f"\n{'='*30}\nProcessing: Client = {cid}\n{'='*30}")
        
        client_df = group_df_full.set_index('date').sort_index()
        ts = client_df['total_cases_opened']

        best_model = None
        model_params = {}

        if cid == 'Credit - Customer Support':
            logger.info("Applying WeightedLagBlend model for Customer Support.")
            best_model = 'WeightedLagBlend'
            
            feat = pd.DataFrame({'y': ts})
            for lag in [1, 2, 3]: feat[f'lag_{lag}'] = feat['y'].shift(lag)
            feat.dropna(inplace=True)
            
            y_v, l1, l2, l3 = feat['y'], feat['lag_1'], feat['lag_2'], feat['lag_3']
            best_weights, best_s = (1.0, 0.0, 0.0), np.inf
            for w1 in np.arange(0, 1.05, 0.05):
                for w2 in np.arange(0, 1.05-w1, 0.05):
                    w3 = 1 - w1 - w2
                    s = smape(y_v, w1*l1 + w2*l2 + w3*l3)
                    if s < best_s: best_weights, best_s = (w1, w2, w3), s
            
            model_params['weights'] = best_weights
            audit_rows.append({'client_id':cid, 'model':best_model, 'winner_model':best_model, 'w1':best_weights[0], 'w2':best_weights[1], 'w3':best_weights[2]})

        elif cid == 'Credit - Tech Support':
            logger.info("Applying OLS model with TotalOrders and lag_1 for Tech Support.")
            best_model = 'OLS_DriverAndLag1'
            
            feat = pd.DataFrame({'y': ts, 'TotalOrders': client_df['TotalOrders'], 'lag_1': client_df['lag_1']})
            feat.dropna(inplace=True)
            
            y_train_log = np.log1p(feat['y'])
            # --- FIX: Explicitly cast training data to float ---
            X_train = sm.add_constant(feat[['TotalOrders', 'lag_1']].astype(float))
            
            model = sm.OLS(y_train_log, X_train).fit()
            model_params['model'] = model
            audit_rows.append({'client_id':cid, 'model':best_model, 'winner_model':best_model, 'r_squared':model.rsquared})

        if not best_model:
            logger.warning(f"No specific model assigned for {cid}. Skipping.")
            continue

        # --- Recursive Forecasting ---
        future_idx = pd.date_range(ts.index[-1], periods=FORECAST_HORIZON+1, freq='MS')[1:]
        hist = deque(ts.tolist(), maxlen=24)
        drv_hist = deque(client_df['TotalOrders'].dropna().tolist(), maxlen=24)
        
        last_total_orders = client_df['TotalOrders'].dropna().iloc[-1] if not client_df['TotalOrders'].dropna().empty else 0
        driver_fx_lookup = {pd.Timestamp(d):v for d,v in drv_fx_df[['date','TotalOrders_fx']].values} if not drv_fx_df.empty else {}
        if driver_fx_lookup: logger.info(f"· Forecast will use forecasted drivers.")
        
        for d in future_idx:
            pred = np.nan
            current_total_orders = driver_fx_lookup.get(d, last_total_orders)
            
            if best_model == 'WeightedLagBlend':
                w1, w2, w3 = model_params['weights']
                y1 = hist[-1] if len(hist) >= 1 else 0
                y2 = hist[-2] if len(hist) >= 2 else y1
                y3 = hist[-3] if len(hist) >= 3 else y2
                pred = w1*y1 + w2*y2 + w3*y3
            
            elif best_model == 'OLS_DriverAndLag1':
                y1 = hist[-1] if len(hist) >= 1 else 0
                x_row = pd.DataFrame([{'const': 1, 'TotalOrders': current_total_orders, 'lag_1': y1}])
                raw_pred = model_params['model'].predict(x_row)[0]
                pred = np.expm1(raw_pred)
            
            hist.append(pred)
            drv_hist.append(current_total_orders)
            forecasts.append({'fx_date':d,'client_id':cid,'vol_type':'tickets','fx_vol':safe_round(pred),'fx_id':f"{FX_ID_PREFIX[best_model]}_{STAMP:%Y%m%d}",'fx_status':"A",'load_ts':STAMP})

    if forecasts:
        fx_df = pd.DataFrame(forecasts).sort_values(['client_id','fx_date'])
        fx_df.to_csv(LOCAL_CSV, index=False)
        pd.DataFrame(audit_rows).sort_values(['client_id','model']).to_csv(AUDIT_CSV, index=False)
        logger.info(f"✓ Forecast files saved locally to:\n    {LOCAL_CSV}\n    {AUDIT_CSV}")
        logger.info("✓ Credit ticket forecasting completed successfully (local only).")
    else:
        logger.warning("No forecasts were generated.")

except Exception as exc:
    email_manager.handle_error("Credit Ticket Forecasting Script Failure", exc, is_test=True)


(venv_Master) PS C:\WFM_Scripting\Automation> & C:/Scripting/Python_envs/venv_Master/Scripts/python.exe c:/WFM_Scripting/Automation/testing2.py
Traceback (most recent call last):
  File "c:\WFM_Scripting\Automation\testing2.py", line 221, in <module>
    email_manager.handle_error("Credit Ticket Forecasting Script Failure", exc, is_test=True)
  File "c:\WFM_Scripting\Automation\scripthelper.py", line 1319, in handle_error
    raise exception
  File "c:\WFM_Scripting\Automation\testing2.py", line 173, in <module>
    model = sm.OLS(y_train_log, X_train).fit()
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\statsmodels\regression\linear_model.py", line 921, in __init__
    super().__init__(endog, exog, missing=missing,
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\statsmodels\regression\linear_model.py", line 746, in __init__
    super().__init__(endog, exog, missing=missing,
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\statsmodels\regression\linear_model.py", line 200, in __init__
    super().__init__(endog, exog, **kwargs)
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\statsmodels\base\model.py", line 270, in __init__
    super().__init__(endog, exog, **kwargs)
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\statsmodels\base\model.py", line 95, in __init__
    self.data = self._handle_data(endog, exog, missing, hasconst,
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\statsmodels\base\model.py", line 135, in _handle_data
    data = handle_data(endog, exog, missing, hasconst, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\statsmodels\base\data.py", line 675, in handle_data
    return klass(endog, exog=exog, missing=missing, hasconst=hasconst,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\statsmodels\base\data.py", line 84, in __init__
    self.endog, self.exog = self._convert_endog_exog(endog, exog)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Scripting\Python_envs\venv_Master\Lib\site-packages\statsmodels\base\data.py", line 509, in _convert_endog_exog
    raise ValueError("Pandas data cast to numpy dtype of object. "
ValueError: Pandas data cast to numpy dtype of object. Check input data with np.asarray(data).
