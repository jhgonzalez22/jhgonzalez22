"""
CX Credit - Call Volume Forecast - Production Hybrid v5.4
========================================================
PURPOSE
- Pulls Credit phone/chat historicals from BigQuery (via SQL file), merges with workload driver history + future FX.
- Runs a simple bake-off per (BU, Client, Group, VolumeType) combo.
- Exports forecast + audit files locally.

WHAT'S NEW IN v5.4 (CRITICAL FIX)
- pearsonr() dtype crash fix:
  * Enforced numeric float dtypes on df_wl_wide / df_wl_fut_wide AFTER pivot/unstack.
  * Enforced numeric float arrays immediately before pearsonr() (last line of defense).
  * Handles NaNs + constant series safely before correlation.

ASSUMPTIONS
- You have scripthelper.py at: C:\WFM_Scripting\Automation\scripthelper.py
- Your BigQuery SQL file exists at:
  C:\WFM_Scripting\Forecasting\GBQ - Credit Historicals Phone - Chat Volumes.sql
- Your BigQuery views/tables exist:
  * tax_clnt_svcs.view_cx_nontax_platforms_workload_drivers
  * tax_clnt_svcs.cx_nontax_platforms_workload_drivers_fx
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import pytz

# SciPy pearson correlation (we will guard inputs heavily)
from scipy.stats import pearsonr

# Simple regression model used for MLR workload model (single-feature)
from sklearn.linear_model import LinearRegression

# ------------------- PATHS & CONFIGURATION --------------------
# Ensure scripthelper can be imported
sys.path.append(r"C:\WFM_Scripting\Automation")

try:
    from scripthelper import Config, Logger, BigQueryManager, EmailManager
except ImportError:
    print("CRITICAL ERROR: scripthelper.py not found at C:\\WFM_Scripting\\Automation")
    sys.exit(1)

# Initialize config + core managers using your standard framework
config = Config(rpt_id=299)
logger = Logger(config)
bigquery_manager = BigQueryManager(config)
email_manager = EmailManager(config)

# Ignore noisy warnings for production runs (you can turn this off for debugging)
warnings.filterwarnings("ignore")

# ------------------- PARAMETERS --------------------
BACKCAST_MODE = False            # If True, use suffix in outputs (logic not expanded here)
FORECAST_HORIZON = 12            # Months ahead to forecast
VAL_LEN = 3                      # Validation length (months) for model selection bake-off
STAMP = datetime.now(pytz.timezone("America/Chicago"))

BASE_DIR = r"C:\WFM_Scripting"

# File Paths (historicals pulled via SQL file)
MAIN_DATA_SQL_PATH = fr"{BASE_DIR}\Forecasting\GBQ - Credit Historicals Phone - Chat Volumes.sql"

# Local Output Paths
suffix = "_BACKCAST" if BACKCAST_MODE else ""
LOCAL_CSV = fr"{BASE_DIR}\forecast_results_credit{suffix}.csv"
AUDIT_CSV = fr"{BASE_DIR}\model_eval_debug{suffix}.csv"
CORRELATION_CSV = fr"{BASE_DIR}\feature_correlations_credit{suffix}.csv"
MODELING_DATA_CSV = fr"{BASE_DIR}\modeling_data_credit{suffix}.csv"
STATS_CSV = fr"{BASE_DIR}\statistical_tests{suffix}.csv"
SUMMARIES_FILE = fr"{BASE_DIR}\model_summaries_credit{suffix}.txt"  # reserved for future text summaries

# Model ID Mapping (what you push downstream / store as IDs)
MODEL_IDS = {
    "SeasonalNaiveGR": "snaive_gr_workload",
    "Native3m": "native3m_workload",
    "MLR": "mlr_workload",
    "WLB_Fixed": "wlb_fix_workload",
    "Workload_Ratio": "ratio_workload",
    # Reserved / not implemented in this slim bakeoff
    "XGBoost": "xgb_workload",
    "Hybrid_WLB_MLR": "hybrid_wlb_mlr_workload",
}

# ------------------- BUSINESS LOGIC: LAUNCH DATES --------------------
# If a client/volumetype started later than the full history, cut training data accordingly.
# Keys are (client, 'All') or (client, 'Chat'/'Phone' etc. capitalized).
LAUNCH_DATES = {
    ('CrossCountry Mortgage', 'All'): '2025-07-01',
    ('General', 'Chat'): '2024-07-01',
    ('Prosperity', 'All'): '2025-01-01',
    ('Rapid Recheck', 'Chat'): '2025-01-01',
    ('Zillow', 'All'): '2025-10-01'
}

# ------------------- UTILITIES -------------------------
def safe_round(x) -> int:
    """
    Convert a value to a non-negative int safely.
    Used if you want integer rounding for output; kept here for convenience.
    """
    try:
        if x is None:
            return 0
        xf = float(x)
        if not np.isfinite(xf):
            return 0
        return int(max(0, round(xf)))
    except Exception:
        return 0


def smape(actual, forecast) -> float:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE), in %.
    - Handles zero denominators safely.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    denom = (np.abs(actual) + np.abs(forecast)) / 2.0
    denom[denom == 0] = 1.0  # avoid divide-by-zero

    return np.mean(np.abs(actual - forecast) / denom) * 100.0


def force_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    HARDENING HELPER:
    Convert all columns in a DataFrame to numeric floats (where possible).
    Anything non-numeric becomes NaN then filled with 0.0.

    This is critical when BigQuery returns Decimal / object types that later crash SciPy.
    """
    if df is None or df.empty:
        return df
    df2 = df.copy()
    for c in df2.columns:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    df2 = df2.fillna(0.0).astype(float)
    return df2


def safe_pearsonr(y: pd.Series, x: pd.Series):
    """
    Compute pearson correlation safely, returning (r, p) or (None, None) if invalid.

    Guards:
    - Force numeric to float
    - Drop NaNs
    - Require >=3 points
    - Skip constant vectors (std == 0)
    """
    # Force numeric conversion (LAST LINE OF DEFENSE)
    y_num = pd.to_numeric(y, errors="coerce")
    x_num = pd.to_numeric(x, errors="coerce")

    tmp = pd.concat([y_num, x_num], axis=1).dropna()
    if len(tmp) < 3:
        return None, None

    yv = tmp.iloc[:, 0].astype(float).to_numpy()
    xv = tmp.iloc[:, 1].astype(float).to_numpy()

    # pearson is undefined for constants
    if np.std(yv) == 0 or np.std(xv) == 0:
        return None, None

    r, p = pearsonr(yv, xv)
    return float(r), float(p)


# ------------------- CORE PIPELINE -------------------------
def run_pipeline():
    """
    Main end-to-end pipeline:
    1) Load call volume history (cpbd) from GBQ SQL file
    2) Load workload driver history + future FX
    3) For each (BU, Client, Group, VolumeType) combo:
       - build monthly ts
       - correlate drivers to pick best driver (optional)
       - bake-off models using last VAL_LEN months as validation
       - generate next FORECAST_HORIZON months forecast
    4) Export local CSV outputs
    """
    try:
        logger.info(f"Starting Credit Pipeline v5.4. Mode: {'BACKCAST' if BACKCAST_MODE else 'PROD'}")

        # ============================================================
        # 1) LOAD CALL DATA (HISTORICAL CPBD)
        # ============================================================
        df_calls = bigquery_manager.run_gbq_sql(MAIN_DATA_SQL_PATH, return_dataframe=True)
        if not isinstance(df_calls, pd.DataFrame):
            raise ValueError("BigQueryManager failed to return a DataFrame for MAIN_DATA_SQL_PATH. Check SQL file path or GBQ auth.")

        # Normalize column names to avoid casing issues
        df_calls.columns = [c.lower().strip() for c in df_calls.columns]

        # The historical SQL may name cpbd differently; normalize to "cpbd"
        vol_cols = ['cpbd', 'volume_per_business_day', 'callsoffered_per_business_day', 'total_calls_offered']
        found_vol_col = next((c for c in vol_cols if c in df_calls.columns), None)
        if found_vol_col and found_vol_col != 'cpbd':
            df_calls.rename(columns={found_vol_col: 'cpbd'}, inplace=True)

        # HARD FIX: force numeric on cpbd (prevents mixed object dtype later)
        if 'cpbd' not in df_calls.columns:
            raise ValueError("Expected a volume column (cpbd/volume_per_business_day/etc.) but none were found in df_calls.")

        df_calls['cpbd'] = pd.to_numeric(df_calls['cpbd'], errors='coerce').fillna(0.0).astype(float)

        # Ensure month is datetime (month-start preferred)
        if 'month' not in df_calls.columns:
            raise ValueError("Expected 'month' column in df_calls but it was missing. Check MAIN_DATA_SQL_PATH output.")
        df_calls['month'] = pd.to_datetime(df_calls['month'])

        # Validate expected dimensional columns exist
        required_dim_cols = ['bu', 'client', 'groups', 'volumetype']
        missing_dims = [c for c in required_dim_cols if c not in df_calls.columns]
        if missing_dims:
            raise ValueError(f"Missing required dimension columns in call data: {missing_dims}. Check SQL output.")

        # Collapsing groups into a controlled list to prevent overly granular modeling
        valid_groups = [
            'Main', 'General', 'Level', 'Rapid Recheck', 'Tax Transcripts', 'TS2', 'Billing', 'MtgGrp', 'Escalations'
        ]
        df_calls['groups_collapsed'] = df_calls['groups'].apply(
            lambda x: x if x in valid_groups else 'Other Credit Groups'
        )

        # ============================================================
        # 2) LOAD WORKLOAD DRIVERS (HISTORICAL + FUTURE FX)
        # ============================================================
        # Historical workload: month/product/orders
        wl_sql = """
        SELECT
          MonthOfOrder AS month,
          product,
          TotalOrders
        FROM tax_clnt_svcs.view_cx_nontax_platforms_workload_drivers
        WHERE client_id = 'Credit'
        """
        wl_hist_raw = bigquery_manager.run_gbq_sql(wl_sql, return_dataframe=True)

        # Create wide driver history: one row per month, one column per product
        df_wl_wide = pd.DataFrame()
        driver_features = []  # will hold base + lag feature names

        if isinstance(wl_hist_raw, pd.DataFrame) and not wl_hist_raw.empty:
            # Normalize + force month dtype
            wl_hist_raw.columns = [c.lower().strip() for c in wl_hist_raw.columns]
            wl_hist_raw['month'] = pd.to_datetime(wl_hist_raw['month'])

            # Force numeric on raw metric before pivot (but BigQuery can still return Decimal later)
            wl_hist_raw['totalorders'] = pd.to_numeric(wl_hist_raw['totalorders'], errors='coerce').fillna(0.0)

            # Pivot wide; groupby sum ensures unique (month, product) index pairs
            df_wl_wide = (
                wl_hist_raw
                .groupby(['month', 'product'])['totalorders']
                .sum()
                .unstack()
                .fillna(0.0)
            )
            df_wl_wide.index = pd.to_datetime(df_wl_wide.index)

            # *** CRITICAL HARD FIX ***
            # After unstack/pivot, df can still be object dtype (Decimals/mixed).
            # Force all driver columns to float now.
            df_wl_wide = force_numeric_df(df_wl_wide)

            # Create lag features for each product driver (L1 and L2)
            for col in df_wl_wide.columns:
                df_wl_wide[f"{col}_L1"] = df_wl_wide[col].shift(1)
                df_wl_wide[f"{col}_L2"] = df_wl_wide[col].shift(2)
                driver_features.extend([col, f"{col}_L1", f"{col}_L2"])

        # Future workload drivers (forecasted volumes)
        fut_wl_sql = """
        SELECT
          fx_date AS month,
          product,
          fx_vol
        FROM tax_clnt_svcs.cx_nontax_platforms_workload_drivers_fx
        WHERE client_id = 'Credit'
          AND fx_status = 'A'
        """
        wl_fut_raw = bigquery_manager.run_gbq_sql(fut_wl_sql, return_dataframe=True)

        df_wl_fut_wide = pd.DataFrame()
        if isinstance(wl_fut_raw, pd.DataFrame) and not wl_fut_raw.empty:
            wl_fut_raw.columns = [c.lower().strip() for c in wl_fut_raw.columns]
            wl_fut_raw['month'] = pd.to_datetime(wl_fut_raw['month'])

            wl_fut_raw['fx_vol'] = pd.to_numeric(wl_fut_raw['fx_vol'], errors='coerce').fillna(0.0)

            df_wl_fut_wide = (
                wl_fut_raw
                .groupby(['month', 'product'])['fx_vol']
                .sum()
                .unstack()
                .fillna(0.0)
            )
            df_wl_fut_wide.index = pd.to_datetime(df_wl_fut_wide.index)

            # *** CRITICAL HARD FIX ***
            df_wl_fut_wide = force_numeric_df(df_wl_fut_wide)

        # ============================================================
        # 3) BAKE-OFF ENGINE
        # ============================================================
        forecast_rows = []   # final forecast outputs
        eval_logs = []       # model winner + score per combo
        corr_logs = []       # feature correlations per combo

        # Build unique modeling combos
        combos = df_calls[['bu', 'client', 'groups_collapsed', 'volumetype']].drop_duplicates()

        for _, combo in combos.iterrows():
            bu = combo['bu']
            client = combo['client']
            group = combo['groups_collapsed']
            vtype = combo['volumetype']

            # Filter subset and ensure sorted monthly order
            subset = (
                df_calls[
                    (df_calls['client'] == client) &
                    (df_calls['groups_collapsed'] == group) &
                    (df_calls['volumetype'] == vtype)
                ]
                .sort_values('month')
            )

            # Apply launch date rule if configured
            start_key = LAUNCH_DATES.get((client, 'All')) or LAUNCH_DATES.get((client, str(vtype).capitalize()))
            if start_key:
                subset = subset[subset['month'] >= pd.to_datetime(start_key)]

            # Build monthly unique time series: groupby sum guarantees unique month index
            ts = subset.groupby('month')['cpbd'].sum().astype(float)

            # Skip if too short to validate + forecast reliably
            if len(ts) < max(6, VAL_LEN + 3):
                continue

            # If no workload drivers, we still run naïve models; workload models just won't be eligible
            merged = pd.DataFrame({'cpbd': ts})

            # Join with workload history on month (inner join ensures aligned months)
            if not df_wl_wide.empty:
                merged = pd.concat([merged, df_wl_wide], axis=1, join='inner')

            merged = merged.dropna()

            # ========================================================
            # 3A) DRIVER SELECTION (choose best correlated driver)
            # ========================================================
            best_driver = None
            max_corr = 0.0

            # Only attempt driver selection if we have driver features present
            if driver_features and len(merged) >= 5:
                for feat in driver_features:
                    if feat not in merged.columns:
                        continue

                    r, p = safe_pearsonr(merged['cpbd'], merged[feat])
                    if r is None or p is None:
                        continue

                    corr_logs.append({
                        'Client': client,
                        'Group': group,
                        'VolumeType': vtype,
                        'Feature': feat,
                        'Corr': r,
                        'Pval': p
                    })

                    # Keep strongest statistically significant correlation
                    if p < 0.05 and abs(r) > max_corr:
                        max_corr = abs(r)
                        best_driver = feat

            # ========================================================
            # 3B) TRAIN/VAL SPLIT
            # ========================================================
            train_ts = ts.iloc[:-VAL_LEN]
            val_actuals = ts.iloc[-VAL_LEN:]

            # Container for model predictions over historical index
            # (we will compute sMAPE on the last VAL_LEN months)
            m_preds = {}

            # ========================================================
            # MODEL 1: WLB Fixed (simple weighted lag blend)
            # - Uses lag-1, lag-3, lag-12 with fixed weights.
            # ========================================================
            l1 = train_ts.shift(1)
            l3 = train_ts.shift(3)
            l12 = train_ts.shift(12)

            # If lag12 is missing early, fallback to lag1
            wlb_series = (l1 * 0.6) + (l3 * 0.2) + (l12.fillna(l1) * 0.2)
            m_preds['WLB_Fixed'] = wlb_series

            # ========================================================
            # MODEL 2: SeasonalNaiveGR
            # - Uses last year's month (lag-12) scaled by recent growth rate.
            # - Growth rate uses (t / t-3) on training tail, guarded for division by zero.
            # ========================================================
            if len(train_ts) > 4 and train_ts.iloc[-4] != 0:
                gr = float(train_ts.iloc[-1] / train_ts.iloc[-4])
            else:
                gr = 1.0

            snaive_gr = train_ts.shift(12) * gr
            m_preds['SeasonalNaiveGR'] = snaive_gr

            # ========================================================
            # MODEL 3: Native3m
            # - 3-month rolling mean shifted by 1 month
            # ========================================================
            native3m = train_ts.rolling(3).mean().shift(1)
            m_preds['Native3m'] = native3m

            # ========================================================
            # MODEL 4/5: Workload models (Ratio and MLR)
            # - Only eligible if we found a strong driver.
            # - We require a strong absolute correlation threshold (ex: > 0.6)
            # ========================================================
            ratio_val = None
            reg = None

            if best_driver and max_corr > 0.6 and best_driver in merged.columns:
                # Align training months to merged frame
                m_train = merged.loc[train_ts.index].copy()

                # Ensure driver column is numeric float
                m_train['cpbd'] = pd.to_numeric(m_train['cpbd'], errors='coerce')
                m_train[best_driver] = pd.to_numeric(m_train[best_driver], errors='coerce')

                # ---- Workload Ratio ----
                # ratio = cpbd / driver; use last 6 valid months average
                denom = m_train[best_driver].replace(0, np.nan)
                ratio_series = (m_train['cpbd'] / denom).replace([np.inf, -np.inf], np.nan).dropna()

                if len(ratio_series) >= 3:
                    ratio_val = float(ratio_series.tail(6).mean())
                    m_preds['Workload_Ratio'] = merged[best_driver].astype(float) * ratio_val

                # ---- MLR (single-feature linear regression) ----
                # Fit: cpbd ~ driver
                # Only fit if we have enough non-null records
                fit_df = m_train[['cpbd', best_driver]].dropna()
                if len(fit_df) >= 4:
                    X = fit_df[[best_driver]].astype(float).values
                    y = fit_df['cpbd'].astype(float).values
                    reg = LinearRegression().fit(X, y)
                    m_preds['MLR'] = pd.Series(
                        reg.predict(merged[[best_driver]].astype(float).values),
                        index=merged.index
                    )

            # ========================================================
            # 3C) PICK WINNER BY sMAPE ON VALIDATION WINDOW
            # ========================================================
            scores = {}
            for model_name, preds in m_preds.items():
                # Compare only validation months; fill missing with 0 to keep scoring stable
                aligned = preds.reindex(val_actuals.index).fillna(0.0)
                scores[model_name] = smape(val_actuals.values, aligned.values)

            # Winner = lowest sMAPE
            winner = min(scores, key=scores.get)

            eval_logs.append({
                'Client': client,
                'Group': group,
                'VolumeType': vtype,
                'Winner': winner,
                'sMAPE': scores[winner],
                'Best_Driver': best_driver if best_driver else "None",
                'AbsCorr': max_corr
            })

            # ========================================================
            # 4) FORECAST GENERATION (next FORECAST_HORIZON months)
            # ========================================================
            # Build future month sequence starting next month after last historical month
            future_dates = pd.date_range(ts.index.max() + pd.DateOffset(months=1),
                                        periods=FORECAST_HORIZON, freq='MS')

            # Precompute fallback mean (for naïve non-workload forecast fallback)
            fallback_mean = float(ts.tail(3).mean())

            for d in future_dates:
                # Workload-based forecast uses future workload where possible
                if winner in ['Workload_Ratio', 'MLR'] and best_driver:
                    # Parse lag from feature name (e.g., "Zillow_L2" => base "Zillow" lag 2)
                    if "_L" in best_driver:
                        base_p = best_driver.split("_L")[0]
                        lag_v = int(best_driver.split("_L")[1])
                    else:
                        base_p = best_driver
                        lag_v = 0

                    # If best driver is lagged, the workload month needed is shifted back
                    tgt_d = d - pd.DateOffset(months=lag_v)

                    # Pull future workload if available; otherwise fallback to historical mean
                    if (not df_wl_fut_wide.empty) and (tgt_d in df_wl_fut_wide.index) and (base_p in df_wl_fut_wide.columns):
                        wl_val = float(df_wl_fut_wide.loc[tgt_d, base_p])
                    else:
                        # If we don't have future driver value, fallback to historical driver mean if present
                        wl_val = float(df_wl_wide[base_p].mean()) if (not df_wl_wide.empty and base_p in df_wl_wide.columns) else 0.0

                    # Compute prediction using ratio or regression
                    if winner == 'Workload_Ratio' and ratio_val is not None:
                        pred_v = wl_val * ratio_val
                    elif winner == 'MLR' and reg is not None:
                        pred_v = float(reg.predict(np.array([[wl_val]], dtype=float))[0])
                    else:
                        # If something wasn't fit properly, fallback
                        pred_v = fallback_mean
                else:
                    # For all other winners, use a simple stable fallback (mean of last 3 months)
                    # NOTE: your original code did this; keeping consistent.
                    pred_v = fallback_mean

                forecast_rows.append({
                    'Month': d,
                    'bu': bu,
                    'Client': client,
                    'Groups': group,
                    'VolumeType': vtype,
                    'Forecast_CPBD': round(max(0.0, float(pred_v)), 2),
                    'Model_ID': MODEL_IDS.get(winner, winner),
                    'Best_Driver': best_driver if best_driver else "None"
                })

        # ============================================================
        # 5) EXPORT LOCAL FILES
        # ============================================================
        # Forecast output
        pd.DataFrame(forecast_rows).to_csv(LOCAL_CSV, index=False)

        # Model evaluation summary
        pd.DataFrame(eval_logs).to_csv(AUDIT_CSV, index=False)

        # Correlation log output (driver screening)
        pd.DataFrame(corr_logs).to_csv(CORRELATION_CSV, index=False)

        # Save full call data used for modeling (raw-ish, after normalization)
        df_calls.to_csv(MODELING_DATA_CSV, index=False)

        # Save model ID reference (this file name kept for your existing workflow)
        pd.DataFrame([{'Model': k, 'ID': v} for k, v in MODEL_IDS.items()]).to_csv(STATS_CSV, index=False)

        logger.info(f"Pipeline Finished Successfully. Output files written under: {BASE_DIR}")

    except Exception as e:
        # Log the root failure
        logger.error(f"Critical Failure: {e}")

        # scripthelper EmailManager.handle_error raises the exception after sending notifications.
        # is_test=True prevents blasting prod DLs (per your standard usage).
        email_manager.handle_error("Credit Forecast Script Failure", e, is_test=True)


if __name__ == "__main__":
    run_pipeline()
