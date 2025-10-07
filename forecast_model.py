"""
CX Credit - Call Volume Forecast - Production Hybrid v3.0

WHAT THIS SCRIPT DOES
---------------------
Generates 12-month forecasts for monthly volumes by BU for two channels:
  • PHONE → model bake-off (CWLB vs XGB vs Native3mGR), choose best by 6m SMAPE
  • CHAT  → simple Weighted Lag Blend using lags {1, 3}, weights learned by 3m SMAPE

It then writes the winning forecasts to BigQuery with:
  • vol_type ∈ {'phone', 'chat'} only (no suffix in the table)
  • exactly ONE row per (client_id, vol_type, fx_date) in this run
  • an UPDATE pass that inactivates any older rows for the same BU + month + channel
    (works even against historical vol_types like 'phone_xgboost', since we collapse
     to the base channel via REGEXP_EXTRACT).

WHY THESE MODELS
----------------
• CWLB (Constrained Weighted Lag Blend, PHONE):
  - A convex combination of target lags (1, 3, 6, 12).
  - Constraints encourage seasonal memory (weight floor across 3,6,12) and cap
    short-memory dominance (lag_1 max).
  - Advantage: low-variance, interpretable, fast.

• XGB (XGBoost w/ features, PHONE):
  - Targets log space; predicts log(y+1), then back-transforms safely.
  - Features: trend, seasonality (sin/cos month), short/long lags (1,2,3,6,12),
              and FRED lagged economic features (3,6,12).
  - Adds short-term memory (lag_1, lag_2) to avoid "straight-line" forecasts.
  - Advantage: can pick up non-linear effects and richer dynamics.

• Native3mGR (PHONE):
  - Seasonal naive using y[t-12] scaled by average 3-month year-over-year factor
    estimated from the last 3 most recent months.
  - Advantage: robust, seasonally anchored fallback.

• WLB(1,3) (CHAT ONLY):
  - A convex combination of lag_1 and lag_3, weights learned on the latest 3 months
    to minimize SMAPE; then held fixed and used recursively.
  - Advantage: simple, responsive to recent level changes, avoids overfitting.

KEY SAFETY / ROBUSTNESS
-----------------------
• Numeric coercion for stats and modeling inputs
• SMAPE utility is NaN-safe
• JSON-safe serialization (NumPy → native)
• FX de-duplication in-memory + GBQ UPDATE to inactivate older rows per BU×month×channel
• Fixed issue: using .dt.date once, then min/max on dates (no .date() on date)

REQUIRED LOCAL DEPENDENCIES
---------------------------
Relies on your helper package at C:\WFM_Scripting\Automation:
  - scripthelper.Config / Logger / BigQueryManager / EmailManager
Also expects your GBQ SQL path for source data and FRED to exist.

OUTPUTS
-------
Local CSVs: forecast results, model evals, stats, XGB importances, modeling data.
BigQuery:   appends the forecast rows, then inactivates older active rows.

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
VAL_LEN_PHONE_6M   = 6   # bakeoff horizon for PHONE
VAL_LEN_CHAT_3M    = 3   # weight-learning horizon for CHAT
STAMP              = datetime.now(pytz.timezone("America/Chicago"))
GRID_STEP          = 0.05
ECONOMIC_FEATURES  = ['UNRATE', 'HSN1F', 'FEDFUNDS', 'MORTGAGE30US']

# ------------------- utilities -------------------------
def to_native(obj):
    """Convert NumPy scalars/lists/dicts to native Python for JSON serialization."""
    if isinstance(obj, dict):   return {to_native(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_native(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)): return obj.item()
    if isinstance(obj, (np.bool_,)): return bool(obj)
    return obj

def smape(actual, forecast):
    """Symmetric MAPE (%). Safe for zeros (denominator floored to 1)."""
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    denom = (np.abs(actual) + np.abs(forecast)) / 2.0
    denom[denom == 0] = 1.0
    return np.mean(np.abs(actual - forecast) / denom) * 100

def safe_round(x):
    """Round to nearest non-negative int; treat non-finite as 0."""
    x = 0.0 if (x is None or not np.isfinite(x)) else x
    return int(max(0, round(x)))

def safe_expm1(x, lo=-20.0, hi=20.0):
    """Back-transform from log1p with clipping to avoid blow-ups."""
    arr = np.asarray(x, dtype=float)
    return np.expm1(np.clip(arr, lo, hi))

def find_best_weights(ts: pd.Series, lags: list, constraints: dict, val_window_len: int, step: float):
    """
    Grid-search convex weights over given lag set to minimize SMAPE on the last val_window_len points.
    constraints:
      - per-lag: {lag: {'min':..., 'max':...}}
      - 'combined': [{'lags':[...], 'min':...}, ...]
    """
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
                if 'max' in limits and weights[lag] > limits['max'] + 1e-9: valid = False; break
                if 'min' in limits and weights[lag] < limits['min'] - 1e-9: valid = False; break
        if not valid:
            continue

        # Combined constraints
        if 'combined' in constraints:
            for combo in constraints['combined']:
                combo_sum = sum(weights.get(l, 0.0) for l in combo['lags'])
                if 'min' in combo and combo_sum < combo['min'] - 1e-9: valid = False; break
            if not valid:
                continue

        # Validation forecast from weighted shifted series
        forecast_vals = 0.0
        for lag, w in weights.items():
            forecast_vals += ts.shift(lag).reindex(validation_data.index) * w

        mask = forecast_vals.notna() & validation_data.notna()
        if not mask.any():
            continue

        current_smape = smape(validation_data[mask], forecast_vals[mask])
        if current_smape < best_smape - 0.1:  # tie-break stability
            best_smape, best_weights, best_preds = current_smape, weights, forecast_vals

    return best_weights, best_smape, best_preds

def perform_statistical_tests(bu_data: pd.DataFrame, bu_name: str, features: list):
    """
    Basic stationarity & correlation diagnostics (for audit only).
    ADF on target (numeric coerced); Pearson correlations for provided features.
    """
    results = {'bu': bu_name}
    target_col = 'TotalCallsOffered'

    # ADF (if enough data)
    adf_series = pd.to_numeric(bu_data[target_col], errors='coerce').dropna()
    if len(adf_series) > 12:
        try:
            adf_p = adfuller(adf_series)[1]
            results['adf_p_value'] = float(adf_p)
            results['is_stationary'] = bool(adf_p < 0.05)
        except Exception:
            results['adf_p_value'] = None
            results['is_stationary'] = None

    # Pearson correlations (numeric-only, guard against zero variance / non-finite)
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

# -------- PHONE helper: "Native3mGR" seasonal naive w/ growth factor --------
def compute_3m_yoy_factor(series: pd.Series) -> float:
    """
    Estimate mean YoY factor using last k∈{1,2,3} months vs same month last year.
    Fallback to 1.0 if insufficient data or unstable ratios.
    """
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
    """
    Predict y_hat[t] = factor * y[t-12], where factor = mean 3m YoY ratio (above).
    """
    factor = compute_3m_yoy_factor(train_series)
    preds = []
    s = train_series.copy()
    for d in pred_index:
        t12 = s.get(d - pd.DateOffset(months=12), np.nan)
        preds.append((t12 if np.isfinite(t12) else np.nan) * factor if np.isfinite(factor) else np.nan)
    return pd.Series(preds, index=pred_index, dtype='float')

# ------------------- MAIN EXECUTION -------------------------
try:
    logger.info("Starting Credit Forecasting v3.0 (Phone + Chat)...")

    # 1) Fetch & Prepare Data ---------------------------------------------------
    source_df = bigquery_manager.run_gbq_sql(SQL_QUERY_PATH, return_dataframe=True)
    source_df = source_df.rename(columns={'Month': 'date'})
    source_df['date'] = pd.to_datetime(source_df['date'])
    source_df['TotalCallsOffered'] = pd.to_numeric(source_df['TotalCallsOffered'], errors='coerce')

    # Keep only required columns; enforce canonical order
    df = (source_df[['date', 'bu', 'VolumeType', 'TotalCallsOffered']]
            .drop_duplicates()
            .sort_values(['bu', 'date'])
          )

    # 2) Economic Data (FRED) for PHONE/XGB ------------------------------------
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

    # 3) Feature engineering (seasonality/trend + target lags) ------------------
    df['month']     = df['date'].dt.month
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['trend']     = df.groupby(['VolumeType','bu']).cumcount() + 1

    def add_target_lags(frame, voltype, lags):
        """Add lag columns for the target within each (voltype, bu) series."""
        m = frame['VolumeType'] == voltype
        for L in lags:
            frame.loc[m, f'lag_{L}'] = frame[m].groupby('bu')['TotalCallsOffered'].shift(L)

    # PHONE needs short+long lags (XGB), CWLB uses 1/3/6/12
    add_target_lags(df, 'Phone', [1,2,3,6,12])
    # CHAT uses only lag_1 & lag_3 (by design)
    add_target_lags(df, 'Chat',  [1,3])

    # 4) Output holders ---------------------------------------------------------
    forecasts, audit_rows, all_modeling_data, stat_test_results, xgb_imp_rows = [], [], [], [], []

    # ======================== PHONE: BAKE-OFF ==================================
    logger.info("\n" + "="*60 + "\nProcessing Channel: Phone\n" + "="*60)
    phone_df = df[df['VolumeType'] == 'Phone'].copy()

    # Require at least lag_12 (and econ lag_12 if present) to avoid sparse ends
    req_cols = ['lag_12']
    if lagged_econ_cols:
        req_cols += [c for c in lagged_econ_cols if c.endswith('_lag_12')]
    phone_df.dropna(subset=[c for c in req_cols if c in phone_df.columns], inplace=True)

    for bu_name in phone_df['bu'].unique():
        logger.info(f"\n--- Phone :: BU: {bu_name} ---")
        bu = (phone_df[phone_df['bu'] == bu_name]
                .set_index('date')
                .sort_index()
                .copy()
             )
        bu['TotalCallsOffered'] = pd.to_numeric(bu['TotalCallsOffered'], errors='coerce')
        bu_ts = bu['TotalCallsOffered'].asfreq('MS')  # ensure monthly start freq
        all_modeling_data.append(bu.copy())

        # XGB features — include lag_1 and lag_2 to preserve short-term volatility
        xgb_feats = ['trend','sin_month','cos_month','lag_1','lag_2','lag_3','lag_6','lag_12'] + \
                    [c for c in [f'{x}_lag_{l}' for x in ECONOMIC_FEATURES for l in [3,6,12]] if c in bu.columns]

        # Light stats for audit
        stat_test_results.append(perform_statistical_tests(bu, f"Phone:{bu_name}", xgb_feats))

        # Train/validation split for bake-off (6 months)
        if len(bu) <= VAL_LEN_PHONE_6M:
            logger.warning(f"  Not enough data for 6m validation: Phone:{bu_name}; skipping.")
            continue
        train, valid = bu.iloc[:-VAL_LEN_PHONE_6M], bu.iloc[-VAL_LEN_PHONE_6M:]
        y_val = valid['TotalCallsOffered']

        # --- CWLB (1,3,6,12) with constraints favoring seasonal weight ----
        cwlb_weights, cwlb_smape, _ = find_best_weights(
            bu_ts, [1,3,6,12],
            constraints={1:{'max':0.5}, 3:{'min':0.2}, 'combined':[{'lags':[3,6,12], 'min':0.5}]},
            val_window_len=VAL_LEN_PHONE_6M,
            step=GRID_STEP
        )
        logger.info(f"  -> CWLB SMAPE={f'{cwlb_smape:.2f}' if np.isfinite(cwlb_smape) else 'N/A'}")

        # --- XGB (log-target) ---------------------------------------------------
        Xtr, Xv = train[xgb_feats], valid[xgb_feats]
        ytr = train['TotalCallsOffered']

        xgb = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )
        xgb.fit(Xtr, np.log1p(ytr))
        xgb_preds = safe_expm1(xgb.predict(Xv))
        xgb_smape = smape(y_val, xgb_preds)
        logger.info(f"  -> XGB  SMAPE={xgb_smape:.2f}")

        imp = {str(f): float(s) for f, s in zip(xgb.feature_names_in_, xgb.feature_importances_)}
        xgb_imp_rows.append({'bu': f"Phone:{bu_name}", 'feature_importance': json.dumps(to_native(imp))})

        # --- Native3mGR ---------------------------------------------------------
        native_val_preds = seasonal_naive_3m_gr_predict(train['TotalCallsOffered'], valid.index)
        native_smape = smape(y_val, native_val_preds)
        logger.info(f"  -> Native3mGR SMAPE={native_smape:.2f}")

        # --- Winner selection ----------------------------------------------------
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

        # --- Produce 12m recursive forecast with the winner ---------------------
        if np.isfinite(winner_smape):
            hist = deque(bu_ts.tolist(), maxlen=24)
            future_idx = pd.date_range(bu_ts.index[-1], periods=FORECAST_HORIZON + 1, freq='MS')[1:]

            long_weights = None
            if winner == 'CWLB':
                # Long-horizon CWLB re-tuned to (3,6,12) only
                long_weights, _, _ = find_best_weights(
                    bu_ts, [3,6,12],
                    constraints={3:{'min':0.4}, 6:{'min':0.2}, 12:{'min':0.1}},
                    val_window_len=12,
                    step=GRID_STEP
                )

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
                    # Carry short-term volatility via lag_1/lag_2 in the future rows
                    row = {
                        'trend': (train['trend'].max() or 0) + len(valid) + i,
                        'sin_month': np.sin(2*np.pi*d.month/12),
                        'cos_month': np.cos(2*np.pi*d.month/12),
                        'lag_1':  hist[-1] if len(hist) >= 1 else np.nan,
                        'lag_2':  hist[-2] if len(hist) >= 2 else (hist[-1] if len(hist) else np.nan),
                        'lag_3':  hist[-3] if len(hist) >= 3 else (hist[-1] if len(hist) else np.nan),
                        'lag_6':  hist[-6] if len(hist) >= 6 else (hist[-1] if len(hist) else np.nan),
                        'lag_12': hist[-12] if len(hist) >= 12 else (hist[-1] if len(hist) else np.nan),
                    }
                    # Reuse last known FRED lag features (remain constant in horizon)
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

    # ======================== CHAT: WLB(1,3) ================================
    logger.info("\n" + "="*60 + "\nProcessing Channel: Chat (WLB lag1,lag3; 3m SMAPE)\n" + "="*60)
    chat_df = df[df['VolumeType'] == 'Chat'].copy()

    for bu_name in chat_df['bu'].unique():
        logger.info(f"\n--- Chat :: BU: {bu_name} ---")
        bu = (chat_df[chat_df['bu'] == bu_name]
                .set_index('date')
                .sort_index()
                .copy()
             )
        bu['TotalCallsOffered'] = pd.to_numeric(bu['TotalCallsOffered'], errors='coerce')
        bu_ts = bu['TotalCallsOffered'].asfreq('MS')
        all_modeling_data.append(bu.copy())

        # Minimal stats for audit
        stat_test_results.append(perform_statistical_tests(bu, f"Chat:{bu_name}", [c for c in ['lag_1','lag_3'] if c in bu.columns]))

        if len(bu) <= VAL_LEN_CHAT_3M:
            logger.warning(f"  Not enough data for 3m validation: Chat:{bu_name}; skipping.")
            continue

        # Learn convex weights on {1,3} over last 3 months (minimize SMAPE)
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

        # Fixed learned weights, then recursive forecast
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

    # 5) Save local artifacts ----------------------------------------------------
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

    # 6) Persist forecasts locally + PUSH to GBQ w/ strict uniqueness ----------
    if forecasts:
        fx_df = pd.DataFrame(forecasts)

        # Enforce one row per (client_id, vol_type, fx_date) in this batch
        fx_df = (fx_df
                 .sort_values(['client_id','vol_type','fx_date','load_ts'])
                 .drop_duplicates(subset=['client_id','vol_type','fx_date'], keep='last')
                 .sort_values(['client_id', 'vol_type', 'fx_date'])
        )

        # Save locally
        fx_df.to_csv(FORECAST_RESULTS_CSV, index=False)
        logger.info(f"✓ Forecast data saved locally to {FORECAST_RESULTS_CSV}")

        # --- GBQ PUSH ---
        logger.info(f"Pushing {len(fx_df):,} forecast rows to {DEST_TABLE}...")
        bigquery_manager.import_data_to_bigquery(
            fx_df, DEST_TABLE, gbq_insert_action="append", auto_convert_df=True
        )

        # Deactivate older rows for the same BU + month + channel
        # Normalize fx_date once to pure date; then min/max (no .date() on date)
        fx_df['fx_date'] = pd.to_datetime(fx_df['fx_date']).dt.date
        client_list = fx_df['client_id'].dropna().unique().tolist()
        date_min = min(fx_df['fx_date'])
        date_max = max(fx_df['fx_date'])

        if client_list:
            client_list_str = ", ".join([f"'{c}'" for c in client_list])
            dedup_sql = f"""
            UPDATE `{DEST_TABLE}` t
            SET t.fx_status = 'I'
            WHERE t.fx_status = 'A'
              AND t.client_id IN ({client_list_str})
              AND t.fx_date BETWEEN DATE('{date_min}') AND DATE('{date_max}')
              AND EXISTS (
                SELECT 1
                FROM `{DEST_TABLE}` s
                WHERE s.client_id = t.client_id
                  AND s.fx_date = t.fx_date
                  -- same channel (phone/chat) regardless of historical suffixes
                  AND REGEXP_EXTRACT(s.vol_type, r'^(phone|chat)') = REGEXP_EXTRACT(t.vol_type, r'^(phone|chat)')
                  AND s.load_ts > t.load_ts
              )
            """
            bigquery_manager.run_gbq_sql(dedup_sql, return_dataframe=False)
            bigquery_manager.update_log_in_bigquery()
            logger.info("✓ Forecasts pushed to GBQ and older active rows deactivated (per BU-month-channel).")
        else:
            logger.warning("Client list empty after forecast build; skipped dedup SQL.")

    else:
        logger.warning("No forecasts were generated; nothing to push.")

except Exception as exc:
    email_manager.handle_error("Credit Forecast Script Failure", exc, is_test=True)
