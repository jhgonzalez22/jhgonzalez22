# ───────────────── workload_forecast_multi_model.py ─────────────────
"""
CX Platforms – Workload Drivers Forecast (TotalOrders)

Pipeline:
  1) Query monthly client time series (TotalOrders) from GBQ.
  2) Build features: lags 3/6/12 only (to avoid overfitting).
  3) Benchmark 4 models on the last 6 months with SMAPE:
       A) SeasonalNaive
       B) SeasonalNaiveGR
       C) SeasonalNaive3mDiffGR
       D) WeightedLagBlend
  4) Select best model per client (lowest SMAPE).
  5) Produce recursive 15-month forecast with the winner.
  6) Push results to GBQ, inactivate older rows, save CSVs, log.

Why only these 4?
  • OLS removed – recursive forecasts can blow up after a few months.
  • ARIMA/HoltWinters removed – too heavy, not stable enough cross-clients.
"""

# ------------------- standard imports -----------------------
import os, sys, warnings, pytz
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ------------------- helper-package path --------------------
sys.path.append(r'C:\WFM_Scripting\Automation')
from scripthelper import Config, Logger, BigQueryManager, EmailManager

# ------------------- initialise scripthelper ---------------
config            = Config(rpt_id=288)
logger            = Logger(config)
email_manager     = EmailManager(config)
bigquery_manager  = BigQueryManager(config)

# ------------------- file / table paths --------------------
GBQ_VIEW_QUERY = """
SELECT
  MonthOfOrder AS date,
  client_id,
  TotalOrders  AS target_volume
FROM `tax_clnt_svcs.view_cx_nontax_platforms_workload_drivers`
WHERE MonthOfOrder < DATE_TRUNC(CURRENT_DATE(), MONTH)
"""
DEST_TABLE = "tax_clnt_svcs.cx_nontax_platforms_workload_drivers_fx"

LOCAL_CSV = r"C:\WFM_Scripting\cx_nontax_workload_drivers_fx_results.csv"
AUDIT_CSV = r"C:\WFM_Scripting\cx_nontax_workload_drivers_fx_model_eval.csv"

# ------------------- parameters -----------------------------
FORECAST_HORIZON = 15
TEST_LEN         = 6
LAGS             = (3, 6, 12)
GRID_STEP        = 0.05
MIN_W12          = 0.0

STAMP_TZ = pytz.timezone("America/Chicago")
STAMP    = datetime.now(STAMP_TZ)

FX_ID_PREFIX = {
    "SeasonalNaive":           "snaive_workload",
    "SeasonalNaiveGR":         "snaive_gr_workload",
    "SeasonalNaive3mDiffGR":   "snaive_diffgr_workload",
    "WeightedLagBlend":        "wblend_workload",
}

# ------------------- metrics -------------------------------
def smape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    denom = (np.abs(a) + np.abs(f)) / 2.0
    out = np.where(denom == 0.0, 0.0, np.abs(a - f) / denom)
    return float(np.mean(out) * 100.0)

def mape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    return float(np.mean(np.abs((a - f) / np.where(a == 0, 1, a))) * 100.0)

def rmse(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    return float(np.sqrt(np.mean((a - f) ** 2)))

# ------------------- helpers: lags + growth -----------------
def add_lags(ts: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": ts})
    for L in LAGS:
        df[f"lag_{L}"] = df["y"].shift(L)
    return df

def recent_3m_growth(ts: pd.Series, pos: int) -> float:
    if pos < 6: return 0.0
    cur3  = ts.iloc[pos-3:pos].values
    prev3 = ts.iloc[pos-6:pos-3].values
    prev_mean = np.mean(prev3)
    return (np.mean(cur3) / prev_mean - 1.0) if prev_mean > 0 else 0.0

def prior_year_3m_growth(ts: pd.Series, pos: int) -> float:
    if pos < 18: return 0.0
    win_py  = ts.iloc[pos-15:pos-12].values
    prev_py = ts.iloc[pos-18:pos-15].values
    prev_mean = np.mean(prev_py)
    return (np.mean(win_py) / prev_mean - 1.0) if prev_mean > 0 else 0.0

# ------------------- models (validation) -------------------
def seasonal_naive_series(ts, idx):
    return ts.shift(12).reindex(idx)

def seasonal_naive_gr_series(ts, idx):
    out, s12 = [], ts.shift(12)
    for d in idx:
        base = s12.get(d, np.nan)
        if not np.isfinite(base) or d not in ts.index:
            out.append(np.nan); continue
        pos = ts.index.get_loc(d)
        r   = recent_3m_growth(ts, pos)
        out.append(max(0.0, base * (1.0 + r)))
    return pd.Series(out, index=idx, dtype=float)

def seasonal_naive_3m_diff_gr_series(ts, idx):
    out, s12 = [], ts.shift(12)
    for d in idx:
        base = s12.get(d, np.nan)
        if not np.isfinite(base) or d not in ts.index:
            out.append(np.nan); continue
        pos = ts.index.get_loc(d)
        r  = recent_3m_growth(ts, pos)
        rp = prior_year_3m_growth(ts, pos)
        out.append(max(0.0, base * (1.0 + (r - rp))))
    return pd.Series(out, index=idx, dtype=float)

def weight_grid(step=GRID_STEP, min_w12=MIN_W12):
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    for w3 in vals:
        for w6 in vals:
            w12 = 1.0 - (w3 + w6)
            if w12 < -1e-12: continue
            w12 = max(0.0, w12)
            if w12 + 1e-12 >= min_w12:
                yield (w3, w6, w12)

# ------------------- MAIN -----------------------------------
try:
    logger.info("Starting Workload Forecasting Script (Rpt 288)…")

    # 1) Pull data ------------------------------------------------
    wd_df = bigquery_manager.run_gbq_sql(GBQ_VIEW_QUERY, return_dataframe=True)
    if wd_df.empty: raise ValueError("No data returned.")
    wd_df["date"] = pd.to_datetime(wd_df["date"])
    wd_df = wd_df.sort_values(["client_id", "date"])

    forecasts, audit_rows = [], []

    # 2) Iterate clients -----------------------------------------
    for cid, g in wd_df.groupby("client_id", sort=False):
        ts = g.set_index("date")["target_volume"].asfreq("MS").sort_index()
        ts_model = ts.ffill().bfill()  # stabilize lags

        feat = add_lags(ts_model)
        if feat.shape[0] <= TEST_LEN:
            logger.info(f"· Skipping {cid} – not enough history.")
            continue

        val_index = feat.index[-TEST_LEN:]
        tr_index  = feat.index[:-TEST_LEN]

        preds = {}
        preds["SeasonalNaive"]         = seasonal_naive_series(ts_model, val_index)
        preds["SeasonalNaiveGR"]       = seasonal_naive_gr_series(ts_model, val_index)
        preds["SeasonalNaive3mDiffGR"] = seasonal_naive_3m_diff_gr_series(ts_model, val_index)

        # WeightedLagBlend
        wlf_tr = feat.loc[tr_index].dropna()
        wlf_va = feat.loc[val_index].dropna()
        wblend_weights = None
        if not wlf_tr.empty and not wlf_va.empty:
            best_w, best_s = (1,0,0), np.inf
            for w3, w6, w12 in weight_grid():
                yhat = w3*wlf_tr["lag_3"] + w6*wlf_tr["lag_6"] + w12*wlf_tr["lag_12"]
                s = smape(wlf_tr["y"], yhat)
                if np.isfinite(s) and s < best_s:
                    best_s, best_w = s, (w3, w6, w12)
            wblend_weights = {"w3": best_w[0], "w6": best_w[1], "w12": best_w[2]}
            preds["WeightedLagBlend"] = (best_w[0]*wlf_va["lag_3"]
                                       + best_w[1]*wlf_va["lag_6"]
                                       + best_w[2]*wlf_va["lag_12"])

        # Evaluate
        smapes = {}
        for m, pser in preds.items():
            pser = pd.Series(pser, index=val_index, dtype=float)
            y_true = ts_model.reindex(pser.index).astype(float)
            idx = y_true.index[y_true.notna() & pser.notna()]
            if len(idx)>0:
                yv, pv = y_true.loc[idx], pser.loc[idx]
                smapes[m] = smape(yv, pv)
                audit_rows.append({"client_id": cid, "model": m,
                                   "SMAPE": smapes[m], "MAPE": mape(yv,pv), "RMSE": rmse(yv,pv)})

        if not smapes: continue
        priority = ["SeasonalNaive3mDiffGR", "SeasonalNaiveGR", "SeasonalNaive", "WeightedLagBlend"]
        best_model = min(smapes, key=lambda k: (smapes[k], priority.index(k)))
        logger.info(f"· {cid} Best={best_model} (SMAPE {smapes[best_model]:.2f})")

        # Recursive forecast
        hist = deque(ts_model.dropna().tolist(), maxlen=240)
        last_date = ts_model.index.max()
        future_idx = pd.date_range(last_date, periods=FORECAST_HORIZON+1, freq="MS")[1:]
        fx_tag = f"{FX_ID_PREFIX[best_model]}_{STAMP:%Y%m%d}"
        load_ts_str = STAMP.strftime("%Y-%m-%d %H:%M:%S")

        def append_row(d,v):
            v = 0.0 if not np.isfinite(v) else max(0.0,float(v))
            forecasts.append({"fx_date":d.strftime("%Y-%m-01"),"client_id":cid,
                              "fx_vol":int(round(v)),"fx_id":fx_tag,
                              "fx_status":"forecast","load_ts":load_ts_str})

        for d in future_idx:
            n, last_list = len(hist), list(hist)
            if best_model=="SeasonalNaive":
                pred = last_list[-12] if n>=12 else last_list[-1]
            elif best_model=="SeasonalNaiveGR":
                base = last_list[-12] if n>=12 else last_list[-1]
                r=(np.mean(last_list[-3:])/np.mean(last_list[-6:-3])-1) if n>=6 else 0
                pred=base*(1+r)
            elif best_model=="SeasonalNaive3mDiffGR":
                base = last_list[-12] if n>=12 else last_list[-1]
                r=(np.mean(last_list[-3:])/np.mean(last_list[-6:-3])-1) if n>=6 else 0
                rpy=(np.mean(last_list[-15:-12])/np.mean(last_list[-18:-15])-1) if n>=18 else 0
                pred=base*(1+(r-rpy))
            elif best_model=="WeightedLagBlend" and wblend_weights:
                x3=last_list[-3] if n>=3 else last_list[-1]
                x6=last_list[-6] if n>=6 else last_list[-1]
                x12=last_list[-12] if n>=12 else last_list[-1]
                pred=wblend_weights["w3"]*x3+wblend_weights["w6"]*x6+wblend_weights["w12"]*x12
            else:
                pred=last_list[-1]
            hist.append(pred); append_row(d,pred)

        audit_rows.append({"client_id":cid,"model":f"{best_model}_WINNER",
                           "SMAPE":smapes[best_model],"MAPE":np.nan,"RMSE":np.nan})

    # 3) Push to GBQ + dedup + save --------------------------------
    if forecasts:
        fx_df=pd.DataFrame(forecasts)
        bigquery_manager.import_data_to_bigquery(fx_df, DEST_TABLE,
                                                 gbq_insert_action="append",
                                                 auto_convert_df=True)
        dedup_sql=f"""
        UPDATE `{DEST_TABLE}` t
        SET fx_status='inactive'
        WHERE EXISTS (
          SELECT 1 FROM `{DEST_TABLE}` sub
          WHERE sub.client_id=t.client_id AND sub.fx_date=t.fx_date
            AND sub.load_ts>t.load_ts
        );
        """
        bigquery_manager.run_gbq_sql(dedup_sql, return_dataframe=False)
        fx_df.to_csv(LOCAL_CSV,index=False)
        pd.DataFrame(audit_rows).to_csv(AUDIT_CSV,index=False)
        bigquery_manager.update_log_in_bigquery()
        logger.info("✓ Forecasting complete.")
    else:
        logger.warning("No forecasts generated.")

except Exception as exc:
    email_manager.handle_error("Workload Forecasting Script Failure (Rpt 288)", exc, is_test=True)
