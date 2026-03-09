import os
import math
import pandas as pd
import numpy as np

from scripthelper import Config, Logger, BigQueryManager

# =============================================================================
# CONFIGURATION
# =============================================================================

# Origins where Erlang C applies (real-time queue, agents must be seated).
# Any origin NOT in this set falls back to the linear workload method.
ERLANG_ORIGINS = {'phone', 'chat'}

# Weeks per month (standard WFM constant)
WEEKS_IN_MONTH = 4.333

# Default SLA parameters used when interval data has no values
DEFAULT_SL_TARGET    = 0.80
DEFAULT_SL_THRESHOLD = 30       # seconds
DEFAULT_AHT          = 300      # seconds

# SQL paths
INTERVAL_SQL_PATH = r"C:\WFM_Scripting\Automation\GBQ - cx_interval_30mins_vols.sql"
FTE_SQL_PATH      = r"C:\WFM_Scripting\Automation\GBQ - cx_fte_calculator.sql"


# =============================================================================
# MATH HELPERS
# =============================================================================

def erlang_c_probability(A: float, N: int) -> float:
    """
    Calculates P(wait) — the probability a contact waits in queue — using
    the Erlang C formula.  Returns 1.0 (worst case) if N <= A (server
    utilisation >= 100 %).
    """
    if N <= A:
        return 1.0
    inv_B = 1.0
    for i in range(1, int(N) + 1):
        inv_B = 1.0 + inv_B * (i / A)
    B = 1.0 / inv_B
    C = (N * B) / (N - A * (1.0 - B))
    return C


def required_agents(
    volume: float,
    aht: float,
    target_sl: float = DEFAULT_SL_TARGET,
    target_time: float = DEFAULT_SL_THRESHOLD,
    interval_secs: int = 1800,
) -> int:
    """
    Iterates agent counts upward from the traffic intensity floor until the
    Erlang C service level meets or exceeds target_sl within target_time
    seconds.  Returns 0 for null / zero inputs.
    """
    if pd.isna(volume) or volume <= 0:
        return 0
    if pd.isna(aht) or aht <= 0:
        aht = DEFAULT_AHT

    A = (volume * aht) / interval_secs
    N = math.ceil(A)

    for _ in range(300):                        # hard cap prevents infinite loop
        C  = erlang_c_probability(A, N)
        sl = 1.0 - C * math.exp(-(N - A) * (target_time / aht))
        if sl >= target_sl:
            return N
        N += 1

    return N                                    # return best-effort if cap hit


def calc_final_fte(
    base_fte: float,
    pct: float,
    occ: float,
    shrink: float,
    attr: float,
) -> float:
    """
    Converts a raw base FTE (scheduled seats) to payroll FTE by applying
    vendor split, occupancy cap, shrinkage, and attrition.
    Returns 0.0 for invalid/zero pct or occupancy inputs.
    """
    if pd.isna(pct) or pct <= 0 or pd.isna(occ) or occ <= 0:
        return 0.0
    return (base_fte * pct) / (occ * (1.0 - shrink)) * (1.0 + attr)


def linear_base_fte(
    volume: float,
    aht: float,
    monthly_work_hours: float,
) -> float:
    """
    Flat linear workload method: (Volume × AHT) / 3600 / Monthly Work Hours.
    Used as a fallback for async origins (email) where Erlang C is not valid.
    """
    if pd.isna(volume) or volume <= 0 or monthly_work_hours <= 0:
        return 0.0
    return (volume * aht) / 3600.0 / monthly_work_hours


# =============================================================================
# INTERVAL PROFILE BUILDER
# =============================================================================

def build_arrival_profiles(df_intervals: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates raw 30-minute interval data into a normalised arrival profile
    keyed on (bu, origin, dow, time).

    Each row carries:
      - bucket_vol       : mean offered volume for that slot
      - Avg_AHT          : volume-weighted mean AHT
      - SL_Threshold_Seconds / Target_SLA : mean SLA params
      - Interval_Weight  : fraction of the origin's weekly volume that lands
                           in this specific (dow, time) bucket — used to
                           distribute a monthly forecast across the week.
    """
    df = df_intervals.copy()

    # Normalise origin casing
    df['origin'] = df['origin'].str.lower().str.strip()

    # Parse timestamps
    df['interval_start'] = pd.to_datetime(df['interval_start'])
    df['dow']  = df['interval_start'].dt.dayofweek      # 0=Mon … 6=Sun
    df['time'] = df['interval_start'].dt.time

    # Numeric coercion (Avg_AHT / Service_Level come through as object)
    for col in ['Offered_Volume', 'Avg_AHT', 'SL_Threshold_Seconds']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Derived workload column for weighted-AHT calculation
    df['workload'] = df['Offered_Volume'] * df['Avg_AHT']

    # Aggregate: mean across the historical window so the profile represents
    # a "typical" week rather than the sum of all history.
    profile = (
        df.groupby(['bu', 'origin', 'dow', 'time'])
        .agg(
            bucket_vol            = ('Offered_Volume',       'mean'),
            bucket_workload       = ('workload',             'mean'),
            SL_Threshold_Seconds  = ('SL_Threshold_Seconds', 'mean'),
        )
        .reset_index()
    )

    # Defensive defaults
    profile['SL_Threshold_Seconds'] = profile['SL_Threshold_Seconds'].where(
        profile['SL_Threshold_Seconds'] > 0, DEFAULT_SL_THRESHOLD
    )

    # Volume-weighted AHT per bucket
    profile['Avg_AHT'] = np.where(
        profile['bucket_vol'] > 0,
        profile['bucket_workload'] / profile['bucket_vol'],
        DEFAULT_AHT,
    )

    # Fixed SLA target (all Answerlink contracts = 80 / 30s currently)
    # If your BQ query ever exposes a per-client Target_SLA, join it here.
    profile['Target_SLA'] = DEFAULT_SL_TARGET

    # Interval weight = this bucket's share of the origin's total weekly volume
    origin_totals = (
        profile.groupby(['bu', 'origin'])['bucket_vol']
        .sum()
        .reset_index()
        .rename(columns={'bucket_vol': 'origin_total_vol'})
    )
    profile = profile.merge(origin_totals, on=['bu', 'origin'])
    profile['Interval_Weight'] = np.where(
        profile['origin_total_vol'] > 0,
        profile['bucket_vol'] / profile['origin_total_vol'],
        0.0,
    )

    return profile


# =============================================================================
# ERLANG ENGINE  (pooled, then apportioned)
# =============================================================================

def run_erlang_pool(
    df_fte: pd.DataFrame,
    profile_df: pd.DataFrame,
    logger,
) -> pd.DataFrame:
    """
    Core computation loop.

    Groups the FTE forecast by (date, bu, origin) to form the agent pool,
    runs Erlang C on the pooled volume using the arrival profile, then
    apportions the resulting base FTE back to each client row by workload
    share.

    For origins outside ERLANG_ORIGINS the linear method is used instead.

    Returns df_fte with six appended columns:
        Domestic_Actual_Erlang_FTE
        Telus_Actual_Erlang_FTE
        Total_Actual_Erlang_FTE
        Domestic_Forecast_Erlang_FTE
        Telus_Forecast_Erlang_FTE
        Total_Forecast_Erlang_FTE
    """

    # Pre-compute workload columns for both actual and forecast volumes.
    #
    # AHT resolution priority:
    #   1. Actual_AHT  — available for historical months with real call data
    #   2. domestic_fixed_aht — populated for all rows including future forecast
    #      months where Actual_AHT is 0 (no actuals exist yet)
    #
    # This ensures forecast_workload is never zero for future periods, which
    # would cause workload-share apportionment to produce NaN/0 results.
    df_fte = df_fte.copy()
    df_fte['origin_norm'] = df_fte['origin'].str.lower().str.strip()

    df_fte['resolved_aht'] = np.where(
        df_fte['Actual_AHT'] > 0,
        df_fte['Actual_AHT'],
        df_fte['domestic_fixed_aht'],
    )

    df_fte['actual_workload']   = df_fte['Actual_Volume']       * df_fte['resolved_aht']
    df_fte['forecast_workload'] = df_fte['Net_Forecast_Volume'] * df_fte['resolved_aht']

    results: dict = {}     # index → result dict

    pool_keys = ['date', 'bu', 'origin_norm']

    total_pools = df_fte.groupby(pool_keys).ngroups
    pool_counter = 0

    for (date_val, bu_val, origin_val), group in df_fte.groupby(pool_keys):

        pool_counter += 1
        logger.info(
            f"  [{pool_counter}/{total_pools}] Pool: bu={bu_val} | "
            f"origin={origin_val} | date={date_val} | rows={len(group)}"
        )

        # ── Monthly work hours ────────────────────────────────────────────
        b_days  = group['business_day_count'].max()
        b_hours = group['business_hours'].max()
        monthly_work_hours = (b_days * b_hours) if (b_days * b_hours) > 0 else 157.5

        # ── Pool totals ───────────────────────────────────────────────────
        total_actual_vol    = group['Actual_Volume'].sum()
        total_forecast_vol  = group['Net_Forecast_Volume'].sum()
        total_actual_wl     = group['actual_workload'].sum()
        total_forecast_wl   = group['forecast_workload'].sum()

        # ── Dispatch by method ────────────────────────────────────────────
        use_erlang = origin_val in ERLANG_ORIGINS

        if use_erlang:
            origin_profile = profile_df[
                (profile_df['bu']     == bu_val) &
                (profile_df['origin'] == origin_val)
            ]

            if origin_profile.empty:
                logger.warning(
                    f"    No interval profile found for bu={bu_val}, "
                    f"origin={origin_val}. Falling back to linear method."
                )
                use_erlang = False

        # ──────────────────────────────────────────────────────────────────
        # PATH A: Erlang C (phone / chat)
        # ──────────────────────────────────────────────────────────────────
        if use_erlang:
            actual_base_fte   = _erlang_base_fte(
                total_actual_vol, origin_profile, monthly_work_hours
            )
            forecast_base_fte = _erlang_base_fte(
                total_forecast_vol, origin_profile, monthly_work_hours
            )
        # ──────────────────────────────────────────────────────────────────
        # PATH B: Linear fallback (email / unknown)
        # ──────────────────────────────────────────────────────────────────
        else:
            # For async work (email) there is no queue concept so we use the
            # pool's aggregate linear FTE, then apportion by workload share.
            pool_aht = (
                (group['actual_workload'].sum()   / total_actual_vol)
                if total_actual_vol > 0 else DEFAULT_AHT
            )
            forecast_aht = (
                (group['forecast_workload'].sum() / total_forecast_vol)
                if total_forecast_vol > 0 else DEFAULT_AHT
            )
            actual_base_fte   = linear_base_fte(total_actual_vol,   pool_aht,     monthly_work_hours)
            forecast_base_fte = linear_base_fte(total_forecast_vol, forecast_aht, monthly_work_hours)

        # ──────────────────────────────────────────────────────────────────
        # Apportion base FTE to each client row by workload share, then
        # apply per-row shrinkage / occupancy / attrition.
        # ──────────────────────────────────────────────────────────────────
        for idx, row in group.iterrows():

            actual_share   = row['actual_workload']   / total_actual_wl   if total_actual_wl   > 0 else 0.0
            forecast_share = row['forecast_workload'] / total_forecast_wl if total_forecast_wl > 0 else 0.0

            client_actual_base   = actual_base_fte   * actual_share
            client_forecast_base = forecast_base_fte * forecast_share

            dom_actual   = calc_final_fte(client_actual_base,   row['Domestic_Pct'], row['domestic_occupancy'], row['domestic_shrinkage'], row['domestic_attrition'])
            tel_actual   = calc_final_fte(client_actual_base,   row['Telus_Pct'],    row['telus_occupancy'],    row['telus_shrinkage'],    row['telus_attrition'])
            dom_forecast = calc_final_fte(client_forecast_base, row['Domestic_Pct'], row['domestic_occupancy'], row['domestic_shrinkage'], row['domestic_attrition'])
            tel_forecast = calc_final_fte(client_forecast_base, row['Telus_Pct'],    row['telus_occupancy'],    row['telus_shrinkage'],    row['telus_attrition'])

            results[idx] = {
                'Domestic_Actual_Erlang_FTE':   round(dom_actual,   2),
                'Telus_Actual_Erlang_FTE':       round(tel_actual,   2),
                'Total_Actual_Erlang_FTE':       round(dom_actual   + tel_actual,   2),
                'Domestic_Forecast_Erlang_FTE':  round(dom_forecast, 2),
                'Telus_Forecast_Erlang_FTE':     round(tel_forecast, 2),
                'Total_Forecast_Erlang_FTE':     round(dom_forecast + tel_forecast, 2),
            }

    # Reconstruct in original index order and concat
    result_rows = [
        results.get(i, {
            'Domestic_Actual_Erlang_FTE':  0.0,
            'Telus_Actual_Erlang_FTE':     0.0,
            'Total_Actual_Erlang_FTE':     0.0,
            'Domestic_Forecast_Erlang_FTE':0.0,
            'Telus_Forecast_Erlang_FTE':   0.0,
            'Total_Forecast_Erlang_FTE':   0.0,
        })
        for i in df_fte.index
    ]
    df_results = pd.DataFrame(result_rows)

    # Drop the helper columns before returning
    df_fte.drop(columns=['origin_norm', 'resolved_aht', 'actual_workload', 'forecast_workload'], inplace=True)

    return pd.concat([df_fte.reset_index(drop=True), df_results], axis=1)


def _erlang_base_fte(
    total_pool_vol: float,
    origin_profile: pd.DataFrame,
    monthly_work_hours: float,
) -> float:
    """
    Simulates one month of Erlang C across all 336 weekly half-hour slots
    for a given pooled volume, then converts total agent-hours to base FTE.

    Args:
        total_pool_vol:    Total monthly contact volume for the entire pool.
        origin_profile:    Filtered profile DataFrame for this (bu, origin).
        monthly_work_hours: Business days × business hours for the month.

    Returns:
        Base FTE (scheduled seats, before shrinkage / occupancy).
    """
    if total_pool_vol <= 0 or origin_profile.empty:
        return 0.0

    weekly_vol = total_pool_vol / WEEKS_IN_MONTH
    weekly_agent_hours = 0.0

    for _, p_row in origin_profile.iterrows():
        sim_vol = weekly_vol * p_row['Interval_Weight']
        if sim_vol < 0.01:
            continue
        agents = required_agents(
            volume      = sim_vol,
            aht         = p_row['Avg_AHT'],
            target_sl   = p_row['Target_SLA'],
            target_time = p_row['SL_Threshold_Seconds'],
        )
        weekly_agent_hours += agents * 0.5      # each slot = 0.5 hrs

    monthly_agent_hours = weekly_agent_hours * WEEKS_IN_MONTH
    return monthly_agent_hours / monthly_work_hours if monthly_work_hours > 0 else 0.0


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prep_fte_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerces all numeric FTE-math columns to float and fills NaN → 0.
    Parses the date column to datetime.  Safe for mixed BigQuery date types.
    """
    numeric_cols = [
        'Actual_Volume', 'Net_Forecast_Volume', 'Actual_AHT',
        'domestic_fixed_aht', 'telus_fixed_aht',
        'business_day_count', 'business_hours',
        'Domestic_Pct', 'Telus_Pct',
        'domestic_occupancy', 'domestic_shrinkage', 'domestic_attrition',
        'telus_occupancy',    'telus_shrinkage',    'telus_attrition',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['date'] = pd.to_datetime(df['date'])
    return df


def prep_interval_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerces numeric columns and normalises origin casing in the interval frame.
    """
    for col in ['Offered_Volume', 'Avg_AHT', 'SL_Threshold_Seconds']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['origin'] = df['origin'].str.lower().str.strip()
    df['interval_start'] = pd.to_datetime(df['interval_start'])
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = Config()
    logger = Logger()
    bq     = BigQueryManager(config)

    logger.info("=" * 70)
    logger.info("Starting Generalised Erlang C FTE Pipeline")
    logger.info("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────
    logger.info("Loading SQL files...")
    with open(INTERVAL_SQL_PATH, 'r') as f:
        interval_sql = f.read()
    with open(FTE_SQL_PATH, 'r') as f:
        fte_sql = f.read()

    logger.info("Executing BigQuery queries...")
    df_intervals_raw = bq.fetch_data_bigquery(interval_sql)
    df_fte_raw       = bq.fetch_data_bigquery(fte_sql)

    logger.info(f"  Interval rows loaded : {len(df_intervals_raw):,}")
    logger.info(f"  FTE rows loaded      : {len(df_fte_raw):,}")

    # ── 2. Prep ───────────────────────────────────────────────────────────
    logger.info("Preparing dataframes...")
    df_fte       = prep_fte_dataframe(df_fte_raw.copy())
    df_intervals = prep_interval_dataframe(df_intervals_raw.copy())

    # Log coverage summary
    for bu in df_fte['bu'].unique():
        sub = df_fte[df_fte['bu'] == bu]
        logger.info(
            f"  FTE  | bu={bu:<30} | "
            f"rows={len(sub):>4} | "
            f"origins={sorted(sub['origin'].str.lower().unique().tolist())}"
        )
    for bu in df_intervals['bu'].unique():
        sub = df_intervals[df_intervals['bu'] == bu]
        logger.info(
            f"  IVLS | bu={bu:<30} | "
            f"rows={len(sub):>6} | "
            f"origins={sorted(sub['origin'].unique().tolist())}"
        )

    # ── 3. Build arrival profiles ─────────────────────────────────────────
    logger.info("Building pooled arrival profiles...")
    profile_df = build_arrival_profiles(df_intervals)
    logger.info(
        f"  Profile rows: {len(profile_df):,} "
        f"across {profile_df[['bu','origin']].drop_duplicates().shape[0]} bu/origin pools"
    )

    # ── 4. Run Erlang engine ──────────────────────────────────────────────
    logger.info("Running Erlang C engine across all BU/origin pools...")
    df_final = run_erlang_pool(df_fte, profile_df, logger)

    logger.info(f"  Output rows: {len(df_final):,} | columns: {len(df_final.columns)}")

    # ── 5. Export ─────────────────────────────────────────────────────────
    desktop_path = os.path.join(
        os.environ['USERPROFILE'], 'Desktop', 'Erlang_FTE_Calculator_All_BUs.csv'
    )
    df_final.to_csv(desktop_path, index=False)
    logger.info(f"Export complete → {desktop_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
