/* =============================================================================
   DYNAMIC MONTHLY LINEAR FTE CAPACITY MODEL — Platforms (Client: FNC - CMS) (Email/Web)
   AUTHOR: Jhonnatan Gonzalez
   DATA SOURCE: cx_ucrm_daily_kpi (historical actuals) + cx_nontax_fx (base forecast)
                + cx_nontax_bu_adj_vol (BU volume adjustments)

   STRUCTURE:
   Every month returns ONE row with BOTH actual and forecast columns populated
   side by side — enabling direct gap analysis (actual vs forecast) in any BI tool
   without pivoting or joining. 

   VOLUME LOGIC (3 LAYERS):
     Layer 1 — Historical Actuals  : cx_ucrm_daily_kpi  (cases_opened - cases_cancelled)
     Layer 2 — Base Forecast       : cx_nontax_fx        (vol_type = 'email/web', fx_status = 'A')
     Layer 3 — BU Adjustment       : cx_nontax_bu_adj_vol (origin = 'email/web', status = 'A')
     Net Forecast Volume           : GREATEST(base_fx_vol + bu_adj_vol, 0)

   FORMULA (same for both actual and forecast paths):
     FTE = ((Volume × AHT_secs) / 3600)
           / (capacity_days × daily_op_hours × occupancy × (1 - shrinkage))
           × (1 + attrition)

   WHERE (FNC - CMS SPECIFIC):
     capacity_days    = weekdays - weekday holidays
     daily_op_hours   = 12.0   (07:00–19:00 CST, M-F)
     occupancy        = 0.70   (70% target for FNC - CMS email/web)
     shrinkage        = 0.30   (30% off-task time)
     attrition        = 0.10   (10% hiring buffer above net seat requirement)
     aht_secs         = 381.0  (6.35 min — FNC - CMS email/web standard)
============================================================================= */

WITH

/* -----------------------------------------------------------------------
   STEP 1: MONTH CALENDAR
   Spans from 2025-01-01 through 15 months out to cover full forecast horizon.
----------------------------------------------------------------------- */
MonthCalendar AS (
  SELECT
    DATE_TRUNC(dm.MDY_NBR_FRMT_DAY_DT, MONTH)                          AS month_start,
    COUNT(
      CASE WHEN dm.DAY_OF_WK_SHORTDESC NOT IN ('SAT', 'SUN') THEN 1 END
    )                                                                   AS weekday_count
  FROM `taxdw.datemaster` dm
  WHERE dm.MDY_NBR_FRMT_DAY_DT BETWEEN '2025-01-01'
    AND DATE_ADD(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 15 MONTH)
  GROUP BY 1
),

/* -----------------------------------------------------------------------
   STEP 2: HOLIDAY CALENDAR — weekday holidays only (M-F schedule)
----------------------------------------------------------------------- */
HolidayCalendar AS (
  SELECT
    DATE_TRUNC(holdate, MONTH)                                          AS month_start,
    COUNT(*)                                                            AS holiday_count
  FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.holiday_calendar`
  WHERE holdate >= DATE('2025-01-01')
    AND EXTRACT(DAYOFWEEK FROM holdate) BETWEEN 2 AND 6
  GROUP BY 1
),

/* -----------------------------------------------------------------------
   STEP 3: BUSINESS DAYS & CAPACITY DENOMINATOR
----------------------------------------------------------------------- */
BusinessDays AS (
  SELECT
    mc.month_start,
    mc.weekday_count,
    COALESCE(hc.holiday_count, 0)                                       AS holiday_count,
    GREATEST(mc.weekday_count - COALESCE(hc.holiday_count, 0), 0)       AS capacity_days,

    -- Capacity policy constants 
    12.0                                                                AS daily_op_hours,
    0.70                                                                AS occupancy,
    0.30                                                                AS shrinkage,
    0.10                                                                AS attrition,
    867.0                                                               AS aht_secs,

    -- Denominator build steps
    ROUND(GREATEST(mc.weekday_count - COALESCE(hc.holiday_count, 0), 0)
          * 12.0, 2)                                                    AS gross_available_hrs_per_fte,

    ROUND(GREATEST(mc.weekday_count - COALESCE(hc.holiday_count, 0), 0)
          * 12.0 * (1 - 0.30), 2)                                      AS on_floor_hrs_per_fte,

    ROUND(GREATEST(mc.weekday_count - COALESCE(hc.holiday_count, 0), 0)
          * 12.0 * (1 - 0.30) * 0.70, 2)                               AS productive_hrs_per_fte

  FROM MonthCalendar mc
  LEFT JOIN HolidayCalendar hc ON mc.month_start = hc.month_start
),

/* -----------------------------------------------------------------------
   STEP 4: CLIENT SCAFFOLD
   Filters specifically for BU = Platforms AND Client = FNC - CMS.
----------------------------------------------------------------------- */
ValidMappings AS (
  SELECT DISTINCT
    ck.Client                                                           AS client,
    CAST(ck.Id AS STRING)                                               AS client_id,
    ck.`Groups`                                                         AS `groups`,
    ck.`Business Unit`                                                  AS business_unit
  FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.cx_ucrm_daily_kpi` kpi
  JOIN `clgx-taxbi-reg-bf03.tax_clnt_svcs.cc_client_key` ck
    ON CAST(kpi.client_id AS STRING) = CAST(ck.Id AS STRING)
  WHERE ck.`Business Unit` = 'Platforms'
    AND ck.Client = 'FNC - CMS'
    AND kpi.origin NOT IN ('Phone', 'Chat', 'Unknown Origin')
),

/* -----------------------------------------------------------------------
   STEP 5: HISTORICAL ACTUALS (UCRM)
   Now pulling from Platforms. (The LEFT JOIN in FullSpine ensures 
   only FNC - CMS client data survives).
----------------------------------------------------------------------- */
HistoricalActuals AS (
  SELECT
    DATE_TRUNC(dt, MONTH)                                               AS month_start,
    CAST(client_id AS STRING)                                           AS client_id,
    SUM(cases_opened) - SUM(cases_cancelled)                            AS actual_volume
  FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.cx_ucrm_daily_kpi`
  WHERE bus_unit = 'Platforms'
    AND LOWER(origin) IN (
        'portal', 'web', 'web portal', 'email',
        'internal request', 'internal_request'
    )
    AND DATE_TRUNC(dt, MONTH) < DATE_TRUNC(CURRENT_DATE(), MONTH)
  GROUP BY 1, 2
),

/* -----------------------------------------------------------------------
   STEP 6: BASE FORECAST (cx_nontax_fx)
----------------------------------------------------------------------- */
BaseForecast AS (
  SELECT
    DATE(fx_date)                                                       AS month_start,
    CAST(client_id AS STRING)                                           AS client_id,
    SUM(fx_vol)                                                         AS base_fx_vol
  FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.cx_nontax_fx`
  WHERE bu = 'Platforms'
    AND LOWER(vol_type) = 'email/web'
    AND fx_status = 'A'
    AND DATE(fx_date) >= '2025-01-01'
  GROUP BY 1, 2
),

/* -----------------------------------------------------------------------
   STEP 7: BU VOLUME ADJUSTMENTS (cx_nontax_bu_adj_vol)
----------------------------------------------------------------------- */
BuAdjustments AS (
  SELECT
    DATE_TRUNC(dt, MONTH)                                               AS month_start,
    CAST(client_id AS STRING)                                           AS client_id,
    SUM(bu_adj_vol)                                                     AS bu_adj_vol
  FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.cx_nontax_bu_adj_vol`
  WHERE bu = 'Platforms'
    AND origin = 'email/web'
    AND status = 'A'
  GROUP BY 1, 2
),

/* -----------------------------------------------------------------------
   STEP 8: NET FORECAST VOLUME
----------------------------------------------------------------------- */
NetForecast AS (
  SELECT
    bf.month_start,
    bf.client_id,
    bf.base_fx_vol,
    COALESCE(ba.bu_adj_vol, 0)                                          AS bu_adj_vol,
    GREATEST(bf.base_fx_vol + COALESCE(ba.bu_adj_vol, 0), 0)           AS net_forecast_vol
  FROM BaseForecast bf
  LEFT JOIN BuAdjustments ba
    ON  bf.month_start = ba.month_start
    AND bf.client_id   = ba.client_id
),

/* -----------------------------------------------------------------------
   STEP 9: FULL MONTH × CLIENT SPINE
----------------------------------------------------------------------- */
FullSpine AS (
  SELECT
    bd.month_start,
    vm.business_unit,
    vm.client,
    vm.client_id,
    vm.`groups`,
    'email/web'                                                         AS origin,
    bd.weekday_count,
    bd.holiday_count,
    bd.capacity_days,
    bd.daily_op_hours,
    bd.occupancy,
    bd.shrinkage,
    bd.attrition,
    bd.aht_secs,
    bd.gross_available_hrs_per_fte,
    bd.on_floor_hrs_per_fte,
    bd.productive_hrs_per_fte
  FROM BusinessDays bd
  CROSS JOIN ValidMappings vm
)

/* -----------------------------------------------------------------------
   FINAL OUTPUT
----------------------------------------------------------------------- */
SELECT
  fs.month_start                                                        AS call_month,
  fs.business_unit,
  fs.client,
  fs.client_id,
  fs.`groups`,
  fs.origin,

  /* --- CAPACITY BUILDING BLOCKS --- */
  fs.weekday_count,
  fs.holiday_count,
  fs.capacity_days,
  fs.daily_op_hours,
  fs.occupancy,
  fs.shrinkage,
  fs.attrition,
  fs.aht_secs,
  fs.gross_available_hrs_per_fte,
  fs.on_floor_hrs_per_fte,
  fs.productive_hrs_per_fte,

  /* --- ACTUAL VOLUME LAYER --- */
  ha.actual_volume,

  /* --- FORECAST VOLUME LAYERS --- */
  nf.base_fx_vol,
  nf.bu_adj_vol,
  nf.net_forecast_vol,

  /* --- ACTUAL FTE --- */
  ROUND(COALESCE(ha.actual_volume, 0) * fs.aht_secs / 3600, 4)          AS actual_workload_hours,
  ROUND(
    SAFE_DIVIDE(
      COALESCE(ha.actual_volume, 0) * fs.aht_secs / 3600,
      fs.productive_hrs_per_fte
    ), 4)                                                               AS actual_net_fte,
  ROUND(
    SAFE_DIVIDE(
      COALESCE(ha.actual_volume, 0) * fs.aht_secs / 3600,
      fs.productive_hrs_per_fte
    ) * (1 + fs.attrition), 2)                                         AS actual_gross_fte,

  /* --- FORECAST FTE --- */
  ROUND(COALESCE(nf.net_forecast_vol, 0) * fs.aht_secs / 3600, 4)       AS forecast_workload_hours,
  ROUND(
    SAFE_DIVIDE(
      COALESCE(nf.net_forecast_vol, 0) * fs.aht_secs / 3600,
      fs.productive_hrs_per_fte
    ), 4)                                                               AS forecast_net_fte,
  ROUND(
    SAFE_DIVIDE(
      COALESCE(nf.net_forecast_vol, 0) * fs.aht_secs / 3600,
      fs.productive_hrs_per_fte
    ) * (1 + fs.attrition), 2)                                         AS forecast_gross_fte

FROM FullSpine fs
LEFT JOIN HistoricalActuals ha
  ON  fs.month_start = ha.month_start
  AND fs.client_id   = ha.client_id
LEFT JOIN NetForecast nf
  ON  fs.month_start = nf.month_start
  AND fs.client_id   = nf.client_id

ORDER BY fs.month_start ASC, fs.client ASC;