/* =============================================================================
   FNC - CMS (Platforms) — LINEAR DEMAND DATA LAYER  (Email/Web)
   AUTHOR: refactor of gbq_cms_linear_tickets_fte.sql (Jhonnatan Gonzalez)

   PURPOSE
   -------
   Emit ONLY the raw monthly building blocks the Linear FTE model needs:
     - the M-F business-day calendar
     - the ticket-volume layers (actual, base forecast, BU adjustment, net forecast)

   All capacity *policy* and the FTE math are REMOVED. They now live in the
   workbook Config tab as editable inputs, so occupancy / shrinkage / attrition /
   op-hours / AHT can be tuned without re-running SQL.

   REMOVED FROM THIS QUERY  (now Config-tab inputs / Summary formulas):
     daily_op_hours, occupancy, shrinkage, attrition, aht_secs (CMS = 867),
     gross_available_hrs_per_fte, on_floor_hrs_per_fte, productive_hrs_per_fte,
     actual_workload_hours,  actual_net_fte,   actual_gross_fte,
     forecast_workload_hours, forecast_net_fte, forecast_gross_fte

   KEPT HERE  (measured / calendar facts):
     total_days_in_month, weekday_count, weekday_holiday_count, capacity_days,
     actual_volume, base_fx_vol, bu_adj_vol, net_forecast_vol

   VOLUME LOGIC — UNCHANGED FROM THE PRIOR MODEL
     - actual_volume    = cx_ucrm_daily_kpi  (cases_opened - cases_cancelled), NOT adjusted
     - base_fx_vol      = cx_nontax_fx       (vol_type = 'email/web', fx_status = 'A')
     - bu_adj_vol       = cx_nontax_bu_adj_vol (origin = 'email/web', status = 'A')
     - net_forecast_vol = GREATEST(base_fx_vol + bu_adj_vol, 0)   -- signed adj, ADDED

   OUTPUT KEY-TYPE
     call_month is TEXT 'yyyy-mm-dd' (FORMAT_DATE) to match the workbook text-key lookups.
============================================================================= */

WITH

MonthCalendar AS (
  SELECT
    DATE_TRUNC(dm.MDY_NBR_FRMT_DAY_DT, MONTH)                                AS month_start,
    EXTRACT(DAY FROM LAST_DAY(DATE_TRUNC(dm.MDY_NBR_FRMT_DAY_DT, MONTH)))    AS total_days_in_month,
    COUNT(CASE WHEN dm.DAY_OF_WK_SHORTDESC NOT IN ('SAT','SUN') THEN 1 END)  AS weekday_count
  FROM `taxdw.datemaster` dm
  WHERE dm.MDY_NBR_FRMT_DAY_DT BETWEEN '2025-01-01'
    AND DATE_ADD(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 15 MONTH)
  GROUP BY 1, 2
),

HolidayCalendar AS (
  SELECT
    DATE_TRUNC(holdate, MONTH)  AS month_start,
    COUNT(*)                    AS holiday_count
  FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.holiday_calendar`
  WHERE holdate >= DATE('2025-01-01')
    AND EXTRACT(DAYOFWEEK FROM holdate) BETWEEN 2 AND 6
  GROUP BY 1
),

BusinessDays AS (
  SELECT
    mc.month_start,
    mc.total_days_in_month,
    mc.weekday_count,
    COALESCE(hc.holiday_count, 0)                                   AS weekday_holiday_count,
    GREATEST(mc.weekday_count - COALESCE(hc.holiday_count, 0), 0)   AS capacity_days
  FROM MonthCalendar mc
  LEFT JOIN HolidayCalendar hc ON mc.month_start = hc.month_start
),

ValidMappings AS (
  SELECT DISTINCT
    ck.Client                  AS client,
    CAST(ck.Id AS STRING)      AS client_id,
    ck.`Groups`                AS `groups`,
    ck.`Business Unit`         AS business_unit
  FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.cx_ucrm_daily_kpi` kpi
  JOIN `clgx-taxbi-reg-bf03.tax_clnt_svcs.cc_client_key` ck
    ON CAST(kpi.client_id AS STRING) = CAST(ck.Id AS STRING)
  WHERE ck.`Business Unit` = 'Platforms'
    AND ck.Client = 'FNC - CMS'
    AND kpi.origin NOT IN ('Phone', 'Chat', 'Unknown Origin')
),

HistoricalActuals AS (
  SELECT
    DATE_TRUNC(dt, MONTH)                       AS month_start,
    CAST(client_id AS STRING)                   AS client_id,
    SUM(cases_opened) - SUM(cases_cancelled)    AS actual_volume
  FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.cx_ucrm_daily_kpi`
  WHERE bus_unit = 'Platforms'
    AND LOWER(origin) IN (
        'portal', 'web', 'web portal', 'email',
        'internal request', 'internal_request'
    )
    AND DATE_TRUNC(dt, MONTH) < DATE_TRUNC(CURRENT_DATE(), MONTH)
  GROUP BY 1, 2
),

BaseForecast AS (
  SELECT
    DATE(fx_date)                AS month_start,
    CAST(client_id AS STRING)    AS client_id,
    SUM(fx_vol)                  AS base_fx_vol
  FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.cx_nontax_fx`
  WHERE bu = 'Platforms'
    AND LOWER(vol_type) = 'email/web'
    AND fx_status = 'A'
    AND DATE(fx_date) >= '2025-01-01'
  GROUP BY 1, 2
),

BuAdjustments AS (
  SELECT
    DATE_TRUNC(dt, MONTH)        AS month_start,
    CAST(client_id AS STRING)    AS client_id,
    SUM(bu_adj_vol)              AS bu_adj_vol
  FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.cx_nontax_bu_adj_vol`
  WHERE bu = 'Platforms'
    AND origin = 'email/web'
    AND status = 'A'
  GROUP BY 1, 2
),

NetForecast AS (
  SELECT
    bf.month_start,
    bf.client_id,
    bf.base_fx_vol,
    COALESCE(ba.bu_adj_vol, 0)                                AS bu_adj_vol,
    GREATEST(bf.base_fx_vol + COALESCE(ba.bu_adj_vol, 0), 0)  AS net_forecast_vol
  FROM BaseForecast bf
  LEFT JOIN BuAdjustments ba
    ON  bf.month_start = ba.month_start
    AND bf.client_id   = ba.client_id
),

FullSpine AS (
  SELECT
    bd.month_start,
    vm.business_unit,
    vm.client,
    vm.client_id,
    vm.`groups`,
    'email/web'              AS origin,
    bd.total_days_in_month,
    bd.weekday_count,
    bd.weekday_holiday_count,
    bd.capacity_days
  FROM BusinessDays bd
  CROSS JOIN ValidMappings vm
)

/* -----------------------------------------------------------------------
   FINAL OUTPUT  —  TICKET VOLUMES + CALENDAR ONLY  (no policy, no FTE)
----------------------------------------------------------------------- */
SELECT
  FORMAT_DATE('%Y-%m-%d', fs.month_start)   AS call_month,   -- TEXT key 'yyyy-mm-dd'
  fs.business_unit,
  fs.client,
  fs.client_id,
  fs.`groups`,
  fs.origin,

  /* --- Business-day calendar --- */
  fs.total_days_in_month,
  fs.weekday_count,
  fs.weekday_holiday_count,
  fs.capacity_days,

  /* --- Ticket-volume layers --- */
  ha.actual_volume,
  nf.base_fx_vol,
  COALESCE(nf.bu_adj_vol, 0)  AS bu_adj_vol,
  nf.net_forecast_vol

FROM FullSpine fs
LEFT JOIN HistoricalActuals ha
  ON  fs.month_start = ha.month_start
  AND fs.client_id   = ha.client_id
LEFT JOIN NetForecast nf
  ON  fs.month_start = nf.month_start
  AND fs.client_id   = nf.client_id

ORDER BY fs.month_start ASC, fs.client ASC;