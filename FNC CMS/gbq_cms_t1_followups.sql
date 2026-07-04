WITH 
/* -----------------------------------------------------------------------
   STEP 0: CONFIGURATION VARIABLES
   Update these values to change the query's behavior globally.
----------------------------------------------------------------------- */
Config AS (
  SELECT 
    'FNC CMS' AS target_bu,
    TIMESTAMP('2025-06-01') AS filter_start_date,--When to trust the data
    CAST('07:00:00' AS TIME) AS biz_start_time, -- operation hours
    CAST('19:00:00' AS TIME) AS biz_end_time, -- operational hours
    1440.0 AS target_follow_up_minutes --- Follow Up definition from Salesforce
),

/* -----------------------------------------------------------------------
   STEP 1: RAW CASE TIMESTAMPS
----------------------------------------------------------------------- */
RawCases AS (
  SELECT
    c.Id AS case_id,
    
    -- Visible CST converted column for the month grouping
    DATETIME(c.CreatedDate, 'America/Chicago') AS start_dt_cst,
    
    -- Hidden calculation column: uses CURRENT_TIMESTAMP if open to calculate hours elapsed so far
    DATETIME(COALESCE(c.CX_ResolutionTimestamp__c, c.ClosedDate, CURRENT_TIMESTAMP()), 'America/Chicago') AS calc_end_dt_cst

  FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.cx_ucrm_case` c
  CROSS JOIN Config cfg
  WHERE c.CreatedDate IS NOT NULL
    AND c.CreatedDate >= cfg.filter_start_date
    AND c.CX_BusinessUnit__c = cfg.target_bu
),

/* -----------------------------------------------------------------------
   STEP 2: UNROLL DATES & APPLY CONFIG BUSINESS HOURS
----------------------------------------------------------------------- */
ExpandedCaseDates AS (
  SELECT
    r.case_id,
    r.start_dt_cst,
    r.calc_end_dt_cst, 
    d AS current_date,
    -- Define the boundary for each specific calendar day using the Config CTE
    DATETIME(d, cfg.biz_start_time) AS biz_start,
    DATETIME(d, cfg.biz_end_time) AS biz_end
  FROM RawCases r
  CROSS JOIN Config cfg,
  UNNEST(GENERATE_DATE_ARRAY(DATE(r.start_dt_cst), DATE(GREATEST(r.start_dt_cst, r.calc_end_dt_cst)))) AS d
),

/* -----------------------------------------------------------------------
   STEP 3: EXCLUDE WEEKENDS & HOLIDAYS, CALCULATE DAILY OVERLAP
----------------------------------------------------------------------- */
DailyBusinessHours AS (
  SELECT
    e.case_id,
    e.current_date,
    GREATEST(e.biz_start, e.start_dt_cst) AS day_start,
    LEAST(e.biz_end, e.calc_end_dt_cst) AS day_end
  FROM ExpandedCaseDates e
  LEFT JOIN `clgx-taxbi-reg-bf03.tax_clnt_svcs.holiday_calendar` hc
    ON e.current_date = hc.holdate
  WHERE 
    -- Exclude Weekends: 1 = Sunday, 7 = Saturday in BigQuery DAYOFWEEK
    EXTRACT(DAYOFWEEK FROM e.current_date) NOT IN (1, 7)
    AND hc.holdate IS NULL
),

/* -----------------------------------------------------------------------
   STEP 4: SUM HOURS PER CASE
----------------------------------------------------------------------- */
CaseTotalHours AS (
  SELECT
    case_id,
    SUM(
      CASE 
        WHEN day_end > day_start THEN DATETIME_DIFF(day_end, day_start, SECOND) / 3600.0 
        ELSE 0 
      END
    ) AS total_business_hours
  FROM DailyBusinessHours
  GROUP BY 1
),

/* -----------------------------------------------------------------------
   STEP 5: CALCULATE CASE-LEVEL FOLLOW-UPS USING CONFIG
----------------------------------------------------------------------- */
CaseLevelMetrics AS (
  SELECT
    r.case_id,
    DATE_TRUNC(DATE(r.start_dt_cst), MONTH) AS created_month,
    -- Calculate expected follow-ups pulling the target SLA from Config
    FLOOR((COALESCE(c.total_business_hours, 0) * 60) / cfg.target_follow_up_minutes) AS expected_follow_ups
  FROM RawCases r
  LEFT JOIN CaseTotalHours c
    ON r.case_id = c.case_id
  CROSS JOIN Config cfg
)

/* -----------------------------------------------------------------------
   FINAL OUTPUT: MONTHLY AGGREGATION
----------------------------------------------------------------------- */
SELECT
  FORMAT_DATE('%Y-%m', created_month) AS created_month,
  COUNT(case_id) AS total_cases,
  ROUND(AVG(expected_follow_ups), 2) AS avg_expected_follow_ups
FROM CaseLevelMetrics
GROUP BY 1
ORDER BY 1 ASC;