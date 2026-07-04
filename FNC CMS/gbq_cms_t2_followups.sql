WITH 
/* -----------------------------------------------------------------------
   STEP 1: RAW CASE TIMESTAMPS
----------------------------------------------------------------------- */
RawT2Cases AS (
  SELECT
    Id AS case_id,
    
    -- Visible CST converted column for the month grouping
    DATETIME(CX_Tier2EscalatedDateTime__c, 'America/Chicago') AS start_dt_cst,
    
    -- Hidden calculation column: uses CURRENT_TIMESTAMP if open to calculate hours elapsed so far
    DATETIME(COALESCE(CX_ResolutionTimestamp__c, ClosedDate, CURRENT_TIMESTAMP()), 'America/Chicago') AS calc_end_dt_cst

  FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.cx_ucrm_case`
  WHERE CX_Tier2EscalatedDateTime__c IS NOT NULL
    AND CX_Tier2EscalatedDateTime__c >= TIMESTAMP('2025-10-01')
    AND CX_BusinessUnit__c = 'FNC CMS'

),

/* -----------------------------------------------------------------------
   STEP 2: UNROLL DATES & DEFINE 7 AM - 7 PM CST WINDOW
----------------------------------------------------------------------- */
ExpandedCaseDates AS (
  SELECT
    r.case_id,
    r.start_dt_cst,
    r.calc_end_dt_cst, 
    d AS current_date,
    -- Define the 7 AM to 7 PM boundary for each specific calendar day
    DATETIME(d, CAST('07:00:00' AS TIME)) AS biz_start,
    DATETIME(d, CAST('19:00:00' AS TIME)) AS biz_end
  FROM RawT2Cases r,
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
   STEP 5: CALCULATE CASE-LEVEL FOLLOW-UPS
----------------------------------------------------------------------- */
CaseLevelMetrics AS (
  SELECT
    r.case_id,
    DATE_TRUNC(DATE(r.start_dt_cst), MONTH) AS escalation_month,
    -- 1,080 minutes = 18 hours. Calculate expected follow-ups per case.
    FLOOR((COALESCE(c.total_business_hours, 0) * 60) / 1080.0) AS expected_follow_ups
  FROM RawT2Cases r
  LEFT JOIN CaseTotalHours c
    ON r.case_id = c.case_id
)

/* -----------------------------------------------------------------------
   FINAL OUTPUT: MONTHLY AGGREGATION
----------------------------------------------------------------------- */
SELECT
  FORMAT_DATE('%Y-%m', escalation_month) AS escalation_month,
  COUNT(case_id) AS total_t2_cases,
  ROUND(AVG(expected_follow_ups), 2) AS avg_expected_follow_ups
FROM CaseLevelMetrics
GROUP BY 1
ORDER BY 1 ASC;