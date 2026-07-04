/* ==============================================================================================
Query Name: Unified Cisco Historical Master (FNC - CMS | Phone Only)
Author: jhonnatan gonzalez
Description: Aggregates historical Phone data for FNC - CMS
             with detailed calendar metrics (business days, holidays).
             Incorporates Cisco UCCE best practices for Slowly Changing Dimensions (SCD),
             Media Routing Domains, and safe arithmetic to prevent division-by-zero.
==============================================================================================
*/

WITH MonthCalendar AS (
    -- =======================================================================================
    -- CTE 1: Scaffolding historical months
    -- =======================================================================================
    SELECT DISTINCT
        DATE_TRUNC(month_start, MONTH) AS date
    FROM UNNEST(GENERATE_DATE_ARRAY(
        DATE('2024-01-01'), 
        DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 DAY), 
        INTERVAL 1 MONTH
    )) AS month_start
),

HolidayCalendar AS (
    -- =======================================================================================
    -- CTE 2: Holiday Aggregation
    -- =======================================================================================
    SELECT
        DATE_TRUNC(holdate, MONTH) AS date,
        COUNT(*) AS holiday_count
    FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.holiday_calendar`
    WHERE holdate >= DATE('2024-01-01')
    GROUP BY 1
),

CalendarMetrics AS (
    -- =======================================================================================
    -- CTE 3: Advanced Calendar Metrics (Business Days Calculation)
    -- =======================================================================================
    SELECT
        mc.date,
        COUNT(day) AS total_days_in_month,
        (COUNTIF(EXTRACT(DAYOFWEEK FROM day) NOT IN (1, 7)) - COALESCE(hc.holiday_count, 0)) AS business_day_count,
        COALESCE(hc.holiday_count, 0) AS holiday_count
    FROM
        MonthCalendar AS mc
    CROSS JOIN UNNEST(GENERATE_DATE_ARRAY(mc.date, LAST_DAY(mc.date), INTERVAL 1 DAY)) AS day
    LEFT JOIN HolidayCalendar hc ON mc.date = hc.date
    GROUP BY 1, hc.holiday_count
),

CiscoHistoricalData AS (
    -- =======================================================================================
    -- CTE 4: Cisco UCCE Core Aggregation
    -- =======================================================================================
    SELECT
        DATE_TRUNC(DATE(sgi.DateTime), MONTH) AS date,
        ck.`Business Unit` AS bu,
        ck.Client AS client,
        ck.Id AS client_id,
        ck.`Groups` AS `groups`,
        
        INITCAP(CASE
          WHEN sg.MRDomainID IN (5000, 5003) THEN 'Chat'
          WHEN sg.MRDomainID = 1 OR sg.MRDomainID IS NULL THEN 'Phone'
          ELSE 'Other'
        END) AS Origin,
        
        SUM(COALESCE(sgi.CallsHandled, 0)) AS Total_Handled,
        SUM(COALESCE(sgi.ServiceLevelCalls, 0)) AS Calls_Handled_Within_SL,
        SUM(COALESCE(sgi.CallsHandled, 0) + COALESCE(sgi.RouterCallsAbandQ, 0)) AS Total_Offered,
        SUM(COALESCE(sgi.HandledCallsTime, 0)) AS Total_Handle_Time
        
    FROM
        `clgx-taxbi-reg-bf03.taxbibt.dtp_skill_group_interval` AS sgi
        
    JOIN `clgx-taxbi-reg-bf03.taxbibt.dtp_skill_group` AS sg 
        ON sgi.SkillTargetID = sg.SkillTargetID 
        AND sg.version_expired_date IS NULL 
        
    JOIN `clgx-taxbi-reg-bf03.tax_clnt_svcs.cc_skillgroupkey` AS sgk 
        ON sg.SkillTargetID = sgk.SkillGroupID
    JOIN `clgx-taxbi-reg-bf03.tax_clnt_svcs.cc_client_key` AS ck 
        ON sgk.ClientID = ck.Id
        
    WHERE
        DATE(sgi.DateTime) >= '2024-01-01'
        AND DATE(sgi.DateTime) < DATE_TRUNC(CURRENT_DATE(), MONTH)
        AND sg.EnterpriseName NOT LIKE '%Mercury%'
        -- Keeping the BU filter as requested in the original script
        AND ck.`Business Unit` = 'Platforms'  
    GROUP BY 1, 2, 3, 4, 5, 6
)

-- ===========================================================================================
-- FINAL SELECT: Bringing it all together
-- ===========================================================================================
SELECT
    chd.date,
    chd.bu,
    chd.client,
    chd.client_id,
    chd.`groups`,
    chd.Origin,
    
    -- Calendar Metrics
    cm.total_days_in_month,
    cm.business_day_count,
    cm.holiday_count,

    -- Volume & Handling Metrics
    chd.Total_Offered,
    chd.Total_Handled,
    chd.Calls_Handled_Within_SL,
    chd.Total_Handle_Time,
    
    -- SERVICE LEVEL PERCENTAGE
    ROUND(COALESCE(SAFE_DIVIDE(chd.Calls_Handled_Within_SL, chd.Total_Handled), 0) * 100, 2) AS Service_Level_Percent,
    
    -- AVERAGE HANDLE TIME (AHT)
    ROUND(COALESCE(chd.Total_Handle_Time / NULLIF(chd.Total_Handled, 0), 0), 2) AS AHT,

    -- CALLS OFFERED PER BUSINESS DAY
    ROUND(chd.Total_Offered / NULLIF(cm.business_day_count, 0), 2) AS Offered_Per_Business_Day

FROM
    CiscoHistoricalData chd
LEFT JOIN CalendarMetrics cm ON chd.date = cm.date

WHERE 
    -- Focus ONLY on FNC - CMS
    (chd.client_id = 'FNC - CMS' OR chd.client = 'FNC - CMS')
    
    -- Focus ONLY on Phone
    AND chd.Origin = 'Phone'

ORDER BY 
    date DESC, 
    client_id;