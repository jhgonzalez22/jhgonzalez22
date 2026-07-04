/* =============================================================================
   Mercury — T2 Escalation Ratio + KPI Validation (Transactional Model)
   PURPOSE: T2 FTE Capacity Modeling & Executive Reporting
   TIMEZONE: Enforced CST ('America/Chicago') for accurate month-end boundaries
============================================================================= */

WITH

/* -----------------------------------------------------------------------
   1. CONFIGURATION
----------------------------------------------------------------------- */
config AS (
  SELECT DATE('2026-01-01') AS data_trust_start
),

/* -----------------------------------------------------------------------
   2. DAILY AGGREGATION 
----------------------------------------------------------------------- */
daily AS (
  SELECT
    dt,

    -- [FLOW METRICS] SUMmed for volume
    SUM(cases_opened)              AS contacts_opened,
    SUM(cases_cancelled)           AS contacts_cancelled,
    SUM(t2_esc)                    AS t2_arrivals,       
    SUM(t2_resolved_cnt)           AS t2_resolved,  

    -- [SLA METRICS] Raw counts needed for First Response % and Resolve Rate
    SUM(first_res_exp)             AS first_res_exp,
    SUM(first_res_in_sla)          AS first_res_in_sla,
    SUM(cases_solved)              AS cases_solved,
    SUM(cases_solved_oos)          AS cases_solved_oos,

    -- [STOCK METRICS / WIP] Averaged for inventory
    SUM(eod_pend_t2_esc)           AS t2_wip,            
    SUM(eod_wip)                   AS total_wip,         

    -- [AGE AGGREGATE]
    SUM(eod_t2_esc_bus_days_open)  AS t2_age_days_agg    

  FROM `clgx-taxbi-reg-bf03.tax_clnt_svcs.cx_ucrm_daily_kpi`
  WHERE
    bus_unit = 'Platforms'
    AND sub_bus_unit = 'CMS'
    -- Timeframe starts Jan 1, 2025
    AND dt >= '2025-01-01'
    -- CST Enforced: Drops current month based strictly on America/Chicago time
    AND dt < DATE_TRUNC(CURRENT_DATE('America/Chicago'), MONTH)
  GROUP BY dt
),

/* -----------------------------------------------------------------------
   3. MONTHLY ROLLUP
----------------------------------------------------------------------- */
monthly AS (
  SELECT
    DATE_TRUNC(d.dt, MONTH)             AS month_start,

    SUM(d.contacts_opened)              AS contacts_opened,
    SUM(d.contacts_cancelled)           AS contacts_cancelled,
    SUM(d.contacts_opened)
      - SUM(d.contacts_cancelled)       AS contacts_net,      

    SUM(d.t2_arrivals)                  AS t2_cases,
    SUM(d.t2_resolved)                  AS t2_resolved,

    -- SLA Aggregations
    SUM(d.first_res_exp)                AS first_res_exp,
    SUM(d.first_res_in_sla)             AS first_res_in_sla,
    SUM(d.cases_solved)                 AS cases_solved,
    -- Solved within SLA = Total Solved - Solved Out Of Standard
    SUM(d.cases_solved) - SUM(d.cases_solved_oos) AS cases_solved_in_sla,

    ROUND(AVG(d.t2_wip), 2)             AS t2_avg_inventory,
    ROUND(AVG(d.total_wip), 2)          AS total_avg_inventory,

    ARRAY_AGG(d.t2_wip ORDER BY d.dt DESC LIMIT 1)[OFFSET(0)]
                                        AS t2_month_end_inventory,

    ROUND(
      SAFE_DIVIDE(AVG(d.t2_age_days_agg), NULLIF(AVG(d.t2_wip), 0)), 
      2
    )                                   AS t2_avg_days_open,    

    -- Using MAX() here bypasses the JDBC grouping error entirely
    DATE_TRUNC(MAX(d.dt), MONTH) >= DATE_TRUNC(MAX(c.data_trust_start), MONTH)
                                        AS is_trusted

  FROM daily d
  CROSS JOIN config c
  GROUP BY month_start
),

/* -----------------------------------------------------------------------
   4. RECONCILIATION GAP DIAGNOSTIC
----------------------------------------------------------------------- */
with_gap AS (
  SELECT
    *,
    LAG(t2_month_end_inventory) OVER (ORDER BY month_start) AS prev_month_end_inventory,
    (
      LAG(t2_month_end_inventory) OVER (ORDER BY month_start)
      + t2_cases                   
      - t2_month_end_inventory
    ) AS implied_outflow,
    (t2_resolved + 0)              AS reported_outflow    
  FROM monthly
)

/* -----------------------------------------------------------------------
   5. FINAL OUTPUT FORMATTING
----------------------------------------------------------------------- */
SELECT

  FORMAT_DATE('%Y-%m-%d', month_start)             AS call_month,

  contacts_opened                                  AS offered_cases,
  contacts_cancelled                               AS cancelled_cases,
  contacts_net                                     AS non_cancelled_cases,

  t2_cases                                         AS t2_esc_non_cancelled,
  t2_resolved                                      AS t2_resolved_cnt,

  ROUND(
    SAFE_DIVIDE(t2_cases, NULLIF(contacts_net, 0)), 
    4
  )                                                AS t2_esc_ratio,

  -- New SLA Metrics Added Here
  ROUND(
    SAFE_DIVIDE(first_res_in_sla, NULLIF(first_res_exp, 0)), 
    4
  )                                                AS first_response_pct,
  
  ROUND(
    SAFE_DIVIDE(cases_solved_in_sla, NULLIF(cases_solved, 0)), 
    4
  )                                                AS resolve_rate,

  t2_avg_inventory,
  t2_avg_days_open,
  ROUND(
    SAFE_DIVIDE(t2_resolved, NULLIF(t2_cases, 0)), 
    4
  )                                                AS t2_closure_rate,

  is_trusted,

  ROUND(
    implied_outflow - reported_outflow, 
    2
  )                                                AS reconciliation_gap

FROM with_gap
ORDER BY call_month;