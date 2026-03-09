WITH
  Months AS (
    -- Step 1: Create a list of the first day of each month within the desired date range.
    SELECT DISTINCT
      month_start
    FROM
      UNNEST(
        GENERATE_DATE_ARRAY(
          DATE('2023-01-01'),
          -- THIS LINE IS UPDATED to go 15 months into the future
          DATE_TRUNC(DATE_ADD(CURRENT_DATE(), INTERVAL 15 MONTH), MONTH),
          INTERVAL 1 MONTH
        )
      ) AS month_start
  ),
  HolidayCalendar AS (
    -- Step 2: Count the number of holidays occurring in each month.
    SELECT
      DATE_TRUNC(holdate, MONTH) AS month_start,
      COUNT(*) AS holiday_count
    FROM
      `clgx-taxbi-reg-bf03.tax_clnt_svcs.holiday_calendar`
    WHERE
      -- This start date is fine, the join will handle the range.
      holdate >= DATE('2023-01-01') 
    GROUP BY
      month_start
  ),
  MonthlyBreakdown AS (
    -- Step 3: For each month, calculate the total days and the number of weekdays (Mon-Fri).
    SELECT
      m.month_start,
      DATE_DIFF(LAST_DAY(m.month_start, MONTH), m.month_start, DAY) + 1 AS total_days,
      COUNTIF(EXTRACT(DAYOFWEEK FROM day) NOT IN (1, 7)) AS weekday_count,
      COUNTIF(EXTRACT(DAYOFWEEK FROM day) IN (1, 7)) AS weekend_day_count
    FROM
      Months AS m,
      UNNEST(GENERATE_DATE_ARRAY(m.month_start, LAST_DAY(m.month_start, MONTH))) AS day
    GROUP BY
      m.month_start
  )
-- Final Step: Combine the monthly breakdown with the holiday counts to calculate business days.
SELECT
  mb.month_start,
  mb.total_days,
  mb.weekday_count,
  mb.weekend_day_count,
  COALESCE(hc.holiday_count, 0) AS holiday_count,
  -- Business days are defined as weekdays minus any holidays that fall on those days.
  (mb.weekday_count - COALESCE(hc.holiday_count, 0)) AS business_day_count
FROM
  MonthlyBreakdown AS mb
  LEFT JOIN HolidayCalendar AS hc ON mb.month_start = hc.month_start
ORDER BY
  mb.month_start;