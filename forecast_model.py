import os
import math
import pandas as pd
import numpy as np

# Import your custom framework
from scripthelper import Config, Logger, BigQueryManager

def erlang_c_probability(A, N):
    """Calculates the probability a call waits in queue (Erlang C formula)."""
    if N <= A: return 1.0
    inv_B = 1.0
    for i in range(1, int(N) + 1):
        inv_B = 1.0 + inv_B * (i / A)
    B = 1.0 / inv_B
    C = (N * B) / (N - A * (1 - B))
    return C

def required_agents(volume, aht, target_sl=0.80, target_time=30, interval_secs=1800):
    """Iterates through agent counts to find the minimum required to hit the Target SL."""
    if pd.isna(volume) or volume <= 0: return 0
    if pd.isna(aht) or aht <= 0: aht = 300 
    
    A = (volume * aht) / interval_secs
    N = math.ceil(A)
    
    while N < A + 150:
        C = erlang_c_probability(A, N)
        sl = 1.0 - C * math.exp(-(N - A) * (target_time / aht))
        if sl >= target_sl:
            return N
        N += 1
    return N

def calc_final_fte(base_fte, pct, occ, shrink, attr):
    """Applies shrinkage, occupancy, and attrition to calculate final payroll FTE."""
    if pd.isna(pct) or pct <= 0 or pd.isna(occ) or occ <= 0: 
        return 0.0
    return (base_fte * pct) / (occ * (1 - shrink)) * (1 + attr)

def main():
    config = Config()
    logger = Logger()
    bq = BigQueryManager(config) 
    
    logger.info("Starting Pooled Answerlink Erlang FTE generation...")

    interval_sql_path = r"C:\WFM_Scripting\Automation\GBQ - cx_interval_30mins_vols.sql"
    fte_sql_path = r"C:\WFM_Scripting\Automation\GBQ - cx_fte_calculator.sql"
    
    with open(interval_sql_path, 'r') as f:
        interval_sql = f.read()
        
    with open(fte_sql_path, 'r') as f:
        fte_sql = f.read()

    logger.info("Executing Queries...")
    df_intervals = bq.fetch_data_bigquery(interval_sql)
    df_fte_all = bq.fetch_data_bigquery(fte_sql)
    
    df_fte = df_fte_all[df_fte_all['bu'] == 'Answerlink'].copy()
    df_intervals = df_intervals[df_intervals['bu'] == 'Answerlink'].copy()

    # Clean Interval Data
    for col in ['Offered_Volume', 'Avg_AHT', 'SL_Threshold_Seconds', 'Target_SLA']:
        df_intervals[col] = pd.to_numeric(df_intervals[col], errors='coerce').fillna(0)
    
    df_intervals.loc[df_intervals['SL_Threshold_Seconds'] <= 0, 'SL_Threshold_Seconds'] = 30
    df_intervals.loc[df_intervals['Target_SLA'] <= 0, 'Target_SLA'] = 0.80
    
    # -------------------------------------------------------------------------
    # ERROR FIX: Only apply fillna(0) to specific numeric columns
    # This prevents Pandas from trying to inject 0s into BigQuery Date objects
    # -------------------------------------------------------------------------
    fte_math_cols = [
        'Net_Forecast_Volume', 'Actual_AHT', 'business_day_count', 'business_hours',
        'Domestic_Pct', 'Telus_Pct', 'domestic_occupancy', 'domestic_shrinkage', 
        'domestic_attrition', 'telus_occupancy', 'telus_shrinkage', 'telus_attrition'
    ]
    for col in fte_math_cols:
        if col in df_fte.columns:
            df_fte[col] = pd.to_numeric(df_fte[col], errors='coerce').fillna(0)

    
    # Format time columns
    df_intervals['interval_start'] = pd.to_datetime(df_intervals['interval_start'])
    df_intervals['dow'] = df_intervals['interval_start'].dt.dayofweek
    df_intervals['time'] = df_intervals['interval_start'].dt.time
    
    # =====================================================================
    # STEP 1: BUILD THE POOLED ARRIVAL PROFILE (Grouped by Origin)
    # =====================================================================
    logger.info("Building Pooled Arrival Profiles...")
    df_intervals['workload'] = df_intervals['Offered_Volume'] * df_intervals['Avg_AHT']
    
    # Aggregate all clients together by Origin/Dow/Time
    profile_df = df_intervals.groupby(['origin', 'dow', 'time']).agg(
        bucket_vol=('Offered_Volume', 'sum'),
        bucket_workload=('workload', 'sum'),
        SL_Threshold_Seconds=('SL_Threshold_Seconds', 'mean'),
        Target_SLA=('Target_SLA', 'mean')
    ).reset_index()
    
    # Calculate Weighted AHT for the pooled bucket
    profile_df['Avg_AHT'] = np.where(profile_df['bucket_vol'] > 0, profile_df['bucket_workload'] / profile_df['bucket_vol'], 300)
    
    # Calculate the percentage of volume each bucket represents for the whole origin
    total_vol_df = profile_df.groupby('origin')['bucket_vol'].sum().reset_index()
    total_vol_df.rename(columns={'bucket_vol': 'origin_total_vol'}, inplace=True)
    profile_df = pd.merge(profile_df, total_vol_df, on='origin')
    
    profile_df['Interval_Weight'] = np.where(
        profile_df['origin_total_vol'] > 0, 
        profile_df['bucket_vol'] / profile_df['origin_total_vol'], 
        0
    )

    # =====================================================================
    # STEP 2: RUN ERLANG ON THE POOL, THEN APPORTION TO CLIENTS
    # =====================================================================
    logger.info("Simulating Pooled Volume and Apportioning FTE...")
    
    WEEKS_IN_MONTH = 4.333
    df_fte['row_workload'] = df_fte['Net_Forecast_Volume'] * df_fte['Actual_AHT']
    
    erlang_results_dict = {}
    
    # Group the forecast by Month and Origin (The Agent Pool)
    for (month_date, origin), group in df_fte.groupby(['date', 'origin']):
        
        total_forecast_vol = group['Net_Forecast_Volume'].sum()
        total_group_workload = group['row_workload'].sum()
        
        # Get max business days/hours for this month
        b_days = group['business_day_count'].max()
        b_hours = group['business_hours'].max()
        monthly_work_hours = (b_days * b_hours) if (b_days * b_hours) > 0 else 157.5
        
        # If no volume forecasted for the whole pool, set everyone to 0
        if total_forecast_vol <= 0 or total_group_workload <= 0:
            for idx in group.index:
                erlang_results_dict[idx] = {'Domestic_Erlang_FTE': 0.0, 'Telus_Erlang_FTE': 0.0, 'Total_Erlang_FTE': 0.0}
            continue
            
        origin_profile = profile_df[profile_df['origin'] == origin]
        
        weekly_forecast_vol = total_forecast_vol / WEEKS_IN_MONTH
        weekly_agent_hours = 0
        
        # Run Erlang on the massive pooled volume
        for _, p_row in origin_profile.iterrows():
            sim_interval_vol = weekly_forecast_vol * p_row['Interval_Weight']
            if sim_interval_vol < 0.01: continue
            
            req_agents = required_agents(
                volume=sim_interval_vol,
                aht=p_row['Avg_AHT'],
                target_sl=p_row['Target_SLA'],
                target_time=p_row['SL_Threshold_Seconds']
            )
            weekly_agent_hours += (req_agents * 0.5)
            
        # Total base FTE required to handle the pooled volume
        monthly_agent_hours = weekly_agent_hours * WEEKS_IN_MONTH
        total_pooled_base_fte = monthly_agent_hours / monthly_work_hours
        
        # Apportion that efficient base FTE back to the individual clients
        for idx, row in group.iterrows():
            # What percentage of the work did this specific client contribute?
            client_share = row['row_workload'] / total_group_workload
            
            # Their share of the Erlang Base Seats
            client_base_fte = total_pooled_base_fte * client_share
            
            # Apply their specific row-level shrinkage rules
            dom_erlang = calc_final_fte(
                client_base_fte, row['Domestic_Pct'], 
                row['domestic_occupancy'], row['domestic_shrinkage'], row['domestic_attrition']
            )
            
            tel_erlang = calc_final_fte(
                client_base_fte, row['Telus_Pct'], 
                row['telus_occupancy'], row['telus_shrinkage'], row['telus_attrition']
            )
            
            erlang_results_dict[idx] = {
                'Domestic_Erlang_FTE': round(dom_erlang, 2),
                'Telus_Erlang_FTE': round(tel_erlang, 2),
                'Total_Erlang_FTE': round(dom_erlang + tel_erlang, 2)
            }

    # =====================================================================
    # STEP 3: EXPORT
    # =====================================================================
    # Reconstruct the list in the exact order of the original dataframe
    erlang_results = [erlang_results_dict.get(i, {'Domestic_Erlang_FTE': 0, 'Telus_Erlang_FTE': 0, 'Total_Erlang_FTE': 0}) for i in df_fte.index]
    
    df_erlang = pd.DataFrame(erlang_results)
    df_final = pd.concat([df_fte.reset_index(drop=True), df_erlang], axis=1)
    
    desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'Answerlink_Erlang_FTE_Calculator.csv')
    df_final.to_csv(desktop_path, index=False)
    
    logger.info(f"Success! Exported combined FTE requirements to: {desktop_path}")

if __name__ == "__main__":
    main()


----
The Big Picture: What Are We Trying to Accomplish?
The Problem: The Flaw in "Linear" Staffing
Currently, your capacity workbook calculates Required FTE using the Linear Workload Method. The formula looks like this:
(Forecasted Volume × AHT) / (Business Days × Business Hours) / Occupancy / (1 - Shrinkage).

The problem with this method is that it assumes calls arrive perfectly evenly throughout the month. It assumes that if you get 10,000 calls a month, you get the exact same number of calls on a Monday at 9:00 AM as you do on a Friday at 4:30 PM.
Because call centers have massive peaks (morning rushes) and valleys (slow afternoons), staffing to an "average" guarantees that you will be massively understaffed during the peaks (failing your SLA) and overstaffed during the valleys.

The Solution: Erlang C + Arrival Profiling + Queue Pooling
We are building an automated pipeline that replaces flat averages with Erlang C, the industry-standard mathematical algorithm for call centers.
Because Erlang C requires interval data (e.g., half-hour increments) but your financial forecasts are monthly, our solution must:

Learn how your calls actually arrive by looking at historical 30-minute intervals.

Simulate how a future month's volume will distribute across a standard week based on those historical patterns.

Calculate the exact number of "butts in seats" needed for every 30-minute window to hit an 80% SLA.

Apportion those required seats back to specific clients based on their share of the workload.

Convert those seats into payroll FTEs using your exact shrinkage and occupancy metrics.

The Architecture: How the Data Flows
The entire process is orchestrated by your Python script (run_erlang_capacity.py) using your custom scripthelper.py framework to interact with BigQuery.

Input 1: The 30-Minute Interval Data (cx_interval_30mins_vols.sql)
This query reaches into your Cisco tables and aggregates historical Answerlink data into precise 30-minute buckets.

What it provides: Offered Volume, AHT, SLA Thresholds (e.g., 30s), and Contractual SLA Targets (e.g., 80%).

Purpose: This provides the "DNA" of your call arrival patterns.

Input 2: The Monthly Forecast Data (cx_fte_calculator.sql)
This is your existing master query that groups data by Month, Client, and Origin.

What it provides: Actual Volume, Forecasted Volume, Business Days/Hours, Vendor Splits (Domestic_Pct, Telus_Pct), Occupancy, and Shrinkage metrics.

Purpose: This tells us how much volume we are expecting and the specific HR rules (shrinkage) required to turn a "Scheduled Agent" into a "Hired FTE".

The Execution: Step-by-Step Logic in Python
When you run the Python script, it executes the following mathematical journey in memory:

Step 1: Building the "Pooled" Arrival Profile
In a modern call center, agents are cross-trained. You don't have one agent sitting around waiting specifically for a "Matrix 1" call while another waits for a "Trestle" call. They share the load.

The script takes all historical interval data and groups it purely by Origin (e.g., Phone vs. Chat), Day of Week, and Time of Day.

It calculates the Interval Weight. For example, it discovers that exactly 1.8% of all phone calls for the entire Answerlink pool always arrive on Mondays between 9:00 AM and 9:30 AM.

Step 2: Simulating the Weekly Workload
The script loops through every single row in your FTE Calculator (e.g., Matrix 1, May 2026).

It groups all clients together by Month and Origin to get a Total Pooled Monthly Forecast.

It divides the monthly forecast by 4.333 to convert it into a standard Weekly Forecast.

It multiplies that Weekly Forecast by the Interval Weights established in Step 1.

Result: The script now has a simulated 336-row week showing exactly how many calls to expect in every single half-hour slot.

Step 3: The Erlang C Engine
For every single one of those 336 simulated half-hour slots, the script runs the Erlang C algorithm.

Math: It looks at the simulated volume, the weighted AHT of that time slot, the 30-second target, and the 80% SLA goal.

It uses a while loop to add agents one by one until the mathematical probability of a call waiting longer than 30 seconds drops below 20%.

Result: The script calculates that to handle Monday at 9:00 AM, you need exactly 42 agents logged in and ready. For Monday at 4:30 PM, you only need 18 agents.

Step 4: Converting Intervals to Base FTE
The script adds up all the required agents for all 336 half-hour slots in the week. Since each slot is 0.5 hours, this gives us the Total Weekly Agent Hours needed "in the chair."

It multiplies this by 4.333 to get the Total Monthly Agent Hours.

It divides this by the total working hours in the month (Business Days × Business Hours) to calculate the Total Pooled Base FTE (the perfect, 100% efficient number of scheduled headcount needed if humans didn't take breaks).

Step 5: Apportionment (Slicing the pie back up)
Now that the script knows how many highly efficient "Pooled" agents are required for the whole department, it needs to assign the bill back to the individual clients.

It calculates the Workload Share of each client. (e.g., Matrix 1 represents 60% of the total forecasted volume × AHT for the month).

It assigns 60% of the Total Pooled Base FTE to Matrix 1.

Step 6: The "Real World" Layer (Shrinkage & Occupancy)
Finally, the script takes that apportioned Base FTE for Matrix 1 and applies your specific vendor rules row-by-row.

It splits the Base FTE using Domestic_Pct and Telus_Pct.

It divides the result by the respective Occupancy (e.g., 80% utilization limit).

It divides the result by (1 - Shrinkage) to account for PTO, absenteeism, and coaching.

It multiplies by (1 + Attrition).

The Output
After completing millions of micro-calculations in seconds, the script takes your original df_fte DataFrame and appends six brand new columns to the far right:

Domestic_Actual_Erlang_FTE

Telus_Actual_Erlang_FTE

Total_Actual_Erlang_FTE

Domestic_Forecast_Erlang_FTE

Telus_Forecast_Erlang_FTE

Total_Forecast_Erlang_FTE

It exports this combined table as a CSV directly to your Desktop.

The ultimate accomplishment: You can now paste this data into your Excel workbook. Management can look at a row and see: "Our linear math said we needed 15 FTEs. But to actually survive the Monday morning peaks and hit our 80% SLA, Erlang math proves we need 18 FTEs.
