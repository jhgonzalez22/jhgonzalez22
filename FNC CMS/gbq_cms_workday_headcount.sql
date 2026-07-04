WITH FNC_Headcount AS (
    SELECT 
        DATE_TRUNC(rpt_dt, MONTH) AS report_month, 
        rpt_dt AS snapshot_date,                   
        emp_id,
        pref_nm,
        job_ttl,
        job_cd,
        cc_nm,
        mgr_nm,
        
        -- BU CALCULATION FOR FNC
        CASE 
            -- Explicitly forces Brandi Thompson and Triniti Lacy into Ports
            WHEN emp_id IN ('5005169', '5005076') THEN 'Ports'
            
            -- Explicitly forces these specific employees into CMS (Overrides Associate and Manager defaults)
            WHEN emp_id IN ('5003393', '4009821') THEN 'CMS'
            WHEN emp_id = '4011370' THEN 'MANAGER - CMS'            -- Lashonda Wilson
            WHEN emp_id = '5000434' THEN 'Client Account Services'  -- Madison Abbott
            WHEN job_cd = 'NO1CS020' THEN 'Ports'                   -- Associates default to Ports
            ELSE 'CMS'                                              -- Sr Associates and others default to CMS
        END AS BU,

        -- T1 WEIGHT CALCULATION
        CASE 
            -- 0. Block Managers and Team Leads from receiving a weight (Evaluates first)
            WHEN job_ttl LIKE '%Manager%' OR job_ttl LIKE '%Team Lead%' THEN 0.0

            -- 1. Explicit Overrides forcing T1 = 1.0 
            WHEN emp_id IN (
                '5000434', -- Madison Abbott
                '5007287', -- Bronson Pitts
                '5006747', -- Damien Murphy
                '5004154', -- Dequarius Brown
                '5005218', -- James Clements
                '5006889', -- Matthew Meintasis
                '5003710', -- Rhett Unbehagen
                '4009821', -- Sheree Shegog
                '5001369', -- Kristofer Lowery
                '5005833', -- Will Daniel
                '5005169', -- Brandi Thompson
                '5003393', -- Carley Wilkinson
                '5005076'  -- Triniti Lacy
            ) THEN 1.0  
            
            -- 2. Standard T1 Logic (Associates get 1.0)
            WHEN job_cd = 'NO1CS020' THEN 1.0
            ELSE 0.0 
        END AS T1_weight,

        -- T2 WEIGHT CALCULATION
        CASE 
            -- 0. Block Managers and Team Leads from receiving a weight (Evaluates first)
            WHEN job_ttl LIKE '%Manager%' OR job_ttl LIKE '%Team Lead%' THEN 0.0

            -- 1. Explicit Overrides forcing T2 = 1.0 (Specific Sr Associates)
            WHEN emp_id IN (
                '5002541', -- Christopher Gage
                '4009624', -- Melissa Taylor
                '4009687'  -- Robi Milan
            ) THEN 1.0  
            
            -- Everything else is 0
            ELSE 0.0
        END AS T2_weight

    FROM 
        tax_clnt_svcs.emp_info
    WHERE 
        rpt_dt >= '2025-01-01' 
        AND emp_stat = 'Active'
        
        -- Cost Center inclusion list
        AND (
            cc_nm IN ('Platforms - FNC', 'Tier 1 FNC')
            OR emp_id = '5007287'  -- Guarantees Bronson Pitts is included
        )
        
    QUALIFY 
        snapshot_date = MAX(snapshot_date) OVER (PARTITION BY report_month)
)

-- FINAL SELECT AND FILTER
SELECT * FROM FNC_Headcount
WHERE BU = 'CMS'
ORDER BY 
    report_month DESC, 
    pref_nm;