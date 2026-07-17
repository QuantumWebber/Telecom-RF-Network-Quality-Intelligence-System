

-- 1. TOWER-LEVEL DEGRADATION RANKING (Window Function)
-- Which towers have the worst average signal quality?
SELECT
    tower_id,
    ROUND(AVG(rsrp_proxy)::numeric, 2) AS avg_rsrp,
    ROUND(AVG(sinr_proxy)::numeric, 2) AS avg_sinr,
    COUNT(*) AS total_readings,
    RANK() OVER (ORDER BY AVG(rsrp_proxy) ASC) AS worst_signal_rank
FROM fact_rf_readings
GROUP BY tower_id
ORDER BY worst_signal_rank
LIMIT 20;


-- 2. DEGRADATION CAUSE DISTRIBUTION BY OPERATOR (CTE)
-- Which operator suffers most from which degradation cause?
WITH operator_degradation AS (
    SELECT
        o.operator,
        f.degradation_cause,
        COUNT(*) AS cause_count
    FROM fact_rf_readings f
    JOIN dim_operator o ON f.operator_id = o.operator_id
    GROUP BY o.operator, f.degradation_cause
),
operator_totals AS (
    SELECT operator, SUM(cause_count) AS total
    FROM operator_degradation
    GROUP BY operator
)
SELECT
    od.operator,
    od.degradation_cause,
    od.cause_count,
    ROUND(100.0 * od.cause_count / ot.total, 2) AS pct_of_operator_issues
FROM operator_degradation od
JOIN operator_totals ot ON od.operator = ot.operator
ORDER BY od.operator, od.cause_count DESC;


-- 3. HANDOVER RATE VS DEGRADATION CORRELATION
-- Do towers with high handover rates see more congestion?
SELECT
    CASE
        WHEN handover_rate < 0.3 THEN 'Low (<30%)'
        WHEN handover_rate < 0.6 THEN 'Medium (30-60%)'
        ELSE 'High (>60%)'
    END AS handover_bucket,
    degradation_cause,
    COUNT(*) AS occurrences,
    ROUND(AVG(prb_utilization)::numeric, 2) AS avg_prb_utilization
FROM fact_rf_readings
GROUP BY handover_bucket, degradation_cause
ORDER BY handover_bucket, occurrences DESC;


-- 4. PRB UTILIZATION PERCENTILE RANKING (Window Function)
-- Identify the top 10% most congested readings
SELECT
    reading_id,
    tower_id,
    prb_utilization,
    call_duration,
    NTILE(10) OVER (ORDER BY prb_utilization DESC) AS congestion_decile
FROM fact_rf_readings
WHERE prb_utilization IS NOT NULL
ORDER BY prb_utilization DESC
LIMIT 50;


-- 5. ENVIRONMENT-WISE SIGNAL QUALITY COMPARISON
-- Urban vs home vs open — where is signal worst?
SELECT
    e.environment_type,
    COUNT(*) AS total_calls,
    ROUND(AVG(f.rsrp_proxy)::numeric, 2) AS avg_rsrp,
    ROUND(AVG(f.sinr_proxy)::numeric, 2) AS avg_sinr,
    ROUND(AVG(f.distance_to_tower)::numeric, 2) AS avg_distance_km,
    MODE() WITHIN GROUP (ORDER BY f.degradation_cause) AS most_common_cause
FROM fact_rf_readings f
JOIN dim_environment e ON f.env_id = e.env_id
GROUP BY e.environment_type
ORDER BY avg_rsrp ASC;


-- 6. DAY-OF-WEEK CALL QUALITY TREND (Window Function + Date Dimension)
-- Rolling comparison of signal quality across weekdays
SELECT
    d.day_of_week,
    COUNT(*) AS total_readings,
    ROUND(AVG(f.sinr_proxy)::numeric, 2) AS avg_sinr,
    ROUND(AVG(f.prb_utilization)::numeric, 2) AS avg_prb,
    LAG(ROUND(AVG(f.sinr_proxy)::numeric, 2)) OVER (ORDER BY MIN(d.date_id)) AS prev_day_sinr
FROM fact_rf_readings f
JOIN dim_date d ON f.date_id = d.date_id
GROUP BY d.day_of_week
ORDER BY MIN(d.date_id);


-- 7. TOP DEGRADATION-PRONE CIRCLES (Geographic Analysis)
-- Which telecom circles need the most network intervention?
SELECT
    o.circle,
    o.operator,
    COUNT(*) AS total_issues,
    SUM(CASE WHEN f.degradation_cause = 'interference' THEN 1 ELSE 0 END) AS interference_issues,
    SUM(CASE WHEN f.degradation_cause = 'congestion' THEN 1 ELSE 0 END) AS congestion_issues,
    SUM(CASE WHEN f.degradation_cause = 'distance' THEN 1 ELSE 0 END) AS distance_issues,
    SUM(CASE WHEN f.degradation_cause = 'hardware_fault' THEN 1 ELSE 0 END) AS hardware_issues
FROM fact_rf_readings f
JOIN dim_operator o ON f.operator_id = o.operator_id
GROUP BY o.circle, o.operator
ORDER BY total_issues DESC
LIMIT 15;