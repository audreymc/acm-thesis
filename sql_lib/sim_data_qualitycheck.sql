WITH percentiles AS (
    SELECT 
        id,
        volatility,
        NTILE(100) OVER (PARTITION BY id ORDER BY volatility) AS perc
    FROM sim_data_distribution_shift_700
),
agg AS (
    SELECT 
        id,
        MAX(CASE WHEN perc = 25 THEN volatility END) AS p25,
        MAX(CASE WHEN perc = 50 THEN volatility END) AS p50,
        MAX(CASE WHEN perc = 70 THEN volatility END) AS p70
    FROM percentiles
    WHERE perc IN (25, 50, 70)
    GROUP BY id
)
SELECT id
FROM agg
WHERE ABS(p25 - p50) <= 0.01
  AND ABS(p25 - p70) <= 0.01
  AND ABS(p50 - p70) <= 0.01;
ORDER BY id