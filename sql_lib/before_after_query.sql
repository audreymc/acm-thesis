@set start_dt = '2023-07-01'
@set end_dt = '2023-07-25'
@set current_dt = '2023-07-17'

WITH price_details AS (
    SELECT  b.ticker,
            a.permno,
            a.dlycaldt,
            a.dlyopen,
            coalesce(a.dlyclose, a.dlyprc) AS dlyclose,
            a.dlyhigh,
            a.dlylow,
            a.dlynumtrd,
            a.dlyvol,
            (a.dlyopen - a.dlyclose) as daily_return
    FROM crsp.dsf_v2 a
    JOIN crsp.dsenames AS b
    ON a.permno = b.permno
    AND b.ticker IN (${symbol_lst})
    AND a.dlycaldt BETWEEN date(${start_dt}) AND date(${end_dt})
    AND date(${current_dt}) BETWEEN b.namedt AND b.nameendt
),

days_before AS (
    SELECT ticker,
           MIN(dlycaldt) AS before_date,
           AVG(dlyopen) AS before_avg_open,
           AVG(dlyclose) AS before_avg_close,
           AVG(dlyhigh) AS before_avg_dlyhigh,
           AVG(dlylow) AS before_avg_dlylow,
           AVG(dlynumtrd) AS before_avg_dlynumtrd,
           AVG(dlyvol) AS before_avg_dlyvol
    FROM price_details
    WHERE dlycaldt BETWEEN date(${start_dt}) AND (date(${current_dt}) - INTERVAL '1 day')
    GROUP BY ticker
),

current_days AS (
    SELECT ticker,
           MIN(dlycaldt) AS current_date,
           AVG(dlyopen) AS open_pr,
           AVG(dlyclose) AS close_pr,
           AVG(dlyhigh) AS dlyhigh,
           AVG(dlylow) AS dlylow,
           AVG(dlynumtrd) AS dlynumtrd,
           AVG(dlyvol) AS dlyvol
    FROM price_details
    WHERE dlycaldt = date(${current_dt})
    GROUP BY ticker
),

days_after AS (
    SELECT ticker,
           MAX(dlycaldt) AS after_date,
           AVG(dlyopen) AS after_avg_open,
           AVG(dlyclose) AS after_avg_close,
           AVG(dlyhigh) AS after_avg_dlyhigh,
           AVG(dlylow) AS after_avg_dlylow,
           AVG(dlynumtrd) AS after_avg_dlynumtrd,
           AVG(dlyvol) AS after_avg_dlyvol
    FROM price_details
    WHERE dlycaldt BETWEEN (date(${current_dt}) + INTERVAL '1 day') AND date(${end_dt})
    GROUP BY ticker
),

tic_deets AS (
	SELECT tic, gvkey
	FROM comp.secm
	WHERE tic IN (${symbol_lst})
	and datadate < date(${current_dt})
	ORDER BY datadate DESC
	LIMIT 1
), 

sector_deets AS (
	SELECT td.tic, c.gind, c.gsector, c.gsubind, c.idbflag 
	FROM tic_deets td
	LEFT JOIN comp.company c
	ON td.gvkey = c.gvkey 
)


SELECT coalesce(c.ticker, a.ticker, b.ticker) AS ticker,
       c.current_date,
       COALESCE(b.before_date, TO_DATE(${start_dt}, 'YYY-MM-DD')) AS before_date,
       COALESCE(a.after_date, TO_DATE(${end_dt}, 'YYYY-MM-DD')) AS after_date,
       c.open_pr,
       c.close_pr,
       c.dlyhigh,
       c.dlylow,
       c.dlynumtrd,
       c.dlyvol,
       b.before_avg_open,
       b.before_avg_close,
       b.before_avg_dlyhigh,
       b.before_avg_dlylow,
       b.before_avg_dlynumtrd,
       b.before_avg_dlyvol,
       a.after_avg_open,
       a.after_avg_close,
       a.after_avg_dlyhigh,
       a.after_avg_dlylow,
       a.after_avg_dlynumtrd,
       a.after_avg_dlyvol,
       (SELECT stddev(daily_return) FROM price_details) AS vlty_estimate,
       sd.gind, 
       sd.gsector, 
       sd.gsubind, 
       sd.idbflag 
FROM current_days c
FULL OUTER JOIN days_before b
ON (c.ticker = b.ticker)
FULL OUTER JOIN days_after a
ON (c.ticker = a.ticker)
left join sector_deets sd
on (c.ticker = sd.tic)
