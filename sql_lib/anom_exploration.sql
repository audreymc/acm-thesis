with before_aft as (
	select bar.*
	from before_after_results bar
	where bar.open_pr is not null
	and bar.close_pr is not null
	and bar.dlyhigh is not null
	and bar.dlylow is not null
	and bar.dlynumtrd is not null
),

ev_vals as (
	select *
	from ev_results er 
	where high_score is not null
	and low_score is not null
),

intraday_anomalies as (
	select ha.ticker, 
	       ha.current_date, 
	       ha.before_date, 
	       ha.after_date, 
	       ha.vix_quantile, 
	       ha.vlty_quantile,
	       ha.before_pct_diff, 
	       ha.after_pct_diff,
	       ha.anomaly_fl 
	from high_anomalies ha 
	union all
	select la.ticker, 
	       la.current_date, 
	       la.before_date, 
	       la.after_date, 
	       la.vix_quantile, 
	       la.vlty_quantile,
	       la.before_pct_diff, 
	       la.after_pct_diff,
	       la.anomaly_fl 
	from low_anomalies la 
)

select b.*, 
       e.high_extreme, 
       e.low_extreme, 
       (e.high_score * 100) as high_score, 
       (e.low_score * 100) as low_score,
       i.before_pct_diff,
       i.after_pct_diff,
       i.anomaly_fl,
       i.vlty_quantile
from before_aft b
join ev_vals e
on (b.ticker, b.current_date) = (e.ticker, e.current_date)
left join intraday_anomalies i
on (b.ticker, b.current_date) = (i.ticker, i.current_date)
order by e.high_score


