-- NOTE: This is the anomaly identification query utilized to create the initial sample set
with extreme_spreads as (
	select ticker,
	       eir.current_date,
	       eir.before_date,
	       eir.after_date,
	       highlow_extreme,
	       highlow_score,
	       avg_dlynumtrd,
	       eir.dlynumtrd,
	       highlow_pvalue,
	       high_pvalue,
	       high_score,
	       eir.openclose_extreme,
	       eir.openclose_score,
	       eir.openclose_pvalue
	from ev_interday_results eir 
	where highlow_score is not null
	and highlow_pvalue > 0.05
	and avg_dlynumtrd > 500
	and highlow_score < 0.1
	and high_pvalue > 0.05
	and high_score < 0.5
	and eir.openclose_score > 0.4
	and eir.openclose_pvalue > 0.05
),

open_cl as (
	select bar.ticker, 
	       bar.current_date,
	       bar.open_pr, 
	       bar.close_pr, 
	       abs(bar.open_pr - bar.close_pr) as open_close_diff
	from before_after_results bar 
),

interday_w_closeopen as (
	select es.*, oc.open_pr, oc.close_pr, oc.open_close_diff
	from extreme_spreads es
	left join open_cl oc
	on (es.ticker, es.current_date) = (oc.ticker, oc.current_date)
),

intraday_extremes as (
	select eir.*
	from ev_intraday_results eir 
	where eir.high_extreme is not null
	and eir.high_score is not null
	and eir.high_score < 0.2
	and eir.high_test_statistic > 0.05
),

news_halts as (
	select nh.halt_date, nh.symbol, 'News' as halt_reason
	from nsdq_halts nh 
	where halt_code = 'T3'
)

select co.*, 
       int.high_extreme as intra_high_extreme,
       int.high_score as intra_high_score,
       int.high_pvalue as intra_high_pvalue,
       int.high_test_statistic as intra_high_test_statistic,
       int.high_critical_value as intra_high_critical_value,
       COALESCE(nh.halt_reason, 'Unknown') as halt_reason
from interday_w_closeopen co
join intraday_extremes int
on (co.ticker, co.current_date) = (int.ticker, int.current_date)
left join news_halts nh
on (nh.halt_date, nh.symbol) = (co.current_date, co.ticker)
order by co.highlow_score asc, co.avg_dlynumtrd desc, int.high_score asc

