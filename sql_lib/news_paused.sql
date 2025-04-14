create table news_paused as
with relevant_lulds as (
	select bar.ticker, bar.current_date, bar.dlynumtrd
	from before_after_results bar 
	where ((bar.before_avg_dlynumtrd + bar.after_avg_dlynumtrd) * 1.0)/2 > 100
),

news_halts as (
	select nh.halt_date, nh.symbol, 'News' as halt_reason
	from nsdq_halts nh 
	where halt_code = 'T3'
)

select rl.*
from relevant_lulds rl
join news_halts nh
on (rl.ticker, rl.current_date) = (nh.symbol, nh.halt_date)