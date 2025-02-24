with fitted_evs as (
	select *
	from ev_table2 et
	where et.high_extreme is not null
	and et.low_extreme is not null 
	and et.high_score is not not null 
	and et.low_score is not null
),

unfitted as (
	select *
	from fitted_evs
	where high_pvalue < 0.05
	or low_pvalue < 0.05
)

select u.*,
       bar.before_avg_dlynumtrd,
       bar.after_avg_dlynumtrd,
       bar.dlynumtrd
from unfitted u
left join before_after_results bar
on (u.ticker, u.current_date) = (bar.ticker, bar.current_date)
order by u.high_score desc, bar.before_avg_dlynumtrd desc