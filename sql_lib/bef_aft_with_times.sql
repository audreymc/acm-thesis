with merged as (
    select bar.*, coalesce(nh.halt_time, ny.halt_time) as halt_time
    from before_after_results bar
    left join nsdq_halts nh
    on (bar.ticker, bar.current_date) = (nh.symbol, nh.halt_date) and nh.halt_code = 'LUDP'
    left join nyse_halts ny
    on (bar.ticker, bar.current_date) = (ny.symbol, ny.halt_date) and ny.halt_code = 'LULD Pause'
    where bar.open_pr is not null
    and bar.close_pr is not null
    and bar.dlyhigh is not null
    and bar.dlylow is not null
    and bar.dlynumtrd is not null
) 
select ticker,
       m.current_date,
       before_date,
       after_date,
       open_pr,
       close_pr,
       dlyhigh,
       dlylow,
       dlynumtrd,
       dlyvol,
       before_avg_open,
       before_avg_close,
       before_avg_dlyhigh,
       before_avg_dlylow,
       before_avg_dlynumtrd,
       before_avg_dlyvol,
       after_avg_open,
       after_avg_close,
       after_avg_dlyhigh,
       after_avg_dlylow,
       after_avg_dlynumtrd,
       after_avg_dlyvol,
       vlty_estimate,
       vix_close,
       halt_time
from merged m
where halt_time is not null
order by m.current_date
