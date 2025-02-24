with rw_trades as (
    select *,
           date_trunc('second', time_m) as trunc_time
    from taqm_{yr}.ctm_{yr} c 
    where c.date between date('{start_dt}') and date('{end_dt}')
    and c.sym_root = '{symbol}'
    and time_m >= '09:30:00'
    and time_m <= '16:00:00'
    and tr_scond !~ '[OPQ654]'
    and tr_corr = '00' -- non-corrected/cancelled trades
    and sym_suffix is null -- exclude warrants, rights, units, etc
),

aggregated as (
    select r.date, 
           r.sym_root, 
           r.trunc_time, 
           avg(r.price) as avg_price,
           max(r.price) as max_price,
           min(r.price) as min_price,
           sum(r.size) as volume, 
           max(r.tr_seqnum) as tr_seqnum,
           count(*) as num_trades
    from rw_trades r 
    group by r.date, r.sym_root, r.trunc_time
)

select a.*,
       (a.avg_price - lag(a.avg_price) over (
                    partition by a.date 
                    order by a.date, a.trunc_time
                    )) as avg_price_diff,
        (a.num_trades - lag(a.num_trades) over (
                    partition by a.date 
                    order by a.date, a.trunc_time
                    )) as num_trade_diff,
        EXTRACT(EPOCH FROM (a.trunc_time - LAG(a.trunc_time) OVER (
                    PARTITION BY a.date 
                    ORDER BY a.date, a.trunc_time))) AS time_delta
from aggregated a
order by a.date, a.trunc_time