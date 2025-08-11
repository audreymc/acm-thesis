with quotes_daily as (
	select c.*,
	       date_trunc('second', c.time_m) as trunc_time
	from taqm_{yr}.cqm_{yr} c 
	where c.sym_root = '{symbol}'
	and c.date between date('{start_dt}') and date('{end_dt}')
	and time_m >= '09:30:00'
	and time_m <= '16:00:00'
	and sym_suffix is null
	and qu_cond in ('R', 'Y') -- regular quotes
),

quote_count as (
	select q.date, 
	       q.sym_root, 
	       q.trunc_time, 
	       count(*) as quote_count,
	       max(CASE WHEN q.bid > 0 THEN q.bid END) AS max_bid,
              min(CASE WHEN q.ask > 0 THEN q.ask END) AS min_ask,
              avg((CASE WHEN q.ask > 0 THEN q.ask END) - (CASE WHEN q.bid > 0 THEN q.bid END)) as quote_spread
	from quotes_daily q
	group by 1, 2, 3
	order by 1, 3
),

rw_trades as (
    select *,
           date_trunc('second', time_m) as trunc_time, -- truncating by second
           (c.time_m - (lag(c.time_m) over (partition by c.date, c.sym_root, date_trunc('second', c.time_m) ORDER BY c.time_m))) AS intertrade_time
    from taqm_{yr}.ctm_{yr} c 
    where c.date between date('{start_dt}') and date('{end_dt}')
    and c.sym_root = '{symbol}'
    and time_m >= '09:30:00'
    and time_m <= '16:00:00'
    and tr_scond !~ '[OPQ654ZU]'
    and tr_corr = '00' -- non-corrected/cancelled trades
    and sym_suffix is null -- exclude warrants, rights, units, etc
),

aggregated_trades as (
    select r.date, 
           r.sym_root, 
           r.trunc_time, 
           avg(r.price) as avg_price,
           max(r.price) as max_price,
           min(r.price) as min_price,
           sum(r.size) as volume, 
           max(r.tr_seqnum) as tr_seqnum,
           avg(r.intertrade_time) as avg_intertrade_time,
           count(distinct tr_seqnum) as num_trades
    from rw_trades r 
    group by r.date, r.sym_root, r.trunc_time
)

select a.date, 
       a.sym_root, 
       a.trunc_time,
       coalesce(q.quote_count, 0) as quote_count,
       coalesce(q.max_bid, 0) as max_bid,
       coalesce(q.min_ask, 0) as min_ask,
       q.quote_spread,
       coalesce(a.avg_price, 0) as avg_price,
       coalesce(a.max_price, 0) as max_price,
       coalesce(a.min_price, 0) as min_price,
       coalesce(a.volume, 0) as volume,
       EXTRACT(HOUR FROM a.avg_intertrade_time) * 3600000000000 + 
		  EXTRACT(MINUTE FROM a.avg_intertrade_time) * 60000000000 + 
		  EXTRACT(SECOND FROM a.avg_intertrade_time) * 1000000000 AS avg_intertrade_time,
       coalesce(a.num_trades, 0) as num_trades,
       case
       	when coalesce(a.num_trades, 0) = 0 then null
       	else cast(coalesce(q.quote_count, 0) as real)/cast(coalesce(a.num_trades, 0) as real)
       end as qtr,
       (a.avg_price - lag(a.avg_price) over (
                    partition by a.date 
                    order by a.date, a.trunc_time
                    )) as avg_price_diff,
       (a.num_trades - lag(a.num_trades) over (
                    partition by a.date 
                    order by a.date, a.trunc_time
                    )) as num_trade_diff,
        extract(epoch from (a.trunc_time - lag(a.trunc_time) over (
                    partition by a.date 
                    order by a.date, a.trunc_time))) as time_delta
from aggregated_trades a 
left join quote_count q 
on (q.date, q.sym_root, q.trunc_time) = (a.date, a.sym_root, a.trunc_time) 
order by q.trunc_time