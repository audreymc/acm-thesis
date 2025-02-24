select isd.date,
       isd.sym_root,
       time(isd.trunc_time / 1e9, 'unixepoch') AS trunc_time,
       isd.quote_count,
       isd.max_bid,
       isd.min_ask,
       isd.quote_spread,
       isd.avg_price,
       isd.max_price,
       isd.min_price,
       isd.volume,
       case 
        when isd.avg_intertrade_time = -9223372036854775808 then null
        else isd.avg_intertrade_time
       end as avg_intertrade_time,
       isd.num_trades,
       isd.qtr,
       isd.avg_price_diff,
       isd.num_trade_diff,
       isd.time_delta 
from intraday_symbol_details isd 
where isd.sym_root = '{symbol}'
and isd.date between date('{before_dt}') and date('{after_dt}')
order by isd.date, isd.trunc_time