select *,
       date_trunc('second', time_m) as trunc_time
from taqm_{yr}.ctm_{yr} c 
where c.date = date('{current_dt}')
and c.sym_root = '{symbol}'
and time_m >= '09:30:00'
and time_m <= '16:00:00'
and tr_scond !~ '[OPQ654]'
and tr_corr = '00'
and sym_suffix is null -- exclude warrants, rights, units, etc
order by date, time_m