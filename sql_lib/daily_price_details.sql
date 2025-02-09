SELECT  b.ticker,
        a.permno,
        a.dlycaldt,
        a.dlyopen,
        coalesce(a.dlyclose, a.dlyprc) AS dlyclose,
        a.dlyhigh,
        a.dlylow,
        a.dlynumtrd,
        a.dlyvol
FROM crsp.dsf_v2 a
JOIN crsp.dsenames AS b
ON a.permno = b.permno
AND b.ticker IN ({symbol_lst})
AND a.dlycaldt BETWEEN date('{start_dt}') AND date('{end_dt}')
AND date('{current_dt}') BETWEEN b.namedt AND b.nameendt