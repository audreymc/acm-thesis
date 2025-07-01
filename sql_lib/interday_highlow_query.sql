SELECT *
    FROM (
      SELECT b.ticker,
              a.permno,
              a.dlycaldt,
              a.dlyopen,
              coalesce(a.dlyclose, a.dlyprc) AS dlyclose,
              a.dlyhigh,
              a.dlylow,
              a.dlynumtrd,
              a.dlyvol,
              a.dlyhigh - a.dlylow AS highlow,
              LAG(a.dlyopen) OVER (ORDER BY a.dlycaldt) AS prevopen,
              LAG(a.dlyclose) OVER (ORDER BY a.dlycaldt) AS prevclose,
              LAG(a.dlyhigh) OVER (ORDER BY a.dlycaldt) AS prevhigh,
              a.dlyclose - LAG(a.dlyclose) OVER (ORDER BY a.dlycaldt) AS dlyreturn,
              ln(a.dlyhigh) - LAG(ln(a.dlyhigh)) OVER (ORDER BY a.dlycaldt) AS dlyhighreturn,
              (a.dlyhigh  - a.dlylow) AS highlow,
              (a.dlyhigh  - a.dlylow) - LAG( (a.dlyhigh  - a.dlylow)) OVER (ORDER BY a.dlycaldt) AS highlow_diff,
              ln(a.dlyhigh / a.dlylow) AS log_highlow,
              (ln(a.dlyhigh / a.dlylow) - LAG(ln(a.dlyhigh / a.dlylow)) OVER (ORDER BY a.dlycaldt))*100 AS log_highlow_diff
      FROM crsp.dsf_v2 a
      JOIN crsp.dsenames AS b
      ON a.permno = b.permno
      AND b.ticker = '{symbol}'
      AND a.dlycaldt BETWEEN date('{start_dt}') AND date('{end_dt}')
      AND date('{end_dt}') BETWEEN b.namedt AND b.nameendt
    )
WHERE dlyreturn IS NOT NULL
ORDER BY dlycaldt