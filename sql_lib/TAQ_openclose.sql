WITH before_data AS (
    SELECT date,
           sym_root,
           min(case when ((tr_scond ~ 'O' or tr_scond ~ 'Q') and price != 0) then price else null end) as before_open_pr,
           min(case when ((tr_scond ~ '6' or tr_scond ~ 'M') and price != 0) then price else null end) as before_close_pr
    FROM taqm_{start_year}.ctm_{start_date}
    WHERE sym_root IN ({symbol_lst})
      AND (tr_scond ~ 'O' 
           OR tr_scond ~ '6'
           OR tr_scond ~ 'Q'
           OR tr_scond ~ 'M')
    GROUP BY date, sym_root
),

current_data AS (
    SELECT date,
           sym_root,
           min(case when ((tr_scond ~ 'O' or tr_scond ~ 'Q') and price != 0) then price else null end) as open_pr,
           min(case when ((tr_scond ~ '6' or tr_scond ~ 'M') and price != 0) then price else null end) as close_pr
    FROM taqm_{current_year}.ctm_{current_date}
    WHERE sym_root IN ({symbol_lst})
      AND (tr_scond ~ 'O' 
           OR tr_scond ~ '6'
           OR tr_scond ~ 'Q'
           OR tr_scond ~ 'M')
    GROUP BY date, sym_root
),

after_data AS (
    SELECT date,
           sym_root,
           min(case when ((tr_scond ~ 'O' or tr_scond ~ 'Q') and price != 0) then price else null end) AS after_open_pr,
           min(case when ((tr_scond ~ '6' or tr_scond ~ 'M') and price != 0) then price else null end) AS after_close_pr
    FROM taqm_{end_year}.ctm_{end_date}
    WHERE sym_root IN ({symbol_lst})
      AND (tr_scond ~ 'O' 
           OR tr_scond ~ '6'
           OR tr_scond ~ 'Q'
           OR tr_scond ~ 'M')
    GROUP BY date, sym_root
)

SELECT c.sym_root,
       COALESCE(c.date, TO_DATE('{current_date}', 'YYYYMMDD')) AS date,
       COALESCE(b.date, TO_DATE('{start_date}', 'YYYYMMDD')) AS before_date,
       COALESCE(a.date, TO_DATE('{end_date}', 'YYYYMMDD')) AS after_date,
       c.open_pr,
       c.close_pr,
       b.before_open_pr, 
       b.before_close_pr,  
       a.after_open_pr, 
       a.after_close_pr
FROM current_data c
LEFT JOIN before_data b
ON c.sym_root = b.sym_root
LEFT JOIN after_data a
ON c.sym_root = a.sym_root