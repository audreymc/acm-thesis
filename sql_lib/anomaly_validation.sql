select eir.ticker, eir.current_date, eir.highlow_score, eir.highlow_pvalue, eir.openclose_score, eir.openclose_pvalue 
from ev_interday_results_2 eir 
join manual_anomalies ma
on eir.ticker = ma.symbol
and eir.current_date = ma.date 
and ma.classification = 'Anomaly'
where eir.highlow_score is not null
and eir.highlow_score < 0.1
and eir.openclose_score > 0.4