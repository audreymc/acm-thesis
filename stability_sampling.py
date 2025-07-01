import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch
import numpy as np
from market_utils import MarketUtilities
import datetime

class StabilitySampling:
    def __init__(self, symbol: str, mkt_utils: MarketUtilities, ipo_dt: str, cutoff_dt: str):
        self.symbol = symbol
        self.mkt_utils = mkt_utils
        self.ipo_dt = ipo_dt
        self.cutoff_dt = cutoff_dt
        self.data = self.get_daily_data(self.ipo_dt, self.cutoff_dt) # get daily high/low data

    # get daily high/low data for the symbol
    def get_daily_data(self, start_dt: str, end_dt: str) -> pd.DataFrame:
        with open("sql_lib/interday_highlow_query.sql", "r") as file:
            interday_hl_template = file.read()

        extract_data = self.mkt_utils.wrds_db.raw_sql(interday_hl_template.format(symbol=self.symbol, \
                                                                   start_dt=start_dt, \
                                                                   end_dt=end_dt))
        
        return extract_data
    
    # get the high/low series for a given date interval
    def get_highlow_series(self, start_dt, end_dt, date_col = 'dlycaldt', hl_col = 'log_highlow_diff'):
        # convert to date if needed
        start_dt = self.mkt_utils.to_date(start_dt)
        end_dt = self.mkt_utils.to_date(end_dt)
        
        # get the series
        extract_srs = self.data[(self.data[date_col] >= start_dt) & 
                    (self.data[date_col] <= end_dt)]
        extract_srs = extract_srs.set_index(date_col)[hl_col]
        return extract_srs
    
    # function to reset the data for a new symbol or date range
    def reset_data(self, symbol: str, ipo_dt: str, cutoff_dt: str):
        self.symbol = symbol
        self.ipo_dt = ipo_dt
        self.cutoff_dt = cutoff_dt
        self.data = self.get_daily_data(self.ipo_dt, self.cutoff_dt)
   

    # get set of sample intervals
    def get_date_sample_intervals(self, n=5, sample_size=461, date_col='dlycaldt'):
        # get date range
        earliest_dt = self.mkt_utils.get_after_date(self.ipo_dt, days_after=1)
        latest_dt = self.mkt_utils.get_before_date(current_date=self.cutoff_dt, days_before=1)

        # get trading days
        trading_days = self.data[date_col].sort_values().unique()
        n_trading_days = len(trading_days)

        # ensure there are enough trading days for at least 2 non-overlapping samples
        if (sample_size * 2) >= n_trading_days:
            raise ValueError("Sample size is larger than the number of trading days available.")

        # Compute equally spaced starting indices (allowing overlap)
        if n == 1:
            start_indices = [0]
        else:
            max_start = n_trading_days - sample_size
            start_indices = np.linspace(0, max_start, n, dtype=int)

        sample_intervals = []
        for start_idx in start_indices:
            end_idx = start_idx + sample_size - 1
            start_date = trading_days[start_idx]
            end_date = trading_days[end_idx]
            sample_intervals.append((self.mkt_utils.to_date(start_date), self.mkt_utils.to_date(end_date)))

        return sample_intervals


    # get stability metrics for the series
    def get_stability_metrics(self, extract_srs):
        # metrics
        rolling_std_30d = extract_srs.rolling(window=30).std()
        rolling_mean_30d = extract_srs.rolling(window=30).mean()
        
        # get rolling summary statistics
        min_30std, max_30std, median_30std = rolling_std_30d.min(), rolling_std_30d.max(), rolling_std_30d.median()
        min_30mean, max_30mean, median_30mean = rolling_mean_30d.min(), rolling_mean_30d.max(), rolling_mean_30d.median()
        
        # get overall summary statistics
        overall_std = extract_srs.std()
        overall_mean = extract_srs.mean()
        
        # adfuller test - unit root test
        adf_pvalue = adfuller(extract_srs)[1]
        
        # test for ARCH effects
        arch_test_result = het_arch(extract_srs, nlags=12)
        arch_lmpval, arch_fval = arch_test_result[1], arch_test_result[3]

        return {
            'symbol': self.symbol,
            'min_date': extract_srs.index.min().strftime('%Y-%m-%d'),
            'max_date': extract_srs.index.max().strftime('%Y-%m-%d'),
            'n_days': len(extract_srs),
            'min_30day_std': min_30std,
            'max_30day_std': max_30std,
            'median_30day_std': median_30std,
            'min_30day_mean': min_30mean,
            'max_30day_mean': max_30mean,
            'median_30day_mean': median_30mean,
            'overall_std': overall_std,
            'overall_mean': overall_mean,
            'adf_pvalue': adf_pvalue,
            'arch_lm_pvalue': arch_lmpval,
            'arch_f_pvalue': arch_fval
        }


        
