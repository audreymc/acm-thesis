import datetime as dt
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
import numpy as np
from pyextremes import get_extremes
from pyextremes.plotting import plot_extremes
import wrds
from pyextremes import EVA
from pyextremes.tests import KolmogorovSmirnov
import sqlite3
from scipy import stats
from market_utils import MarketUtilities

class ExtremeValueScoring:
    def __init__(self, wrds_username, path_prefix="/Users/audreymcmillion/Documents/acm-thesis"):
        self.wrds_db = wrds.Connection(wrds_username=wrds_username)
        self.sqlite_conn = sqlite3.connect(f'{path_prefix}/databases/halt_data.db')
        self.mkt_utils = MarketUtilities(wrds_username=wrds_username, wrds_db=self.wrds_db, sqlite_conn=self.sqlite_conn)

        with open(f"{path_prefix}/sql_lib/daily_trades_wquotes.sql", "r") as file:
            self.intra_dly_trades_template = file.read()

        with open(f"{path_prefix}/sql_lib/sqlite_intraday_trades.sql", "r") as file:
            self.sqlite_intraday_tr_template = file.read()

    # function to get daily trades data from WRDS
    def get_daily_trades(self, before_dt, after_dt, symbol, use_sqlite=False, write_sqlite=False):
        dly_trades = self.mkt_utils.intraday_df_w_dates(symbol, 
                                                        before_dt=before_dt, 
                                                        after_dt=after_dt, 
                                                        use_sqlite=use_sqlite, 
                                                        write_sqlite=write_sqlite)
        
        if dly_trades.empty:
            return dly_trades
        
        dly_trades['datetime'] = pd.to_datetime(dly_trades['date'].astype(str) + ' ' + dly_trades['trunc_time'].astype(str))
        dly_trades = dly_trades.set_index('datetime')

        # return daily trades dataframe
        return dly_trades
    
    # function to test the fit of an extreme value distribution to the data
    def test_fit(self, model, significance_level = 0.05):
        if model is None:
            return {'model': None, 'p-value': np.nan, 'test_statistic': np.nan, 'critical_value': np.nan}
        
        kstest = KolmogorovSmirnov(
            extremes=model.extremes,
            distribution=model.distribution.name,
            fit_parameters=model.distribution.mle_parameters,
            significance_level=significance_level,
        )
        
        return {'model': model.distribution.name,
                'p-value': kstest.pvalue, 
                'test_statistic': kstest.test_statistic, 
                'critical_value': stats.kstwo.ppf(1 - 0.05 , len(model.extremes))}

    # function to get extreme value scores for a given dataframe row
    def get_intraday_ev_score(self, row):
        # get date and time values
        current_dt = row["current_date"]
        before_dt = row["before_date"]
        after_dt = row["after_date"]

        # get symbol
        symbol = row["ticker"]

        # print(symbol, current_dt, before_dt, after_dt)

        # get daily trades dataframe
        daily_trades = self.get_daily_trades(before_dt, after_dt, symbol)

        # check if empty
        if daily_trades.empty:
            return pd.Series({
                    "high_extreme": np.nan,
                    "high_score": np.nan,
                    "high_pvalue": np.nan,
                    "high_test_statistic": np.nan,
                    "high_critical_value": np.nan,
                    "high_model": None,
                    "low_extreme": np.nan,
                    "low_score": np.nan,
                    "low_pvalue": np.nan,
                    "low_test_statistic": np.nan,
                    "low_critical_value": np.nan,
                    "low_model": None,
                    "block_size": None
                })

        # get high/low extreme values on dates
        current_dt_dt = pd.to_datetime(current_dt).date()
        high_extreme = daily_trades[daily_trades.date == current_dt_dt].avg_price_diff.max()
        low_extreme = daily_trades[daily_trades.date == current_dt_dt].avg_price_diff.min()

        # choose a block size based on the number of trades in a given day
        # 390 = # of minutes in 6.5 hours (a trading day) -> 1 trade per minute on average
        if row["avg_dlynumtrd"] < 390:
            block_size = "1D"
        else:
            block_size = "1H"

        # fit high extreme value model
        high_model = EVA(daily_trades.dropna().avg_price_diff)
        try:
            high_model.get_extremes(method="BM", block_size=block_size, errors="ignore", extremes_type="high")
            high_model.fit_model()
            high_score = 1 - high_model.model.cdf(high_extreme)
            high_fit = self.test_fit(high_model)
        except:
            high_score = np.nan
            high_fit = self.test_fit(None)

        # fit low extreme value model -> we fit to the negative of the data to get the low extreme since 
        # the low extreme implementation in pyextremes is finnicky
        low_model = EVA((-1)*daily_trades.dropna().avg_price_diff)
        try:
            low_model.get_extremes(method="BM", block_size=block_size, errors="ignore", extremes_type="high")
            low_model.fit_model()
            low_score = 1 - low_model.model.cdf(np.abs(low_extreme))
            low_fit = self.test_fit(low_model)
        except:
            low_score = np.nan
            low_fit = self.test_fit(None)
       
        # return high/low scores
        return pd.Series({
                "high_extreme": high_extreme,
                "high_score": high_score,
                "high_pvalue": high_fit['p-value'],
                "high_test_statistic": high_fit['test_statistic'],
                "high_critical_value": high_fit['critical_value'],
                "high_model": high_fit['model'],
                "low_extreme": low_extreme,
                "low_score": low_score,
                "low_pvalue": low_fit['p-value'],
                "low_test_statistic": low_fit['test_statistic'],
                "low_critical_value": low_fit['critical_value'],
                "low_model": low_fit['model'],
                "block_size": block_size
            })
    
    # function to get interday extreme value scores for a given dataframe row
    # TODO: RERUN WITH 5D BLOCK SIZE
    def get_interday_ev_score(self, row, *, block_size="5D"):
        # get date and time values
        current_dt = row["current_date"]
        before_dt = row["before_date"]
        after_dt = row["after_date"]

        # get symbol
        symbol = row["ticker"]

        # get daily trades dataframe
        interday_df = self.mkt_utils.interday_df_w_dates(symbol, 
                                                         before_dt=before_dt, 
                                                         after_dt=after_dt)
        
        # set index to datetime
        interday_df['dlytime'] = '00:00:00'
        interday_df['datetime'] = pd.to_datetime(interday_df['dlycaldt'].astype(str) + ' ' + interday_df['dlytime'].astype(str))
        interday_df = interday_df.set_index('datetime')

        # get high low difference
        interday_df['dlyhighlowdiff'] = interday_df['dlyhigh'] - interday_df['dlylow']

        # get open close difference
        interday_df['dlyopenclosediff'] = np.abs(interday_df['dlyclose'] - interday_df['dlyopen'])

        # get high/low extreme values on dates
        current_dt_dt = pd.to_datetime(current_dt).date()

        # check if empty
        if interday_df.empty or (current_dt_dt not in interday_df.dlycaldt.values):
            return pd.Series({
                    "high_extreme": np.nan,
                    "high_score": np.nan,
                    "high_pvalue": np.nan,
                    "high_test_statistic": np.nan,
                    "high_critical_value": np.nan,
                    "high_model": None,
                    "low_extreme": np.nan,
                    "low_score": np.nan,
                    "low_pvalue": np.nan,
                    "low_test_statistic": np.nan,
                    "low_critical_value": np.nan,
                    "low_model": None,
                    "highlow_extreme": np.nan,
                    "highlow_score": np.nan,
                    "highlow_pvalue": np.nan,
                    "highlow_test_statistic": np.nan,
                    "highlow_critical_value": np.nan,
                    "highlow_model": None,
                    "openclose_extreme": np.nan,
                    "openclose_score": np.nan,
                    "openclose_pvalue": np.nan,
                    "openclose_test_statistic": np.nan,
                    "openclose_critical_value": np.nan,
                    "openclose_model": None,
                    "vol_extreme": np.nan,
                    "vol_score": np.nan,
                    "vol_pvalue": np.nan,
                    "vol_test_statistic": np.nan,
                    "vol_critical_value": np.nan,
                    "vol_model": None,
                })

        # get relevant values on dates
        current_dt_dt = pd.to_datetime(current_dt).date()
        high_extreme = interday_df[interday_df.dlycaldt == current_dt_dt]["dlyhigh"].iloc[0]
        low_extreme = interday_df[interday_df.dlycaldt == current_dt_dt]["dlylow"].iloc[0]
        vol_extreme = interday_df[interday_df.dlycaldt == current_dt_dt]["dlyvol"].iloc[0]
        open_extreme = interday_df[interday_df.dlycaldt == current_dt_dt]["dlyopen"].iloc[0]
        close_extreme = interday_df[interday_df.dlycaldt == current_dt_dt]["dlyclose"].iloc[0]

        # USING BLOCK SIZE OF 5 DAYS
        # NOTE: previously used 1D but that was too small for interday data

        # fit high extreme value model
        high_model = EVA(interday_df.dropna().dlyhigh)
        try:
            high_model.get_extremes(method="BM", block_size=block_size, errors="ignore", extremes_type="high")
            high_model.fit_model()
            high_score = 1 - high_model.model.cdf(high_extreme)
            high_fit = self.test_fit(high_model)
        except:
            high_score = np.nan
            high_fit = self.test_fit(None)

        # fit low extreme value model
        low_model = EVA(interday_df.dropna().dlylow)
        try:
            low_model.get_extremes(method="BM", block_size=block_size, errors="ignore", extremes_type="low")
            low_model.fit_model()
            low_score = 1 - low_model.model.cdf(np.abs(low_extreme))
            low_fit = self.test_fit(low_model)
        except:
            low_score = np.nan
            low_fit = self.test_fit(None)

        # fit high/low extreme value model
        highlow_model = EVA(interday_df.dropna().dlyhighlowdiff)
        try:
            highlow_model.get_extremes(method="BM", block_size=block_size, errors="ignore", extremes_type="high")
            highlow_model.fit_model()
            highlow_score = 1 - highlow_model.model.cdf(high_extreme - low_extreme)
            highlow_fit = self.test_fit(highlow_model)
        except:
            highlow_score = np.nan
            highlow_fit = self.test_fit(None)

        # fit open/close extreme value model
        openclose_model = EVA(interday_df.dropna().dlyopenclosediff)
        try:
            openclose_model.get_extremes(method="BM", block_size=block_size, errors="ignore", extremes_type="high")
            openclose_model.fit_model()
            openclose_score = 1 - openclose_model.model.cdf(np.abs(close_extreme - open_extreme))
            openclose_fit = self.test_fit(openclose_model)
        except:
            openclose_score = np.nan
            openclose_fit = self.test_fit(None)

        # fit volume extreme value model
        vol_model = EVA(interday_df.dropna().dlyvol)
        try:
            vol_model.get_extremes(method="BM", block_size=block_size, errors="ignore", extremes_type="high")
            vol_model.fit_model()
            vol_score = 1 - vol_model.model.cdf(vol_extreme)
            vol_fit = self.test_fit(vol_model)
        except:  
            vol_score = np.nan
            vol_fit = self.test_fit(None)

        # return high/low scores
        return pd.Series({
                "high_extreme": high_extreme,
                "high_score": high_score,
                "high_pvalue": high_fit['p-value'],
                "high_test_statistic": high_fit['test_statistic'],
                "high_critical_value": high_fit['critical_value'],
                "high_model": high_fit['model'],
                "low_extreme": low_extreme,
                "low_score": low_score,
                "low_pvalue": low_fit['p-value'],
                "low_test_statistic": low_fit['test_statistic'],
                "low_critical_value": low_fit['critical_value'],
                "low_model": low_fit['model'],
                "highlow_extreme": high_extreme - low_extreme,
                "highlow_score": highlow_score,
                "highlow_pvalue": highlow_fit['p-value'],
                "highlow_test_statistic": highlow_fit['test_statistic'],
                "highlow_critical_value": highlow_fit['critical_value'],
                "highlow_model": highlow_fit['model'],
                "openclose_extreme": np.abs(close_extreme - open_extreme),
                "openclose_score": openclose_score,
                "openclose_pvalue": openclose_fit['p-value'],
                "openclose_test_statistic": openclose_fit['test_statistic'],
                "openclose_critical_value": openclose_fit['critical_value'],
                "openclose_model": openclose_fit['model'],
                "vol_extreme": vol_extreme,
                "vol_score": vol_score,
                "vol_pvalue": vol_fit['p-value'],
                "vol_test_statistic": vol_fit['test_statistic'],
                "vol_critical_value": vol_fit['critical_value'],
                "vol_model": vol_fit['model']
                })


    # function to process INTRADAY halt data for extreme value scoring
    def process_intraday_data(self, before_after_df: pd.DataFrame) -> pd.DataFrame:
        results = before_after_df.apply(self.get_intraday_ev_score, axis=1)
        return before_after_df.join(results)
    
    # function to process the INTERDAY halt data for exteme value scoring
    def process_interday_data(self, before_after_df: pd.DataFrame) -> pd.DataFrame:
        results = before_after_df.apply(self.get_interday_ev_score, axis=1)
        return before_after_df.join(results)

