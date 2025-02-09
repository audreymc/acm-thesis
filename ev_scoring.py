import datetime as dt
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
import numpy as np
from pyextremes import get_extremes
from pyextremes.plotting import plot_extremes
import wrds
from pyextremes import EVA
import sqlite3

class ExtremeValueScoring:
    def __init__(self, wrds_username):
        self.wrds_db = wrds.Connection(wrds_username='audreymcmillion')
        self.sqlite_conn = sqlite3.connect('databases/halt_data.db')

        with open("sql_lib/daily_trades.sql", "r") as file:
            self.dly_trades_template = file.read()

    # function to get daily trades data from WRDS
    def get_daily_trades(self, current_dt, before_dt, after_dt, symbol):
        # get daily trades
        if int(before_dt[:4]) == int(after_dt[:4]):
            dly_trades = self.wrds_db.raw_sql(self.dly_trades_template.format(symbol=symbol, yr=current_dt[:4], start_dt=before_dt, end_dt=after_dt))
        else: 
            dly_trades_before = self.wrds_db.raw_sql(self.dly_trades_template.format(symbol=symbol, yr=before_dt[:4], start_dt=before_dt, end_dt=str(int(before_dt[:4])) + "-12-31"))
            dly_trades_after = self.wrds_db.raw_sql(self.dly_trades_template.format(symbol=symbol, yr=after_dt[:4], start_dt=str(int(after_dt[:4])) + "-01-01", end_dt=after_dt))
            dly_trades = pd.concat([dly_trades_before, dly_trades_after])

        if dly_trades.empty:
            return pd.DataFrame()

        # process price extremes
        dly_trades['trunc_time'] = (pd.to_datetime('00:00:00') + dly_trades['trunc_time']).dt.time
        dly_trades['datetime'] = pd.to_datetime(dly_trades['date'].astype(str) + ' ' + dly_trades['trunc_time'].astype(str))
        dly_trades = dly_trades.set_index('datetime')

        # return daily trades dataframe
        return dly_trades

    # function to get extreme value scores for a given dataframe row
    def get_ev_score(self, row):
        # get date and time values
        current_dt = row["current_date"]
        before_dt = row["before_date"]
        after_dt = row["after_date"]

        # get symbol
        symbol = row["ticker"]

        # get daily trades dataframe
        daily_trades = self.get_daily_trades(current_dt, before_dt, after_dt, symbol)

        # check if empty
        if daily_trades.empty:
            return pd.Series({"high_extreme": np.nan, "low_extreme": np.nan, "high_score": np.nan, "low_score": np.nan})

        # get high/low extreme values on dates
        current_dt_dt = pd.to_datetime(current_dt).date()
        high_extreme = daily_trades[daily_trades.date == current_dt_dt].avg_price_diff.max()
        low_extreme = daily_trades[daily_trades.date == current_dt_dt].avg_price_diff.min()

        # fit high extreme value model
        high_model = EVA(daily_trades.dropna().avg_price_diff)
        try:
            high_model.get_extremes(method="BM", block_size="1H", errors="ignore", extremes_type="high")
        except:
            return pd.Series({"high_extreme": np.nan, "low_extreme": np.nan, "high_score": np.nan, "low_score": np.nan})
        try:
            high_model.fit_model()
            high_score = 1 - high_model.model.cdf(high_extreme)
        except:
            high_score = np.nan

        # fit low extreme value model
        low_model = EVA((-1)*daily_trades.dropna().avg_price_diff)
        try:
            low_model.get_extremes(method="BM", block_size="1H", errors="ignore", extremes_type="high")
        except:
            return pd.Series({"high_extreme": np.nan, "low_extreme": np.nan, "high_score": np.nan, "low_score": np.nan})
        try:
            low_model.fit_model()
            low_score = 1 - low_model.model.cdf(np.abs(low_extreme))
        except:
            low_score = np.nan

        # return high/low scores
        return pd.Series({"high_extreme": high_extreme, "low_extreme": low_extreme, "high_score": high_score, "low_score": low_score})

    # function to process halt data for extreme value scoring
    def process_data(self, before_after_df: pd.DataFrame) -> pd.DataFrame:
        results = before_after_df.apply(self.get_ev_score, axis=1)
        return before_after_df.join(results)

