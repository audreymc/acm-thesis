import wrds
import sqlite3
import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt

class MarketUtilities():
    def __init__(self, wrds_username):
        self.wrds_db = wrds.Connection(wrds_username='audreymcmillion')
        self.sqlite_conn = sqlite3.connect('databases/halt_data.db')
        self.nyse = mcal.get_calendar("NYSE")

        with open("sql_lib/daily_price_details.sql", "r") as file:
            self.price_details_query = file.read()

    # function to get industry details from yahoo finance API
    def get_industry_data(self, symbols: list) -> pd.DataFrame:
        industries_dict = {
            "symbol": [],
            "industry_key": [],
            "sector_key": []
        }
        
        for symbol in symbols:
            industries_dict["symbol"].append(symbol)
            ticker_info = yf.Ticker(symbol).info  # Fetch data once

            try:
                industries_dict["industry_key"].append(ticker_info.get("industryKey"))
                industries_dict["sector_key"].append(ticker_info.get("sectorKey"))
            except:
                industries_dict["industry_key"].append(None)
                industries_dict["sector_key"].append(None)

        return pd.DataFrame.from_dict(industries_dict)

    # function to get the trading date days_before the current date
    def get_before_date(self, current_date, days_before):
        # get set of trading days
        trading_days = self.nyse.valid_days(start_date="2000-01-01", end_date=current_date)

        # Find the trade date days_before days before today
        trade_date_days_ago = trading_days[-1 - days_before]

        return trade_date_days_ago.strftime('%Y-%m-%d')

    # function to get the trading date days_after the current date
    def get_after_date(self, current_date, days_after):
        today = pd.Timestamp.today().normalize()

        # get set of trading days
        trading_days = self.nyse.valid_days(start_date=current_date, end_date=today)

        # Find the trade date days_after days after today
        trade_date_days_after = trading_days[days_after]

        return trade_date_days_after.strftime('%Y-%m-%d')
    
    # funnction to get the daily price details for a given symbol from WRDS
    def multiday_df(self, symbol, current_dt, diff_num):
        return self.wrds_db.raw_sql(self.price_details_query.format(start_dt = self.get_before_date(current_dt, diff_num),
                            current_dt = current_dt,
                            end_dt = self.get_after_date(current_dt, diff_num),
                            symbol_lst = ("'" + symbol + "'"))).sort_values("dlycaldt").reset_index(drop=True)

    # function to plot a multi-day chart for a given symbol
    def multiday_chart(self, symbol, current_dt, high = True, diff_num = 15):
        pd_result = self.multiday_df(symbol, current_dt, diff_num)
        
        fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

        if high:
            column_name = 'dlyhigh'
            line_label = 'daily high'
            first_title = "Daily Highs"
        else:
            column_name = 'dlylow'
            line_label = 'daily low'
            first_title = "Daily Lows"
        
        # First plot
        ax[0].plot(pd_result.dlycaldt.to_list(), pd_result[column_name].to_list(), label=line_label, color='blue')
        ax[0].set_title(first_title)
        
        # Second plot
        ax[1].plot(pd_result.dlycaldt.to_list(), pd_result.dlyclose.to_list(), label='daily close', color='red')
        ax[1].set_title("Closing Prices")
        
        # Adjust layout and show
        ax[0].legend()
        ax[1].legend()
        plt.tight_layout()
        plt.show()