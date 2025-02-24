import wrds
import sqlite3
import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import mplfinance as mpf
import time
from tqdm import tqdm


class MarketUtilities():
    def __init__(self, wrds_username, wrds_db = None, sqlite_conn = None):
        if wrds_db is not None:
            self.wrds_db = wrds_db
        else:
            self.wrds_db = wrds.Connection(wrds_username=wrds_username)
        
        if sqlite_conn is not None:
            self.sqlite_conn = sqlite_conn
        else:
            self.sqlite_conn = sqlite3.connect('databases/halt_data.db') 
            
        self.nyse = mcal.get_calendar("NYSE")

        with open("sql_lib/daily_price_details.sql", "r") as file:
            self.price_details_query = file.read()

        with open("sql_lib/intraday_data.sql", "r") as file:
            self.intraday_data_query = file.read()

    # function to get industry details from yahoo finance API
    def get_industry_data(self, symbols: list) -> pd.DataFrame:
        industries_dict = {
            "symbol": [],
            "industry": [],
            "sector": []
        }
        
        for symbol in tqdm(symbols):
            time.sleep(0.25)
            industries_dict["symbol"].append(symbol)
    
            try:
                ticker_info = yf.Ticker(symbol).info  # Fetch data once
                industries_dict["industry"].append(ticker_info.get("industry"))
                industries_dict["sector"].append(ticker_info.get("sector"))
            except:
                industries_dict["industry"].append(None)
                industries_dict["sector"].append(None)

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
    
    # function to get the intraday trade details for a given symbol from WRDS
    def intraday_df_w_dates(self, symbol, current_dt, before_dt, after_dt):
        return self.wrds_db.raw_sql(self.price_details_query.format(start_dt = before_dt,
                            current_dt = current_dt,
                            end_dt = after_dt,
                            symbol_lst = ("'" + symbol + "'"))).sort_values("dlycaldt").reset_index(drop=True)
        
    # function to get the daily price details for a given symbol from WRDS
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

    # function to plot a multi-day candlestick chart for a given symbol
    def multiday_candlestick(self, symbol, current_dt, diff_num=15):
        pd_result = self.multiday_df(symbol, current_dt, diff_num)
        pd_result = pd_result.rename(columns={"dlycaldt": "Date", 
                                              "dlyopen": "Open", 
                                              "dlyhigh": "High", 
                                              "dlylow": "Low", 
                                              "dlyclose": "Close", 
                                              "dlyvol": "Volume"})
        pd_result['Date'] = pd.to_datetime(pd_result['Date'])

        # plot
        mpf.plot(pd_result.set_index("Date"), 
                 type='candle', 
                 style='yahoo', 
                 volume=True, 
                 ylabel='Price', 
                 ylabel_lower='Volume', 
                 title='Candlestick Chart', 
                 figratio=(10,5), 
                 figscale=1)
        
        plt.show()