# Sourcing the data
# Load libraries
import pandas as pd
import bs4 as bs
import datetime as dt
import os
import pandas_datareader.data as web
import pickle
import requests
import csv
# Yahoo for dataReader
import yfinance as yf
yf.pdr_override()

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Load dataset


def get_data_from_yahoo():
    tickers = ['^DJI','^GSPC', '^VIX']
    start = dt.datetime(2007, 1, 1)
    end = dt.datetime.now()
    for ticker in tickers:
        ticker_dataset = yf.download(ticker, start=start, end=end)
        ticker_dataset.to_csv(str(ticker)+"_data.csv")

    return ticker_dataset


get_data_from_yahoo()
