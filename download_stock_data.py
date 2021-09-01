from pandas_datareader import data as pdr
import pandas as pd

import yfinance as yf
yf.pdr_override()


if __name__ == '__main__':
    df = pd.read_csv('historical_qrs.csv')

    quarter_ends = pd.to_datetime(df['Quarter end'], errors='coerce').dropna()
    start_date, end_date = min(quarter_ends).date(), max(quarter_ends).date()
    stocks = df['Stock'].unique().tolist()

    # Download all historical data of each stock over its period
    df = pdr.get_data_yahoo(' '.join(stocks), start=str(start_date), end=str(end_date), group_by='Ticker', threads=True)

    # turn multilevel columns grouped by ticker into flat dataframe with an additional column called ticker
    df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)

    df.index = pd.to_datetime(df.index)
    df.to_csv('historical_stock_data.csv')
