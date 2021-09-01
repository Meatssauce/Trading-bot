from collections import defaultdict
from tqdm import tqdm

from pandas_datareader import data as pdr
import pandas as pd

import yfinance as yf
yf.pdr_override()


if __name__ == '__main__':
    df = pd.read_csv('historical_qrs.csv')

    quarter_ends = pd.to_datetime(df['Quarter end'], errors='coerce').dropna()
    start_date, end_date = min(quarter_ends).date(), max(quarter_ends).date()
    stocks = df['Stock'].unique().tolist()

    # Download info each stock over its period
    tickers = yf.Tickers(' '.join(stocks))
    info_data = defaultdict(list)
    for symbol, ticker in tqdm(tickers.tickers.items()):
        info_data['ticker'].append(symbol)
        for key, val in ticker.info.items():
            info_data[key].append(val)
    df_info = pd.DataFrame.from_dict(info_data)
    df_info.to_csv('stock_info.csv', index=False)
