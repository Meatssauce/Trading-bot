from collections import defaultdict

import numpy as np
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
    info_data = []
    for symbol, ticker in tqdm(tickers.tickers.items()):
        info = ticker.info
        if info is None:
            info = {}
        info['ticker'] = symbol
        info_data.append(info)

    from joblib import dump
    with open('stockInfoTemp', 'wb') as f:
        dump(info_data, f)

    info_data_dict = defaultdict(list)
    max_info = max(info_data, key=lambda x: len(x))
    for info in info_data:
        for col in max_info:
            if col in info:
                info_data_dict[col].append(info[col])
            else:
                info_data.append(np.nan)
    df_info = pd.DataFrame.from_dict(info_data_dict)
    df_info.to_csv('stock_info.csv', index=False)
