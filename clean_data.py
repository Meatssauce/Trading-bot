import os
from enum import Enum

import pandas as pd
import numpy as np
from tqdm import tqdm
from pickle import dump, load

from pandas.tseries.offsets import MonthEnd
import yfinance as yf


class StockClass(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


def set_index_to_date(df):
    """
    Returns a sorted datetime index
    """
    df['Quarter end'] = pd.to_datetime(df['Quarter end'])
    df = df.set_index("Quarter end")
    return df.sort_index(ascending=True)


def class_creation(df, threshold=0.03):
    """
    Creates classes of:
    - hold(0)
    - buy(1)
    - sell(2)

    Threshold can be changed to fit whatever price percentage change is desired
    """
    if df['Price high'] >= threshold and df['Price low'] >= threshold:
        # Buys
        return StockClass.BUY.value

    elif df['Price high'] <= -threshold and df['Price low'] <= -threshold:
        # Sells
        return StockClass.SELL.value

    else:
        # Holds
        return StockClass.HOLD.value


def pad_df_set(df_set, desired_length, padding, truncating):
    def _get_padding_dates(date, length):
        dates = []
        for i in range(1, length + 1):
            dates.append(pd.to_datetime(date, format='%Y%m%d') - MonthEnd(3 * i))
        dates.reverse()
        return dates

    assert padding in ['pre', 'post'] and truncating in ['pre', 'post']

    result = {}
    for i in tqdm(df_set.keys()):
        df = df_set[i]
        padding_length = desired_length - len(df)

        if padding_length > 0:
            # generate padding
            value_to_pad = pd.DataFrame({k: [0] for k in df.columns})
            value_to_pad['Decision'] = StockClass.HOLD.value

            df_to_pad = pd.concat([value_to_pad] * padding_length)
            df_to_pad.index = _get_padding_dates(df.index[-1 if padding == 'post' else 0], padding_length)

            # add padding to df
            df = pd.concat([df, df_to_pad] if padding == 'post' else [df_to_pad, df])
            # padding = np.repeat(one_seq[-1], padding_length).reshape(38, padding_length).transpose()
            # one_seq = np.concatenate([one_seq, padding])

        elif padding_length < 0:
            # truncate df
            df = df[:padding_length] if truncating == 'post' else df[-padding_length:]

        result[i] = df
    return result


"""
Idea:
Takes N most recent QRs as input and outputs buy, sell or hold for next quarter
- if number of most recent QRs < N, then data is padded with last row
- if number of most recent QRs > N, then data is truncated from earliest to latest
- assumes future performance is not always independent from the past (rejects random walk hypothesis)
- this is a multivariate times series classification
# - takes relative change as input but also considers relative total assets (not the change thereof)
# - assumes relative size of a company may matter too 
- if there exists data for a company over a period of N quarters than N-2 different training data can be generated 
using a sliding window method (potential data leak?)
"""

# Load dictionary of QRs by tickers
with open('data/qr_by_tickers.pkl', 'rb') as file:
    df_set = load(file)

for i in tqdm(df_set.keys()):
    # Setting the index as the Date
    df_set[i] = set_index_to_date(df_set[i])

    # Converting all values to numeric values
    df_set[i] = df_set[i].apply(pd.to_numeric, errors='coerce')

    # Interpolate missing values
    df_set[i] = df_set[i].interpolate(method='linear', axis=0)  # use spline yields may better results?

    # forward fill and back fill missing values at the bottom and top respectively
    df_set[i] = df_set[i].fillna(method='ffill')
    df_set[i] = df_set[i].fillna(method='bfill')

    # Replacing values with percent difference or change
    df_set[i] = df_set[i].pct_change(periods=1)

    # Replacing infinite values with NaN
    df_set[i] = df_set[i].replace([np.inf, -np.inf], np.nan)

    # fill any remaining nan with 0
    df_set[i] = df_set[i].fillna(0)

    # Creating the class 'Decision' determining if a quarterly reports improvement is a buy, hold, or sell.
    # Creating the new column with the classes, shifted by -1 in order to know if the prices will increase/decrease in
    # the next quarter.
    df_set[i]['Decision'] = df_set[i].apply(class_creation, axis=1).shift(-1)

    # Excluding the first and last rows (cannot label)
    df_set[i] = df_set[i][1:-1]

    # Dropping the price related columns to prevent data leakage
    df_set[i] = df_set[i].drop(['Price', 'Price high', 'Price low'], axis=1)

    # Add industry, sector, country and market
    # info = yf.Ticker(i).info
    # df_set[i]['Industry'] = info['industry'] * len(df_set[i].index)
    # df_set[i]['Sector'] = info['sector'] * len(df_set[i].index)
    # df_set[i]['Country'] = info['country'] * len(df_set[i].index)
    # df_set[i]['Market'] = info['market'] * len(df_set[i].index)
    # todo: check code, cluster

# sliding window data augmentation
# not recommended?
keys = list(df_set.keys())
for i in tqdm(range(len(keys))):
    ticker = keys[i]
    count = 1
    for j in range(1, len(df_set[ticker].index)):
        df_set[ticker + f'_{count}'] = df_set[ticker][:-j]
        count += 1

# Show distribution of df row counts (company QR counts)
# Padding each df with 0 and hold or truncating to 90% quantile of QR counts
df_set_lengths = []
for df in df_set.values():
    df_set_lengths.append(len(df))
desired_length = int(np.quantile(df_set_lengths, 0.9))

df_set = pad_df_set(df_set, desired_length=desired_length, padding='pre', truncating='pre')
# Use only this line to return ndarray right away (faster but no hard to change later on)
# final_df_set = sequence.pad_sequences(df_set.values(), maxlen=desired_length, padding='pre', dtype='float',
#                                    truncating='pre')
# todo: decide whether to pad to maximum length or desired length
# todo: handle preprocessing within model

# ===============================================

# # Combining all stock DFs into one
# big_df = pd.DataFrame()
# for i in tqdm(sequences.keys()):
#     big_df = big_df.append(sequences[i], sort=False)
#
# # Filling the NaNs with 0
# big_df = big_df.fillna(0)
#
# # Resetting the index because we no longer need the dates
# big_df = big_df.reset_index(drop=True)
#
# # Dropping the price related columns to prevent data leakage
# big_df = big_df.drop(['Price', 'Price high', 'Price low'], axis=1)

y = {k: df.pop('Decision') for k, df in df_set.items()}
X, y = np.stack(list(df_set.values())), np.stack(list(y.values()))

# Exporting the final DataFrames
X_filename = "data/X.pkl"
y_filename = 'data/y.pkl'
os.makedirs(os.path.dirname(X_filename), exist_ok=True)
os.makedirs(os.path.dirname(y_filename), exist_ok=True)
with open(X_filename, 'wb') as X_file, open(y_filename, 'wb') as y_file:
    dump(X, X_file)
    dump(y, y_file)
