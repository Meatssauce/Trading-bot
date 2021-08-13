import os
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pickle import dump, load
from pandas.tseries.offsets import MonthEnd
import yfinance as yf


"""
Idea:
Takes N most recent QRs as input and outputs buy, sell or hold for next quarter
- if number of most recent QRs < N, then data is padded with zeros from the start
- if number of most recent QRs > N, then data is truncated from the start (oldest)
- assumes future performance is not always independent from the past (rejects random walk hypothesis?)
- this is a multivariate times series classification
# - takes relative change as input but also considers relative total assets (not the change thereof)
# - assumes relative size of a company may matter too 
- if there exists data for a company over a period of N quarters than N-2 different training data can be generated 
using a sliding window method (potential data leak?)
"""


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


def pad_data(data, to_length, padding, truncating):
    """
    Transforms all dataframes within the data to a fixed length via padding or truncating. If padding, pad with 0s and
    Hold for the label.

    :param data: a dictionary of dataframes indexed by date
    :param to_length: target row count for the dataframes
    :param padding: {'pre', 'post'} if 'pre' then pad from the start; if 'post' pad from the end
    :param truncating: {'pre', 'post'} if 'pre' then truncate from the start; if 'post' truncate from the end
    :return: a dictionary of dataframes indexed by date
    """

    def _get_padding_dates(date, length):
        dates = []
        for i in range(1, length + 1):
            dates.append(pd.to_datetime(date, format='%Y%m%d') - MonthEnd(3 * i))
        dates.reverse()
        return dates

    assert isinstance(data, dict)
    assert padding in ['pre', 'post'] and truncating in ['pre', 'post']

    result = {}
    for i in tqdm(data.keys()):
        df = data[i]
        padding_length = to_length - len(df.index)

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

        assert len(df.index) == to_length

        result[i] = df
    return result


def save_as_x_y(data, X_path, y_path):
    y = {k: df.pop('Decision') for k, df in data.items()}
    X = data

    os.makedirs(os.path.dirname(X_path), exist_ok=True)
    os.makedirs(os.path.dirname(y_path), exist_ok=True)
    with open(X_path, 'wb') as X_file, open(y_path, 'wb') as y_file:
        dump(X, X_file)
        dump(y, y_file)


# Parameters
add_ticker_info = False  # Fetch and add additional company info (very slow)
augment_data = True  # Augment training data via sliding window, may cause data leak

# Load dictionary of QRs by tickers
with open('data/qr_by_tickers.pkl', 'rb') as file:
    data = load(file)

# Preprocess data
for i in tqdm(data.keys()):
    # Set the index as the Date
    data[i] = set_index_to_date(data[i])

    # Convert all values to numeric
    data[i] = data[i].apply(pd.to_numeric, errors='coerce')

    # Imputation via interpolation
    data[i] = data[i].interpolate(method='linear', limit_direction='both', axis=0)  # spline may be better?

    # Replace values with percent difference or change
    data[i] = data[i].pct_change(periods=1)

    # Replace infinite values and nan with 0
    data[i] = data[i].replace([np.inf, -np.inf], 0)
    data[i] = data[i].fillna(0)

    # Create the class 'Decision' determining if a quarterly reports improvement is a buy, hold, or sell.
    # shifted by -1 to know if the prices will increase/decrease in the next quarter
    data[i]['Decision'] = data[i].apply(class_creation, axis=1).shift(-1)

    # Exclude the first and last rows (cannot label)
    data[i] = data[i][1:-1]

    # Drop the price related columns to prevent data leakage
    data[i] = data[i].drop(['Price', 'Price high', 'Price low'], axis=1)

    # Add additional company info as features
    if add_ticker_info:
        info = yf.Ticker(i).info
        company_info = ['industry', 'sector', 'country', 'market']
        for col in company_info:
            value = info.get(col)
            value = ['N/A'] if value is None else value
            data[i][col] = [value] * len(data[i].index)

# Make a more robust test test
train_data, test_data = [i.to_dict() for i in train_test_split(pd.Series(data), test_size=0.2, random_state=42)]

# Find 90% quantile of data lengths
train_data_lengths = [len(df) for df in train_data.values()]
desired_length = int(np.quantile(train_data_lengths, 0.9))

# Augment training data via sliding window - may cause data leak
if augment_data:
    tickers = list(train_data.keys())
    for ticker in tqdm(tickers):
        ticker_data_length = len(train_data[ticker].index)
        for offset in range(1, ticker_data_length):
            train_data[ticker + f'_{offset}'] = train_data[ticker][:-offset]

# Pad or truncate each df in data to desired length  # 101?
print(f'desired_length: {desired_length}')
train_data = pad_data(train_data, to_length=desired_length, padding='pre', truncating='pre')
test_data = pad_data(test_data, to_length=desired_length, padding='pre', truncating='pre')
# use only line below to return as ndarray (faster but no hard to change later on)
# train_data = sequence.pad_sequences(train_data.values(), maxlen=desired_length, padding='pre', dtype='float',
# truncating='pre')
# todo: handle preprocessing within model?

save_as_x_y(train_data, X_path='data/X_train.pkl', y_path='data/y_train.pkl')
save_as_x_y(test_data, X_path='data/X_test.pkl', y_path='data/y_test.pkl')
