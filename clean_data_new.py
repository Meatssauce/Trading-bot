import os
import random
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pickle import dump, load
from pandas.tseries.offsets import MonthEnd
import yfinance as yf
import matplotlib.pyplot as plt

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


def save_as_x_y(data, X_path, y_path):
    y = {k: df.pop('Decision') for k, df in data.items()}
    X = data

    os.makedirs(os.path.dirname(X_path), exist_ok=True)
    os.makedirs(os.path.dirname(y_path), exist_ok=True)
    with open(X_path, 'wb') as X_file, open(y_path, 'wb') as y_file:
        dump(X, X_file)
        dump(y, y_file)


def clean(df):
    # Convert dates to datetime
    df['Quarter end'] = pd.to_datetime(df['Quarter end'], errors='coerce')

    # Convert numeric data to numeric data
    numeric_columns = df.drop(columns=['Stock', 'Quarter end']).columns
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    for stock in tqdm(df['Stock'].unique()):
        subset = df['Stock'] == stock

        # todo: Remove duplicates
        # do something...

        # todo: Insert missing timestamps
        # do something...

    df = df.sort_values(by=['Quarter end', 'Stock'], axis=0, kind='mergesort')

    return df


class StockFundamentalDataImputer(TransformerMixin):
    def __init__(self, columns=None, method='linear', limit_direction='both'):
        self.columns = columns
        self.limit_direction = limit_direction
        self.method = method
        self.imputer = SimpleImputer()

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns
        self.imputer = self.imputer.fit(X[self.columns])
        return self

    def transform(self, X):
        for stock in tqdm(X['Stock'].unique()):
            subset = X['Stock'] == stock
            numeric_columns = X.drop(columns=['Stock', 'Quarter end']).columns

            # Imputation via interpolation
            X.loc[subset, numeric_columns] = X.loc[subset, numeric_columns].interpolate(
                method=self.method, limit_direction=self.limit_direction, axis=0)  # spline may be better?
        X[self.columns] = pd.DataFrame(self.imputer.transform(X[self.columns]), columns=self.columns, index=X.index)

        return X


def engineer_features(df, add_stock_info=False):
    for stock in tqdm(df['Stock'].unique()):
        subset = df['Stock'] == stock
        numeric_columns = df.drop(columns=['Stock', 'Quarter end']).columns

        # Replace values with percent difference or change
        df.loc[subset, numeric_columns] = df.loc[subset, numeric_columns].pct_change(periods=1)

        # Replace infinite values and nan with 0
        df.loc[subset, numeric_columns] = df.loc[subset, numeric_columns].replace([np.inf, -np.inf], 0)
        df.loc[subset, numeric_columns] = df.loc[subset, numeric_columns].fillna(0)

        # Create the class 'Decision' determining if a quarterly reports improvement is a buy, hold, or sell.
        # shifted by -1 to know if the prices will increase/decrease in the next quarter
        df.loc[subset, 'Decision'] = df[subset].apply(class_creation, axis=1).shift(-1)

        # Exclude the first and last rows (cannot label)
        df = df.drop(index=[df[subset].index[0], df[subset].index[-1]])

        # Add additional company info as features (very slow)
        if add_stock_info:
            info = yf.Ticker(stock).info
            company_info = ['industry', 'sector', 'country', 'market']
            for col in company_info:
                value = info.get(col)
                value = ['N/A'] if value is None else value
                df.loc[subset, col] = value

    # Drop the price related columns to prevent data leakage
    df = df.drop(['Price', 'Price high', 'Price low'], axis=1)

    return df


# Deviation Augmentation
def augment(df, sigma=0.05, size=20):
    scalars = np.random.normal(1, sigma, size)
    result = pd.DataFrame()
    for scalar in scalars:
        new_df = train_data.copy()
        new_df['Stock'] = new_df['Stock'] + str(scalar)
        numeric_columns = [col for col in df.columns if col not in ['Stock', 'Quarter end']]
        new_df[numeric_columns] = new_df[numeric_columns] * scalar
        result = pd.concat([result, new_df])

    return result


class LengthStandardizer(TransformerMixin):
    def __init__(self):
        self.length = None

    def fit(self, X, y=None, quantile=0.9):
        self.length = int(np.quantile(X['Stock'].value_counts(), quantile))
        return self

    def transform(self, X):
        return self._pad_data(X, length=self.length, padding='pre', truncating='pre')

    def _pad_data(self, df, length, padding, truncating):
        """
        Transforms all dataframes within the data to a fixed length via padding or truncating. If padding, pad with 0s and
        Hold for the label.

        :param df: a dictionary of dataframes indexed by date
        :param length: target row count for the dataframes
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

        assert isinstance(df, dict)
        assert padding in ['pre', 'post'] and truncating in ['pre', 'post']

        result = {}
        for stock in tqdm(df['Stock'].unique()):
            subset = df['Stock'] == stock
            padding_length = length - len(df.index)

            if padding_length > 0:
                # generate padding
                value_to_pad = pd.DataFrame({k: [0] for k in df.columns})
                value_to_pad['Decision'] = StockClass.HOLD.value

                df_to_pad = pd.concat([value_to_pad] * padding_length)
                if isinstance(df.index, pd.DatetimeIndex):
                    df_to_pad.index = _get_padding_dates(df.index[-1 if padding == 'post' else 0], padding_length)

                # add padding to df
                df = pd.concat([df, df_to_pad] if padding == 'post' else [df_to_pad, df])
                df = df.reset_index()
                # padding = np.repeat(one_seq[-1], padding_length).reshape(38, padding_length).transpose()
                # one_seq = np.concatenate([one_seq, padding])
            elif padding_length < 0:
                # truncate df
                df = df[:padding_length] if truncating == 'post' else df[-padding_length:]

            assert len(df.index) == length

            result[stock] = df
        return result

# desired_length = 101


# Parameters
use_augmentation = True  # Augment training data via sliding window, may cause data leak

# Load companies quarterly reports
df = pd.read_csv('historical_qrs.csv')

# Split training set, test set and validation set
train_stocks, test_stocks = train_test_split(df['Stock'].unique(), test_size=0.2, random_state=42)
train_data, test_data = df[df['Stock'].isin(train_stocks)], df[df['Stock'].isin(test_stocks)]
train_stocks, val_stocks = train_test_split(train_data['Stock'].unique(), test_size=0.25, random_state=42)
train_data, val_data = train_data[train_data['Stock'].isin(train_stocks)], \
                       train_data[train_data['Stock'].isin(val_stocks)]

# Preprocess data
train_data = clean(train_data)
test_data = clean(test_data)
val_data = clean(val_data)

# Imputation
imputer = StockFundamentalDataImputer(train_data.drop(columns=['Stock', 'Quarter end']).columns)
train_data = imputer.fit_transform(train_data)
test_data = imputer.transform(test_data)
val_data = imputer.transform(val_data)

# Feature engineering
train_data = engineer_features(train_data)
test_data = engineer_features(test_data)
val_data = engineer_features(val_data)

# train_data.to_csv('datasets/train_data.csv', index=False)
# train_data.plot(y=train_data.columns, kind='line')
# # train_data.plot(x=train_data.drop(columns=['Decision']).columns, y='Decision', kind='line')
# plt.show()

# Data augmentation
if use_augmentation:
    train_data = augment(train_data)

# Feature selection
selector = SelectKBest(f_classif, 30)
val_data = selector.fit_transform(val_data.drop(columns=['Decision', 'Stock', 'Quarter end']), val_data['Decision'])
train_data = selector.transform(train_data)
test_data = selector.transform(test_data)

# Save data
train_data.to_csv('datasets/train_data.csv', index=False)
test_data.to_csv('datasets/test_data.csv', index=False)
val_data.to_csv('datasets/val_data.csv', index=False)

# Augment training data via sliding window - may cause data leak
# todo: Use stretching, shifting and dynamic range compression (?) instead - see data augmentation for signals
# if augment_data:
#     tickers = list(train_data.keys())
#     for stock in tqdm(tickers):
#         ticker_data_length = len(train_data[stock].index)
#         for offset in range(1, ticker_data_length):
#             train_data[stock + f'_{offset}'] = train_data[stock][:-offset]

# Pad or truncate each df in data to desired length  # 101?
# use only line below to return as ndarray (faster but no hard to change later on)
# train_data = sequence.pad_sequences(train_data.values(), maxlen=desired_length, padding='pre', dtype='float',
# truncating='pre')
# todo: handle preprocessing within model?

# save_as_x_y(train_data, X_path='data/X_train.pkl', y_path='data/y_train.pkl')
# save_as_x_y(test_data, X_path='data/X_test.pkl', y_path='data/y_test.pkl')
# save_as_x_y(val_data, X_path='data/X_val.pkl', y_path='data/y_val.pkl')
# 'P/B ratio', 'P/E ratio', 'Cumulative dividends per share', 'Dividend payout ratio', 'Long-term debt to equity ratio', 'Equity to assets ratio',
#        'Net margin',
