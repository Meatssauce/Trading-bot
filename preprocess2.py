import os
import random
from enum import Enum
import pandas as pd
import numpy as np
from category_encoders import BinaryEncoder
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, MinMaxScaler
from sklearn.base import TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pickle import dump, load
from pandas.tseries.offsets import MonthEnd
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

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


class StocksImputer(TransformerMixin):
    def __init__(self, method: str = 'linear', limit_direction: str = 'both'):
        self.method = method
        self.limit_direction = limit_direction

    def fit(self, df):
        return self

    def transform(self, df):
        # Interpolate missing values in columns
        for stock in tqdm(df.index.get_level_values('Stock').unique()):
            df.loc[stock, :] = df.xs(stock).interpolate(method=self.method, limit_direction=self.limit_direction).values
            # spline or time may be better?
        return df


class OutlierNullifier(TransformerMixin):
    def __init__(self, **kwargs):
        """
        Create a transformer to remove outliers.

        Returns:
            object: to be used as a transformer method as part of Pipeline()
        """

        self.quantiles = {}

    def fit(self, X, y=None, **fit_params):
        if isinstance(X, np.ndarray):
            for i in range(X.shape[1]):
                self.quantiles[i] = np.quantile(X[i], [0.25, 0.75])
        else:
            for column in X.columns:
                self.quantiles[column] = X[column].quantile(0.25), X[column].quantile(0.75)

        return self

    def transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            for i in range(X.shape[1]):
                q1, q3 = self.quantiles[i]
                iqr = q3 - q1
                X[i] = np.where((X[i] < q1 - 1.5 * iqr) | (X[i] > q3 + 1.5 * iqr), np.nan, X[i])
        else:
            for column in X.columns:
                q1, q3 = self.quantiles[column]
                iqr = q3 - q1
                X[column] = np.where((X[column] < q1 - 1.5 * iqr) | (X[column] > q3 + 1.5 * iqr), np.nan, X[column])

        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X, y)


class OutlierMinMaxTransformer(TransformerMixin):
    def __init__(self):
        self.q1 = None
        self.q3 = None

    def fit(self, X, y=None):
        self.q1 = X.quantile(0.25).tolist()
        self.q3 = X.quantile(0.75).tolist()
        return self

    def transform(self, X):
        for col, q1, q3 in zip(X.columns, self.q1, self.q3):
            X[col] = np.where(X[col] < q1, q1, X[col])
            X[col] = np.where(X[col] > q3, q3, X[col])
        return X


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


def save_as(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        dump(obj, file)


def save_as_x_y(data, X_path, y_path):
    y = {k: df.pop('Label') for k, df in data.items()}
    X = data

    save_as(X, X_path)
    save_as(y, y_path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Convert dates to datetime
    df['Quarter end'] = pd.to_datetime(df['Quarter end'], errors='coerce').dt.date

    # Replace missing flags with np.nan
    df['Stock'] = np.where((df['Stock'] == 'None') | (df['Stock'] == ''), np.nan, df['Stock'])

    # Drop invalid rows
    df = df.dropna(subset=['Stock', 'Quarter end'], how='any')

    # Set and sort multi-index
    df = df.set_index(['Stock', 'Quarter end'])
    df = df.sort_index(level=df.index.names)

    # Remove duplicates
    duplicates = df.index.duplicated(keep='first')
    df = df[~duplicates]

    # Convert numeric data to numeric data
    df = df.apply(pd.to_numeric, errors='coerce')

    # # todo: Insert missing timestamps
    # for stock in tqdm(df.index.get_level_values('Stock').unique()):
    #     timestamps = df.xs(stock).index
    #     idx = pd.period_range(min(timestamps), max(timestamps))
    #     df.loc[stock, :] = df.loc[stock, :].reindex(idx, fill_value=0)

    return df


def engineer_features(df: pd.DataFrame, add_stock_info: bool = False):
    # Add volatility
    df['Volatility'] = (df['Price high'] - df['Price low']) / df['Price']

    # Drop the price related columns to prevent data leakage
    df = df.drop(columns=['Price high', 'Price low'])

    # Replace values with percent difference or change, then replace newly created inf and nans with 0
    df_pct_delta = df.pct_change(periods=1).replace([np.inf, -np.inf], 0).fillna(0)

    # Rename columns in new dataframe
    df_pct_delta = df_pct_delta.rename({col: 'Delta ' + col for col in df_pct_delta.columns if col != 'Price'}, axis=1)

    # Create label from stock price shifted 1 period into the past
    df_pct_delta = df_pct_delta.rename({'Price': 'Label'}, axis=1)
    df_pct_delta['Label'] = df_pct_delta['Label'].shift(-1)

    # Drop price related fields
    df = df.drop(columns=['Price'])

    # Append new dataframe to columns
    df = pd.concat([df, df_pct_delta], axis=1)

    idx_to_drop = []
    for stock in tqdm(df.index.get_level_values('Stock').unique()):
        # Drop the first and last row (cannot label)
        idx_to_drop += [(stock, df.xs(stock).index[0]), (stock, df.xs(stock).index[-1])]

        # # Add additional company info as features (very slow)
        # if add_stock_info:
        #     info = yf.Ticker(stock).info
        #     company_info = ['industry', 'sector', 'country', 'market']
        #     values = []
        #     for col in company_info:
        #         value = info.get(col)
        #         values.append(['N/A'] if value is None else value)
        #     df.loc[stock, company_info] = values
    df = df.drop(idx_to_drop)

    return df


# Deviation Augmentation
def augment(df: pd.DataFrame, sigma: float = 0.05, size: int = 20) -> pd.DataFrame:
    scalars = np.random.normal(1, sigma, size)
    result = pd.DataFrame()
    for scalar in scalars:
        new_df = train_data.copy()
        new_df['Stock'] = new_df['Stock'] + str(scalar)
        # numeric_columns = [col for col in df.columns if col not in ['Stock', 'Quarter end']]
        numeric_columns = [col for col in df.columns if col != 'Stock']
        new_df[numeric_columns] = new_df[numeric_columns] * scalar
        result = pd.concat([result, new_df])

    return result


class PaddingTransformer(TransformerMixin):
    def __init__(self, maxlen: int = None, padding: str = 'pre', truncating: str = 'pre', dtype: str = 'float'):
        self.length = maxlen  # 101
        self.padding = padding
        self.truncating = truncating
        self.dtype = dtype

    def fit(self, X, y=None, quantile=0.9):
        self.length = int(np.quantile(X.index.get_level_values('Stock').value_counts(), quantile))
        return self

    def transform(self, X):
        # X = pd.DataFrame(pad_sequences(X.values, padding=self.padding, truncating=self.truncating, dtype=self.dtype),
        # columns=X.columns)
        X = self._pad_dataframes(X, self.length, self.padding, self.truncating)
        return X

    def _pad_dataframes(self, df: pd.DataFrame, length: int, padding: str, truncating: str) -> pd.DataFrame:
        """
        Transforms all dataframes within the data to a fixed length via padding or truncating. If padding, pad with 0s.

        :param df: dataframe
        :param length: target row count for the dataframes
        :param padding: {'pre', 'post'} if 'pre' then pad from the start; if 'post' pad from the end
        :param truncating: {'pre', 'post'} if 'pre' then truncate from the start; if 'post' truncate from the end
        :return: a dictionary of dataframes indexed by date
        """

        # assert isinstance(df, dict)
        assert padding in ['pre', 'post'] and truncating in ['pre', 'post']

        segments = []
        for stock in tqdm(df.index.get_level_values('Stock').unique()):
            df_subset = df.loc[stock, :]
            padding_length = length - len(df_subset.index)

            if padding_length > 0:
                # pad
                df_to_pad = pd.DataFrame({col: [0.0] * padding_length for col in df_subset.columns})
                if padding == 'post':
                    segments.append(df_subset)
                    segments.append(df_to_pad)
                else:
                    segments.append(df_to_pad)
                    segments.append(df_subset)
            elif padding_length < 0:
                # truncate
                truncated = df_subset[:padding_length] if truncating == 'post' else df_subset[-padding_length:]
                segments.append(truncated)
            else:
                segments.append(df_subset)
        result = pd.concat(segments)

        return result


class Parser(TransformerMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.simple_imputer = None
        self.interpolation_imputer = None
        self.outlier_transformer = None
        self.scaler = None
        self.padder = None

    def fit_transform(self, data, **fit_params):
        # Fill missing values via interpolation
        print("Imputing...")
        self.interpolation_imputer = StocksImputer(method='linear', limit_direction='both')
        data = self.interpolation_imputer.fit_transform(data)

        # Fill remaining nan (nan columns after when grouped by stock)
        self.simple_imputer = SimpleImputer()
        data = pd.DataFrame(self.simple_imputer.fit_transform(data), columns=data.columns, index=data.index)

        # Remove outliers in absolute values?

        # Data augmentation
        # if use_augmentation:
        #     train_data = augment(train_data)

        # Feature engineering
        print("Engineering features...")
        data = engineer_features(data, add_stock_info=False)

        # Remove outliers
        self.outlier_transformer = OutlierMinMaxTransformer()
        data = self.outlier_transformer.fit_transform(data)

        # Scale
        # todo: switch to MinMaxScaler, try StandardScaler
        features = data.drop(columns=['Label']).columns
        self.scaler = MinMaxScaler()
        data[features] = self.scaler.fit_transform(data[features])

        # Feature selection
        # features = train_data.drop(columns=['Label', 'Stock', 'Quarter end']).columns
        # features = train_data.drop(columns=['Label', 'Stock']).columns
        # selector = SelectKBest(f_classif, 30)
        # val_data[features] = selector.fit_transform(val_data[features], val_data['Label'])
        # train_data[features] = selector.transform(train_data[features])
        # test_data[features] = selector.transform(test_data[features])

        # # Reset index and remove useless features
        # df = df.sort_index(level=df.index.names).reset_index()
        # del df['Quarter end']
        #
        # # Encode
        # # todo: scale (and normalise) categorical variables? Never!
        # encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        # train_data['Stock'] = encoder.fit_transform(train_data[['Stock']])

        # Pad data
        self.padder = PaddingTransformer(padding='pre', truncating='pre', dtype='float')
        data = self.padder.fit_transform(data)

        # Extract X, y
        y, X = data.pop('Label'), data

        # Reshape as ndarrays (n_stocks, n_timestamps, n_features)
        X = X.values.reshape(-1, self.padder.length, len(X.columns))
        y = y.values.reshape(-1, self.padder.length, 1)

        return X, y

    def transform(self, data):
        # Fill missing values via interpolation
        print("Imputing...")
        data = self.interpolation_imputer.transform(data)

        # Fill remaining nan (nan columns after when grouped by stock)
        data = pd.DataFrame(self.simple_imputer.transform(data), columns=data.columns, index=data.index)

        # Remove outliers in absolute values?

        # Data augmentation
        # if use_augmentation:
        #     train_data = augment(train_data)

        # Feature engineering
        print("Engineering features...")
        data = engineer_features(data, add_stock_info=False)

        # Remove outliers
        data = self.outlier_transformer.transform(data)

        # Scale
        # todo: switch to MinMaxScaler, try StandardScaler
        features = data.drop(columns=['Label']).columns
        data[features] = self.scaler.transform(data[features])

        # Feature selection
        # features = train_data.drop(columns=['Label', 'Stock', 'Quarter end']).columns
        # features = train_data.drop(columns=['Label', 'Stock']).columns
        # selector = SelectKBest(f_classif, 30)
        # val_data[features] = selector.fit_transform(val_data[features], val_data['Label'])
        # train_data[features] = selector.transform(train_data[features])
        # test_data[features] = selector.transform(test_data[features])

        # # Reset index and remove useless features
        # df = df.sort_index(level=df.index.names).reset_index()
        # del df['Quarter end']
        #
        # # Encode
        # # todo: scale (and normalise) categorical variables? Never!
        # encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        # train_data['Stock'] = encoder.fit_transform(train_data[['Stock']])

        # Pad data
        data = self.padder.transform(data)

        # Extract X, y
        y, X = data.pop('Label'), data

        # Reshape as ndarrays (n_stocks, n_timestamps, n_features)
        X = X.values.reshape(-1, self.padder.length, len(X.columns))
        y = y.values.reshape(-1, self.padder.length, 1)

        return X, y


if __name__ == '__main__':
    # Parameters
    use_augmentation = True  # Augment training data via variance scaling, may cause data leak

    # Load companies quarterly reports
    df = pd.read_csv('historical_qrs.csv')

    # Clean data
    df = clean(df)

    # Split training set, test set and validation set
    stocks = df.index.get_level_values('Stock').unique()
    train_stocks, test_stocks = train_test_split(stocks, test_size=0.2, shuffle=True, random_state=42)
    train_data, test_data = df.loc[train_stocks, :], df.loc[test_stocks, :]

    stocks = train_data.index.get_level_values('Stock').unique()
    train_stocks, val_stocks = train_test_split(stocks, test_size=0.25, shuffle=True, random_state=42)
    train_data, val_data = df.loc[train_stocks, :], df.loc[val_stocks, :]

    # Imputation
    print("Imputing...")
    # imputer = StockFundamentalDataImputer(train_data.drop(columns=['Stock', 'Quarter end']).columns)
    imputer = StocksImputer(method='linear', limit_direction='both')
    train_data = imputer.fit_transform(train_data)
    test_data = imputer.transform(test_data)
    val_data = imputer.transform(val_data)

    # Remove outliers in absolute values?

    # Data augmentation
    # if use_augmentation:
    #     train_data = augment(train_data)

    # Feature engineering
    print("Engineering features...")
    train_data = engineer_features(train_data, add_stock_info=False)
    test_data = engineer_features(test_data, add_stock_info=False)
    val_data = engineer_features(val_data, add_stock_info=False)

    # Remove outliers
    outlier_handler = OutlierMinMaxTransformer()
    train_data = outlier_handler.fit_transform(train_data)
    test_data = outlier_handler.transform(test_data)
    val_data = outlier_handler.transform(val_data)

    # Scale
    # todo: switch to MinMaxScaler, try StandardScaler
    scaler = MinMaxScaler()
    features = train_data.drop(columns=['Label']).columns
    train_data[features] = scaler.fit_transform(train_data[features])
    test_data[features] = scaler.transform(test_data[features])
    val_data[features] = scaler.transform(val_data[features])

    # Feature selection
    # features = train_data.drop(columns=['Label', 'Stock', 'Quarter end']).columns
    # features = train_data.drop(columns=['Label', 'Stock']).columns
    # selector = SelectKBest(f_classif, 30)
    # val_data[features] = selector.fit_transform(val_data[features], val_data['Label'])
    # train_data[features] = selector.transform(train_data[features])
    # test_data[features] = selector.transform(test_data[features])

    # # Reset index and remove useless features
    # df = df.sort_index(level=df.index.names).reset_index()
    # del df['Quarter end']
    #
    # # Encode
    # # todo: scale (and normalise) categorical variables? Never!
    # encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    # train_data['Stock'] = encoder.fit_transform(train_data[['Stock']])
    # test_data['Stock'] = encoder.transform(test_data[['Stock']])
    # val_data['Stock'] = encoder.transform(val_data[['Stock']])

    # Pad data
    padder = PaddingTransformer(padding='pre', truncating='pre', dtype='float')
    train_data = padder.fit_transform(train_data)
    test_data = padder.transform(test_data)
    val_data = padder.transform(val_data)

    # Save data
    train_data.to_csv('datasets/train_data.csv')
    test_data.to_csv('datasets/test_data.csv')
    val_data.to_csv('datasets/val_data.csv')

    # Extract X, y
    y_train, X_train = train_data.pop('Label'), train_data
    y_test, X_test = test_data.pop('Label'), test_data
    y_val, X_val = val_data.pop('Label'), val_data

    # Reshape as ndarrays (n_stocks, n_timestamps, n_features)
    X_train = X_train.values.reshape(-1, padder.length, len(X_train.columns))
    y_train = y_train.values.reshape(-1, padder.length, 1)
    X_test = X_test.values.reshape(-1, padder.length, len(X_test.columns))
    y_test = y_test.values.reshape(-1, padder.length, 1)
    X_val = X_val.values.reshape(-1, padder.length, len(X_val.columns))
    y_val = y_val.values.reshape(-1, padder.length, 1)

    # save_as(X_train, 'datasets/X_train.pkl')
    # save_as(y_train, 'datasets/y_train.pkl')
    # save_as(X_test, 'datasets/X_test.pkl')
    # save_as(y_test, 'datasets/y_test.pkl')
    # save_as(X_val, 'datasets/X_val.pkl')
    # save_as(y_val, 'datasets/y_val.pkl')
