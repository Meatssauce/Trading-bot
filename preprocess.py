import os
import random
from enum import Enum
import pandas as pd
import numpy as np
from category_encoders import BinaryEncoder
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
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


def save_as(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        dump(obj, file)


def save_as_x_y(data, X_path, y_path):
    y = {k: df.pop('Label') for k, df in data.items()}
    X = data

    save_as(X, X_path)
    save_as(y, y_path)


def clean(df):
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
            # numeric_columns = X.drop(columns=['Stock', 'Quarter end']).columns
            numeric_columns = X.drop(columns=['Stock']).columns

            # Imputation via interpolation
            X.loc[subset, numeric_columns] = X.loc[subset, numeric_columns].interpolate(
                method=self.method, limit_direction=self.limit_direction, axis=0)  # spline may be better?
        X[self.columns] = pd.DataFrame(self.imputer.transform(X[self.columns]), columns=self.columns, index=X.index)

        return X


def engineer_features(df, add_stock_info=False, classification=True):
    for stock in tqdm(df['Stock'].unique()):
        subset = df['Stock'] == stock
        # numeric_columns = df.drop(columns=['Stock', 'Quarter end']).columns
        numeric_columns = df.drop(columns=['Stock']).columns

        # Replace values with percent difference or change
        df.loc[subset, numeric_columns] = df.loc[subset, numeric_columns].pct_change(periods=1)

        # Replace infinite values and nan with 0
        df.loc[subset, numeric_columns] = df.loc[subset, numeric_columns].replace([np.inf, -np.inf], 0)
        df.loc[subset, numeric_columns] = df.loc[subset, numeric_columns].fillna(0)

        # Create the class 'Label' determining if a quarterly reports improvement is a buy, hold, or sell.
        # shifted by -1 to know if the prices will increase/decrease in the next quarter
        if classification:
            df.loc[subset, 'Label'] = df[subset].apply(class_creation, axis=1).shift(-1)
        else:
            df.loc[subset, 'Label'] = df.loc[subset, 'Price'].shift(-1)

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
        # numeric_columns = [col for col in df.columns if col not in ['Stock', 'Quarter end']]
        numeric_columns = [col for col in df.columns if col != 'Stock']
        new_df[numeric_columns] = new_df[numeric_columns] * scalar
        result = pd.concat([result, new_df])

    return result


class PadTruncateTransformer(TransformerMixin):
    def __init__(self, maxlen=None, padding='pre', truncating='pre', dtype='float'):
        self.length = None
        self.padding = padding
        self.truncating = truncating
        self.dtype = dtype

    def fit(self, X, y=None, quantile=0.9):
        self.length = int(np.quantile(X['Stock'].value_counts(), quantile))
        return self

    def transform(self, X):
        # X = pd.DataFrame(pad_sequences(X.values, padding=self.padding, truncating=self.truncating, dtype=self.dtype), columns=X.columns)
        X = self._pad_data(X, self.length, self.padding, self.truncating)
        return X

    def _pad_data(self, df, length, padding, truncating):
        """
        Transforms all dataframes within the data to a fixed length via padding or truncating. If padding, pad with 0s and
        Hold for the label.

        :param df: dataframe
        :param length: target row count for the dataframes
        :param padding: {'pre', 'post'} if 'pre' then pad from the start; if 'post' pad from the end
        :param truncating: {'pre', 'post'} if 'pre' then truncate from the start; if 'post' truncate from the end
        :return: a dictionary of dataframes indexed by date
        """

        # assert isinstance(df, dict)
        assert padding in ['pre', 'post'] and truncating in ['pre', 'post']

        segments = []
        for stock in tqdm(df['Stock'].unique()):
            df_subset = df[df['Stock'] == stock]
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
                df_subset = df_subset[:padding_length] if truncating == 'post' else df_subset[-padding_length:]
                segments.append(df_subset)
            else:
                segments.append(df_subset)
        result = pd.concat(segments)
        result = result.reset_index(drop=True)

        return result


# desired_length = 101

if __name__ == '__main__':
    # Parameters
    use_augmentation = True  # Augment training data via variance scaling, may cause data leak

    # Load companies quarterly reports
    try:
        df = pd.read_csv('datasets/historical_qrs.csv')
        df = clean(df)
    except Exception:
        df = pd.read_csv('datasets/historical_qrs.csv')
        df = clean(df)
        df.to_csv('datasets/clean_historical_qrs.csv', index=True)

    # Split training set, test set and validation set
    train_stocks, test_stocks = train_test_split(df['Stock'].unique(), test_size=0.2, random_state=42)
    train_data, test_data = df[df['Stock'].isin(train_stocks)], df[df['Stock'].isin(test_stocks)]
    train_stocks, val_stocks = train_test_split(train_data['Stock'].unique(), test_size=0.25, random_state=42)
    train_data, val_data = train_data[train_data['Stock'].isin(train_stocks)], \
                           train_data[train_data['Stock'].isin(val_stocks)]

    # Imputation
    # imputer = StockFundamentalDataImputer(train_data.drop(columns=['Stock', 'Quarter end']).columns)
    imputer = StockFundamentalDataImputer(train_data.drop(columns=['Stock']).columns)
    train_data = imputer.fit_transform(train_data)
    test_data = imputer.transform(test_data)
    val_data = imputer.transform(val_data)

    # Feature engineering
    train_data = engineer_features(train_data, add_stock_info=False, classification=False)
    test_data = engineer_features(test_data, add_stock_info=False, classification=False)
    val_data = engineer_features(val_data, add_stock_info=False, classification=False)

    # Data augmentation
    # if use_augmentation:
    #     train_data = augment(train_data)

    # Remove outliers
    # outlier_remover = OutlierNullifier()
    # features = train_data.drop(columns=['Label', 'Stock']).columns
    # train_data[features] = outlier_remover.fit_transform(train_data[features])
    # test_data[features] = outlier_remover.transform(test_data[features])
    # val_data[features] = outlier_remover.transform(val_data[features])
    # imputer = StockFundamentalDataImputer(train_data.drop(columns=['Stock']).columns)
    # train_data = imputer.fit_transform(train_data)
    # test_data = imputer.transform(test_data)
    # val_data = imputer.transform(val_data)

    # Encode
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train_data['Stock'] = encoder.fit_transform(train_data[['Stock']])
    test_data['Stock'] = encoder.transform(test_data[['Stock']])
    val_data['Stock'] = encoder.transform(val_data[['Stock']])

    # Scale
    scaler = RobustScaler()
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

    # Pad data
    padder = PadTruncateTransformer(padding='pre', truncating='pre', dtype='float')
    train_data = padder.fit_transform(train_data)
    test_data = padder.transform(test_data)
    val_data = padder.transform(val_data)

    # Save data
    train_data.to_csv('datasets/train_data.csv', index=False)
    test_data.to_csv('datasets/test_data.csv', index=False)
    val_data.to_csv('datasets/val_data.csv', index=False)

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

    save_as(X_train, 'datasets/X_train.pkl')
    save_as(y_train, 'datasets/y_train.pkl')
    save_as(X_test, 'datasets/X_test.pkl')
    save_as(y_test, 'datasets/y_test.pkl')
    save_as(X_val, 'datasets/X_val.pkl')
    save_as(y_val, 'datasets/y_val.pkl')
