from enum import Enum
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, MinMaxScaler, StandardScaler
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from category_encoders import BinaryEncoder

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


class OutlierTransformer(TransformerMixin):
    def __init__(self, columns=None, fill: str = 'median', **kwargs):
        """
        Create a transformer to remove outliers.

        Returns:
            object: to be used as a transformer method as part of Pipeline()
        """

        self.fill = fill
        self.columns = columns
        self.type = None
        self.n_columns = None
        self.q1s = None
        self.q3s = None
        self.medians = None
        self.means = None

    def fit(self, X: np.ndarray or pd.DataFrame, y=None, **fit_params):
        self.type = type(X)
        self.n_columns = X.shape[1]

        if isinstance(X, np.ndarray):
            for i in range(X.shape[1]):
                self.q1s[i], self.q3s[i] = np.quantile(X[i], [0.25, 0.75])
                self.medians[i] = np.median(X[i])
                self.means[i] = np.mean(X[i])

        elif isinstance(X, pd.DataFrame):
            if self.columns is None:
                self.columns = X.columns

            self.q1s = X[self.columns].quantile(0.25)
            self.q3s = X[self.columns].quantile(0.75)
            self.medians = X[self.columns].median()
            self.means = X[self.columns].mean()

        else:
            raise TypeError(f'Invalid input type. Expected np.ndarray or pd.DataFrame but got {type(X)}')

        return self

    def transform(self, X: np.ndarray or pd.DataFrame):
        if not isinstance(X, self.type):
            raise TypeError(f'Inconsistent input type. Expected {self.type} but got {type(X)}')
        if isinstance(X, np.ndarray) and X.shape[1] != self.n_columns:
            raise TypeError(f'Inconsistent input shape. Expected {self.n_columns} but got {X.shape[1]}')

        # Find and replace outliers
        idx = X.shape[1] if self.columns is None else self.columns
        for i, q1, q3 in zip(idx, self.q1s, self.q3s):
            iqr = q3 - q1
            min_val, max_val = q1 - 1.5 * iqr, q3 + 1.5 * iqr

            if self.fill == 'median':
                X[i] = np.where((X[i] < min_val) | (X[i] > max_val), self.medians[i], X[i])
            elif self.fill == 'mean':
                X[i] = np.where((X[i] < min_val) | (X[i] > max_val), self.means[i], X[i])
            elif self.fill == 'nan':
                X[i] = np.where((X[i] < min_val) | (X[i] > max_val), np.nan, X[i])
            elif self.fill == 'nearest':
                X[i] = np.where(X[i] < min_val, min_val, X[i])
                X[i] = np.where(X[i] > max_val, max_val, X[i])
            else:
                raise ValueError('Invalid fill method')

        return X


class OutlierMinMaxTransformer(TransformerMixin):
    def __init__(self):
        self.q1s = None
        self.q3s = None

    def fit(self, X, y=None):
        self.q1s = X.quantile(0.25).tolist()
        self.q3s = X.quantile(0.75).tolist()
        return self

    def transform(self, X):
        for col, q1, q3 in zip(X.columns, self.q1s, self.q3s):
            iqr = q3 - q1
            min_val, max_val = (q1 - 1.5 * iqr), (q3 + 1.5 * iqr)
            X[col] = np.where(X[col] < min_val, min_val, X[col])
            X[col] = np.where(X[col] > max_val, max_val, X[col])
        return X


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
    def __init__(self, return_full_df: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.return_full_df = return_full_df
        self.simple_imputer = None
        self.interpolation_imputer = None
        self.outlier_transformer = None
        self.scaler = None
        self.encoder = None
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
        self.outlier_transformer = OutlierTransformer(columns=data.drop(columns=['Label']).columns, fill='median')
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

        # # Encode
        # # todo: scale (and normalise) categorical variables? Never!
        # data['Stock'] = data.index.get_level_values('Stock')
        # self.encoder = BinaryEncoder()
        # data = self.encoder.fit_transform(data)

        # Pad data
        self.padder = PaddingTransformer(padding='pre', truncating='pre', dtype='float')
        data = self.padder.fit_transform(data)

        if self.return_full_df:
            return data

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

        # # Encode stock to capture characteristic behaviour
        # # todo: scale (and normalise) categorical variables? Never!
        # data['Stock'] = data.index.get_level_values('Stock')
        # data = self.encoder.transform(data)

        # Pad data
        data = self.padder.transform(data)

        if self.return_full_df:
            return data

        # Extract X, y
        y, X = data.pop('Label'), data

        # Reshape as ndarrays (n_stocks, n_timestamps, n_features)
        X = X.values.reshape(-1, self.padder.length, len(X.columns))
        y = y.values.reshape(-1, self.padder.length, 1)

        return X, y


if __name__ == '__main__':
    # # Parameters
    # use_augmentation = True  # Augment training data via variance scaling, may cause data leak

    # Load companies quarterly reports
    df = pd.read_csv('historical_qrs.csv')

    # Clean data
    df = clean(df)

    # Split training set, test set and validation set (6:2:2)
    stocks = df.index.get_level_values('Stock').unique()
    train_stocks, test_stocks = train_test_split(stocks, test_size=0.2, shuffle=True, random_state=42)
    test_data = df.loc[test_stocks, :]

    train_stocks, val_stocks = train_test_split(train_stocks, test_size=0.25, shuffle=True, random_state=42)
    train_data, val_data = df.loc[train_stocks, :], df.loc[val_stocks, :]

    parser = Parser(return_full_df=True)
    train_data = parser.fit_transform(train_data)

    profile = ProfileReport(train_data, minimal=True)
    profile.to_file('PostProcessReport.html')
