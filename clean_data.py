import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import dump, load


def setting_index(df):
    """
    Returns a sorted datetime index
    """
    df['Quarter end'] = pd.to_datetime(df['Quarter end'])
    df = df.set_index("Quarter end")
    return df.sort_index(ascending=False)


def class_creation(df, threshold=3):
    """
    Creates classes of:
    - buy(1)
    - hold(2)
    - sell(0)

    Threshold can be changed to fit whatever price percentage change is desired
    """
    if df['Price high'] >= threshold and df['Price low'] >= threshold:
        # Buys
        return 1

    elif df['Price high'] <= -threshold and df['Price low'] <= -threshold:
        # Sells
        return 0

    else:
        # Holds
        return 2


# stocks_df = pd.read_csv('historical_qrs.csv', index_col=['Ticker', 'Row ID'])
#
# df_by_ticker = {}
# for ticker in tqdm(stocks_df.index.get_level_values('Ticker')):
#     df_by_ticker[ticker] = stocks_df.loc[ticker, :]

"""
Idea:
Takes a five year period of QRs as input and outputs buy, sell or hold for next quarter
- if a company has been traded for less than five years, missing data is replaced with nan (top padded)
- assumes future performance is not always independent from the past (rejects random walk hypothesis)
# - takes relative change as input but also considers relative total assets (not the change thereof)
# - assumes relative size of a company may matter too 
- if there exists data for a company over a period of N quarters than N-2 different training data can be generated 
using a sliding window method 
"""

qr_by_ticker = load('qr_by_ticker.joblib')

# Setting the index as the Date
for i in tqdm(qr_by_ticker.keys()):
    qr_by_ticker[i] = setting_index(qr_by_ticker[i])

# Replacing all "None" values with NaN
for i in tqdm(qr_by_ticker.keys()):
    qr_by_ticker[i] = qr_by_ticker[i].replace("None", np.nan)

# Creating a new dictionary that contains the numerical values, then converting all values to numeric values
num_df = {}
for i in tqdm(qr_by_ticker.keys()):
    num_df[i] = qr_by_ticker[i].apply(pd.to_numeric)

# Replacing values with percent difference or change
pcnt_df = {}
for i in tqdm(num_df.keys()):
    pcnt_df[i] = num_df[i].pct_change(periods=1)

# Replacing infinite values with NaN
for i in tqdm(pcnt_df.keys()):
    pcnt_df[i] = pcnt_df[i].replace([np.inf, -np.inf], np.nan)

# Creating a new DataFrame that contains the class 'Decision' determining if a quarterly reports improvement is a
# buy, hold, or sell.
new_df = {}
for i in tqdm(pcnt_df.keys()):
    # Assigning the new DF
    new_df[i] = pcnt_df[i]

    # Creating the new column with the classes, shifted by -1 in order to know if the prices will increase/decrease in
    # the next quarter.
    new_df[i]['Decision'] = new_df[i].apply(class_creation, axis=1).shift(-1)

# Excluding the first and last rows
for i in tqdm(new_df.keys()):
    new_df[i] = new_df[i][1:-1]

# ===============================================

# Combining all stock DFs into one
big_df = pd.DataFrame()
for i in tqdm(pcnt_df.keys()):
    big_df = big_df.append(new_df[i], sort=False)

# Filling the NaNs with 0
big_df = big_df.fillna(0)

# Resetting the index because we no longer need the dates
big_df = big_df.reset_index(drop=True)

# Dropping the price related columns to prevent data leakage
big_df = big_df.drop(['Price', 'Price high', 'Price low'], axis=1)

# Exporting the final DataFrame
with open("main_df.pkl", 'wb') as fp:
    pickle.dump(big_df, fp)
