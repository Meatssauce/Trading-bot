import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import dump, load

from keras.preprocessing import sequence


def set_index_to_date(df):
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
sequences = load('qr_by_ticker.joblib')

for i in tqdm(sequences.keys()):
    # Setting the index as the Date
    sequences[i] = set_index_to_date(sequences[i])

    # Replacing all "None" values with NaN
    sequences[i] = sequences[i].replace("None", np.nan)

    # Converting all values to numeric values
    sequences[i] = sequences[i].apply(pd.to_numeric)

    # Replacing values with percent difference or change
    sequences[i] = sequences[i].pct_change(periods=1)

    # Replacing infinite values with NaN
    sequences[i] = sequences[i].replace([np.inf, -np.inf], np.nan)

    # Interpolate missing values
    sequences[i] = sequences[i].interpolate(method='linear', axis=0)  # use spline yields may better results?

    # Creating the class 'Decision' determining if a quarterly reports improvement is a buy, hold, or sell.
    # Creating the new column with the classes, shifted by -1 in order to know if the prices will increase/decrease in
    # the next quarter.
    sequences[i]['Decision'] = sequences[i].apply(class_creation, axis=1).shift(-1)

    # Excluding the first and last rows (cannot label)
    sequences[i] = sequences[i][1:-1]

    # Dropping the price related columns to prevent data leakage
    sequences[i] = sequences[i].drop(['Price', 'Price high', 'Price low'], axis=1)

# produces N-2 sets of data from N periods
# not recommended?

# Show distribution of company QR counts
sequence_lens = []
for df in sequences.values():
    sequence_lens.append(len(df))
# print(pd.Series(sequence_lens).describe())
# print(np.quantile(sequence_lens, 0.9))

# Padding the sequence with the values in last row or truncating to 90% quantile of QR counts
desired_length = np.quantile(sequence_lens, 0.9)
new_seq = []
for seq in sequences.values():
    padding_length = desired_length - len(seq)

    if padding_length > 0:
        last_value = seq[-1:]
        padding = pd.concat([last_value] * padding_length)
        seq = pd.concat([seq, padding])
        # padding = np.repeat(one_seq[-1], padding_length).reshape(38, padding_length).transpose()
        # one_seq = np.concatenate([one_seq, padding])

    new_seq.append(seq)

# Use only this line if padding with 0 instead of last row
final_seq = sequence.pad_sequences(new_seq, maxlen=desired_length, padding='post', dtype='float', truncating='post')
# todo: decide whether to use 0 or last row for padding
# todo: decide whether to pad to maximum length or desired length

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

# Exporting the final DataFrame
# with open("main_df.pkl", 'wb') as fp:
#     pickle.dump(sequences, fp)
