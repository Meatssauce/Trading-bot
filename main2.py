import os
import random
from joblib import load, dump
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.callbacks import EarlyStopping
# from keras import backend as K
# from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
# from tslearn.clustering import TimeSeriesKMeans

from preprocess2 import clean, Parser


# class Parser:
#     def __init__(self):
#         self.encoder = OneHotEncoder()
#         self.kmeans = TimeSeriesKMeans(n_clusters=5, max_iter=10, verbose=1, n_jobs=-1, random_state=42)
#
#     def fit(self, X):
#         self.encoder.fit(X)
#         self.kmeans.fit(X)
#         return self
#
#     def transform(self, X):
#         X = self.encoder.transform(X)
#         cluster_predictions = self.kmeans.predict(X)
#         for i in range(cluster_predictions):
#             cluster = [cluster_predictions[i]] * X.shape[1]
#             X[i] = np.hstack((X[i], cluster))
#         return X
#
#     def fit_transform(self, X):
#         self.fit(X)
#         X = self.transform(X)
#         return X

#
# def recall_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall
#
#
# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision
#
#
# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def make_model(timestamps, features):
    # regularisers = [L1L2(l1=0.0, l2=0.0), L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01), L1L2(l1=0.01, l2=0.01)]

    model = Sequential([
        Masking(mask_value=0., input_shape=(timestamps, features)),

        LSTM(15, return_sequences=True, stateful=False, kernel_regularizer=L1L2(l1=0.001, l2=0.001)),
        Dropout(0.1),

        # LSTM(8, return_sequences=True, stateful=True),
        # Dropout(0.2),

        Dense(1)
    ])
    optimizer = optimizers.Adam(clipvalue=0.5)  # to prevent exploding gradient
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
    model.summary()
    return model


# def make_dummy_y(y):
#     y = (np.arange(y.max() + 1) == y[..., None]).astype(int)
#     y = y.reshape(-1, y.shape[1], y.shape[3])
#     return y


if __name__ == '__main__':
    # Load data
    df = pd.read_csv('historical_qrs.csv')

    # Clean data
    df = clean(df)

    # Split training set, test set and validation set (6:2:2)
    stocks = df.index.get_level_values('Stock').unique()
    train_stocks, test_stocks = train_test_split(stocks, test_size=0.2, shuffle=True, random_state=42)
    test_data = df.loc[test_stocks, :]

    train_stocks, val_stocks = train_test_split(train_stocks, test_size=0.25, shuffle=True, random_state=42)
    train_data, val_data = df.loc[train_stocks, :], df.loc[val_stocks, :]

    # Preprocess
    parser = Parser()
    X_train, y_train = parser.fit_transform(train_data)
    X_val, y_val = parser.transform(val_data)

    # Fit model
    model = make_model(X_train.shape[1], X_train.shape[2])
    early_stopping = EarlyStopping(monitor='loss', patience=10, min_delta=0, verbose=1,
                                   restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=64, epochs=80,
                        callbacks=[early_stopping])

    # Generate new id, then save model, parser and relevant files
    existing_ids = [int(name) for name in os.listdir('saved-models/') if name.isnumeric()]
    run_id = random.choice(list(set(range(0, 1000)) - set(existing_ids)))
    save_directory = f'saved-models/{run_id:03d}/'
    os.makedirs(os.path.dirname(save_directory), exist_ok=True)

    model.save(save_directory + 'LSTM_regression_model.hdf5')
    with open(save_directory + 'LSTM_regression_parser', 'wb') as f:
        dump(parser, f, compress=3)
    with open(save_directory + 'train_history', 'wb') as f:
        dump(history.history, f, compress=3)
    pd.DataFrame(history.history).to_csv(save_directory + 'train_history.csv')

    # Plot accuracy
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model accuracy')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(save_directory + 'accuracy.png')

    # Plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(save_directory + 'loss.png')

    # Evaluate
    # run_id = 792
    # with open(f'saved-models/{run_id}/LSTM_regression_parser', 'rb') as f:
    #     parser = load(f)
    # model = load_model(f'saved-models/{run_id}/LSTM_regression_model.hdf5')
    X_test, y_test = parser.transform(test_data)
    scores = model.evaluate(X_test, y_test)
    try:
        df_scores = pd.read_csv('saved-models/scores.csv')
        df_scores.loc[len(df_scores)] = [run_id] + scores + [pd.Timestamp.now(tz='Australia/Melbourne')]
    except FileNotFoundError:
        row = [[run_id] + scores + [pd.Timestamp.now(tz='Australia/Melbourne')]]
        df_scores = pd.DataFrame(row, columns=['id'] + list(model.metrics_names) + ['time'])
    df_scores.to_csv('saved-models/scores.csv', index=False)

    # - loss: 0.0121 - mean_squared_error: 0.0021

    # - loss: 0.4041 - f1_m: 0.7806 - precision_m: 0.9144 - recall_m: 0.6815
    # - loss: 0.7480 - f1_m: 0.4390 - precision_m: 0.9647 - recall_m: 0.2842
    # test results on data that was not used to generate augmented data
    # much lower f1 suggests data leak? maybe from sliding window on original data set?
    # significant overlap in time period among and across training and test examples - a source of bias?
    # may need larger dataset for data on more companies over a longer time period
    # todo: scrape data from SEC
