from pickle import load, dump
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn.preprocessing import OneHotEncoder
from tslearn.clustering import TimeSeriesKMeans

from preprocess import PadTruncateTransformer


class Parser:
    def __init__(self):
        self.encoder = OneHotEncoder()
        self.kmeans = TimeSeriesKMeans(n_clusters=5, max_iter=10, verbose=1, n_jobs=-1, random_state=42)

    def fit(self, X):
        self.encoder.fit(X)
        self.kmeans.fit(X)
        return self

    def transform(self, X):
        X = self.encoder.transform(X)
        cluster_predictions = self.kmeans.predict(X)
        for i in range(cluster_predictions):
            cluster = [cluster_predictions[i]] * X.shape[1]
            X[i] = np.hstack((X[i], cluster))
        return X

    def fit_transform(self, X):
        self.fit(X)
        X = self.transform(X)
        return X


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def make_model(sequence_length, feature_count):
    # regularisers = [L1L2(l1=0.0, l2=0.0), L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01), L1L2(l1=0.01, l2=0.01)]

    model = Sequential([
        Masking(mask_value=0.0, input_shape=(sequence_length, feature_count)),

        LSTM(15, return_sequences=True, stateful=False, kernel_regularizer=L1L2(l1=0.01, l2=0.01)),
        Dropout(0.1),

        # LSTM(8, return_sequences=True, stateful=True),
        # Dropout(0.2),

        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
    model.summary()
    return model


def make_dummy_y(y):
    y = (np.arange(y.max() + 1) == y[..., None]).astype(int)
    y = y.reshape(-1, y.shape[1], y.shape[3])
    return y


if __name__ == '__main__':
    # Load data
    with open('datasets/X_train.pkl', 'rb') as X_file, open('datasets/y_train.pkl', 'rb') as y_file:
        X_train, y_train = load(X_file), load(y_file)
    with open('datasets/X_val.pkl', 'rb') as X_file, open('datasets/y_val.pkl', 'rb') as y_file:
        X_val, y_val = load(X_file), load(y_file)

    # Make dummy y
    # y_train = make_dummy_y(y_train)
    # y_val = make_dummy_y(y_val)

    # Parse train data
    # parser = Parser()
    # X_train = parser.fit_transform(X_train)

    # Fit model
    model = make_model(X_train.shape[1], X_train.shape[2])
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, min_delta=0, verbose=1, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=64, epochs=80, callbacks=[early_stopping])

    # Save model and parser
    model.save('saved-models/LSTM_regression_model.hdf5')
    # with open('saved-models/parser.pkl') as f:
    #     dump(parser, f)

    # "Accuracy"
    # plt.plot(history.history['f1_m'])
    # plt.plot(history.history['val_f1_m'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    # "Loss"
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()

    # Evaluate
    with open('datasets/X_test.pkl', 'rb') as X_file, open('datasets/y_test.pkl', 'rb') as y_file:
        X_test, y_test = load(X_file), load(y_file)
    y_test = make_dummy_y(y_test)
    # X_test = parser.transform(X_test)
    model = load_model('saved-models/LSTM_regression_model.hdf5')
    model.evaluate(X_test, y_test)
    # - loss: 0.0121 - mean_squared_error: 0.0021

    # - loss: 0.4041 - f1_m: 0.7806 - precision_m: 0.9144 - recall_m: 0.6815
    # - loss: 0.7480 - f1_m: 0.4390 - precision_m: 0.9647 - recall_m: 0.2842
    # test results on data that was not used to generate augmented data
    # much lower f1 suggests data leak? maybe from sliding window on original data set?
    # significant overlap in time period among and across training and test examples - a source of bias?
    # may need larger dataset for data on more companies over a longer time period
    # todo: scrape data from SEC
