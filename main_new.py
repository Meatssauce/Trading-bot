from pickle import load, dump
import pandas as pd
import numpy as np
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

from clean_data_new import PadTruncateTransformer


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
    regularisers = [L1L2(l1=0.0, l2=0.0), L1L2(l1=0.01, l2=0.0), L1L2(l1=0.0, l2=0.01), L1L2(l1=0.01, l2=0.01)]
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(sequence_length, feature_count)),

        LSTM(15, return_sequences=True, stateful=False, kernel_regularizer=L1L2(l1=0.1, l2=0.1)),
        Dropout(0.1),

        # LSTM(8, return_sequences=True, stateful=True),
        # Dropout(0.2),

        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[f1_m, precision_m, recall_m])
    model.summary()
    return model


robust_test = False
if not robust_test:
    # Load data
    with open('datasets/X_train.pkl', 'rb') as X_file, open('datasets/y_train.pkl', 'rb') as y_file:
        X_train, y_train = load(X_file), load(y_file)
    with open('datasets/X_test.pkl', 'rb') as X_file, open('datasets/y_test.pkl', 'rb') as y_file:
        X_test, y_test = load(X_file), load(y_file)
    with open('datasets/X_val.pkl', 'rb') as X_file, open('datasets/y_val.pkl', 'rb') as y_file:
        X_val, y_val = load(X_file), load(y_file)

    # X_train, y_train = np.stack(list(X_train.values())), np.stack(list(y_train.values()))
    y_train = (np.arange(y_train.max()+1) == y_train[..., None]).astype(int)
    y_train = y_train.reshape(-1, y_train.shape[1], y_train.shape[3])
    # X_test, y_test = np.stack(list(X_test.values())), np.stack(list(y_test.values()))
    y_test = (np.arange(y_test.max()+1) == y_test[..., None]).astype(int)
    y_test = y_test.reshape(-1, y_test.shape[1], y_test.shape[3])
    # X_val, y_val = np.stack(list(X_val.values())), np.stack(list(y_val.values()))
    y_val = (np.arange(y_val.max()+1) == y_val[..., None]).astype(int)
    y_val = y_val.reshape(-1, y_val.shape[1], y_val.shape[3])

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X,
    #     y,
    #     test_size=0.2,
    #     # shuffle=False,
    #     random_state=42
    # )

    # Parse train data
    # parser = Parser()
    # X_train = parser.fit_transform(X_train)

    # Fit model
    model = make_model(X_train.shape[1], X_train.shape[2])
    early_stopping = EarlyStopping(monitor='val_f1_m', mode='max', patience=10, min_delta=0, verbose=1, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=500, callbacks=[early_stopping])

    # X_test = parser.transform(X_test)
    model.evaluate(X_test, y_test)
    # - loss: 0.4041 - f1_m: 0.7806 - precision_m: 0.9144 - recall_m: 0.6815

    # Save model and parser
    model.save('saved-models/LSTM_model.hdf5')
    # with open('saved-models/parser.pkl') as f:
    #     dump(parser, f)
else:
    # Parse test data and evaluate
    with open('datasets/X_test.pkl', 'rb') as X_file, open('datasets/y_test.pkl', 'rb') as y_file:
        X_test2, y_test2 = load(X_file), load(y_file)
    # X_test2, y_test2 = np.stack(list(X_test2.values())), np.stack(list(y_test2.values()))
    y_test2 = to_categorical(y_test2, 3)

    model = load_model('saved-models/LSTM_model.hdf5',
                       custom_objects={'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[f1_m, precision_m, recall_m])
    model.evaluate(X_test2, y_test2)
    # - loss: 0.7480 - f1_m: 0.4390 - precision_m: 0.9647 - recall_m: 0.2842
    # test results on data that was not used to generate augmented data
    # much lower f1 suggests data leak? maybe from sliding window on original data set?
    # significant overlap in time period among and across training and test examples - a source of bias?
    # may need larger dataset for data on more companies over a longer time period
    # todo: scrape data from SEC
