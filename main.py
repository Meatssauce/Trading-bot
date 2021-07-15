from pickle import load, dump
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn.preprocessing import OneHotEncoder
from tslearn.clustering import TimeSeriesKMeans


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
    model = Sequential([
        LSTM(17, return_sequences=True, input_shape=(sequence_length, feature_count)),
        Dropout(0.2),

        LSTM(8, return_sequences=True),
        Dropout(0.2),

        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[f1_m, precision_m, recall_m])
    model.summary()
    return model


test2 = False
if not test2:
    # Load data
    with open('data/X.pkl', 'rb') as X_file, open('data/y.pkl', 'rb') as y_file:
        X, y = load(X_file), load(y_file)
    X, y = np.stack(list(X.values())), np.stack(list(y.values()))
    y = to_categorical(y, 3)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        # shuffle=False,
        random_state=42
    )

    # Parse train data
    # parser = Parser()
    # X_train = parser.fit_transform(X_train)

    # Fit model
    model = make_model(X_train.shape[1], X_train.shape[2])
    early_stopping = EarlyStopping(monitor='val_f1_m', patience=1000, verbose=1, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.2, batch_size=32, epochs=10000, callbacks=[early_stopping])

    # X_test = parser.transform(X_test)
    model.evaluate(X_test, y_test)
else:
    # Parse test data and evaluate
    with open('data/X_val.pkl', 'rb') as X_file, open('data/y_val.pkl', 'rb') as y_file:
        X_test2, y_test2 = load(X_file), load(y_file)
    X_test2, y_test2 = np.stack(list(X_test2.values())), np.stack(list(y_test2.values()))
    y_test2 = to_categorical(y_test2, 3)

    model = load_model('saved-models/LSTM_model.hdf5',
                       custom_objects={'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[f1_m, precision_m, recall_m])
    model.evaluate(X_test2, y_test2)
    # - loss: 0.7480 - f1_m: 0.4390 - precision_m: 0.9647 - recall_m: 0.2842

# Save model and parser
# model.save('saved-models/LSTM_model.hdf5')
# with open('saved-models/parser.pkl') as f:
#     dump(parser, f)
