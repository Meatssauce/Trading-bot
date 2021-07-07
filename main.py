from pickle import load, dump

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from keras import backend as K


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
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


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


with open('data/X.pkl', 'rb') as X_file, open('data/y.pkl', 'rb') as y_file:
    X, y = load(X_file), load(y_file)

y = to_categorical(y, 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_model(X_train.shape[1], X_train.shape[2])
early_stopping = EarlyStopping(monitor='val_f1_m', patience=1000, verbose=1, restore_best_weights=True)
model.fit(X_train, y_train, validation_split=0.2, batch_size=32, epochs=10000, callbacks=[early_stopping])
model.evaluate(X_test, y_test)

model.save('save-models/LSTM_model.hdf5')
