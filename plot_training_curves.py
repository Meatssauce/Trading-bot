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

from main import make_model, make_dummy_y, f1_m
from tensorflow.keras.metrics import categorical_crossentropy


def plot_learning_curves(model, X_train, y_train, X_val, y_val):
    train_errors, val_errors = [], []
    for m in range(1, len(X_train), 40)[:3]:
        model.fit(X_train[:m], y_train[:m], batch_size=32, epochs=5)
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(categorical_crossentropy(y_train[:m], y_train_predict))
        val_errors.append(categorical_crossentropy(y_val, y_val_predict))
    plt.plot(train_errors, "r-+", linewidth=2, label="train")
    plt.plot(val_errors, "b-", linewidth=3, label="val")


# Load data
with open('datasets/X_train.pkl', 'rb') as X_file, open('datasets/y_train.pkl', 'rb') as y_file:
    X_train, y_train = load(X_file), load(y_file)
with open('datasets/X_test.pkl', 'rb') as X_file, open('datasets/y_test.pkl', 'rb') as y_file:
    X_test, y_test = load(X_file), load(y_file)
with open('datasets/X_val.pkl', 'rb') as X_file, open('datasets/y_val.pkl', 'rb') as y_file:
    X_val, y_val = load(X_file), load(y_file)

# Make dummy y
y_train = make_dummy_y(y_train)
y_test = make_dummy_y(y_test)
y_val = make_dummy_y(y_val)

# Plote learning curves
plot_learning_curves(make_model(X_train.shape[1], X_train.shape[2]), X_train, y_train, X_val, y_val)
