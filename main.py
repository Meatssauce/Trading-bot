from pickle import load, dump
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def make_model(sequence_length, feature_count):
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=(sequence_length, feature_count)),
        Dropout(0.2),

        LSTM(256, return_sequences=True),
        Dropout(0.2),

        LSTM(256, return_sequences=True),
        Dropout(0.2),

        LSTM(256),

        Dense(1, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.summary()
    return model


with open('data/X.pkl', 'rb') as X_file, open('data/y.pkl', 'rb') as y_file:
    X, y = load(X_file), load(y_file)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_model(data.shape[1], data.shape[2])
model.fit(X_train, y_train, validation_split=0.2)
model.evaluate(X_test, y_test)

model.save('save-models/LSTM_model.hdf5')
