from pickle import load, dump
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tslearn.clustering import TimeSeriesKMeans

with open('data/X_train.pkl', 'rb') as X_file, open('data/y_train.pkl', 'rb') as y_file:
    X, y = load(X_file), load(y_file)
X, y = np.stack(list(X.values())), np.stack(list(y.values()))
y = to_categorical(y, 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kmeans = TimeSeriesKMeans(n_clusters=8, max_iter=10, verbose=1, n_jobs=-1, random_state=42)
kmeans.fit(X_train)
print(kmeans.predict(X_test))
