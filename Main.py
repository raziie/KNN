import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import KNN

# read data
data = pd.read_csv('./archive/IRIS.csv')
print(data)

# shuffle the DataFrame rows
shuffled_data = data.sample(frac=1)
print(shuffled_data)

# up_down, left_right
# extract features and labels
X = shuffled_data.iloc[:, :-1].values
Y = shuffled_data.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
test_size = math.floor(0.25 * len(X))
X_test, X_train = X[:test_size], X[test_size:]
Y_test, Y_train = Y[:test_size], Y[test_size:]

print(X_train)
print(Y_train)
print(X_test)
print(Y_test)

# Standardize features by removing the mean and scaling to unit variance
# x = (x - m) / v
train_variance = np.var(X_train, dtype=np.float64)
train_mean = np.mean(X_train)
X_train = (X_train - train_mean) / train_variance

test_variance = np.var(X_test, dtype=np.float64)
test_mean = np.mean(X_test)
X_test = (X_test - test_mean) / test_variance

print(X_train)
print(X_test)

classifier = KNN.KNN(5)
classifier.fit(X_train, Y_train)
Y_prediction = classifier.predict(X_test)
print(Y_prediction == Y_test)
# print(np.max(np.abs(Y_prediction - Y_test)))
