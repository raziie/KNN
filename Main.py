import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import KNN


def my_train_test_split(data, test_size, random_state):
    # to get the same shuffle each time
    # if you change the random_state the result will be different
    np.random.seed(random_state)
    # shuffle the DataFrame rows
    shuffled_data = np.random.permutation(data)

    # print("--------------------shuffled data--------------------")
    # print(shuffled_data)

    # another way but not the same every time
    # shuffled_data = data.sample(frac=1)
    # # up_down, left_right
    # X = shuffled_data.iloc[:, :-1].values
    # Y = shuffled_data.iloc[:, -1].values

    # extract features and labels
    X = shuffled_data[:, :-1]
    Y = shuffled_data[:, -1]
    # without shuffle
    # X = data.iloc[:, :-1].values
    # Y = data.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    test_size = math.floor(test_size * len(X))
    X_test, X_train = X[:test_size], X[test_size:]
    Y_test, Y_train = Y[:test_size], Y[test_size:]

    # print("--------------------X train--------------------")
    # print(X_train)
    # print("--------------------Y train--------------------")
    # print(Y_train)
    # print("--------------------X test--------------------")
    # print(X_test)
    # print("--------------------Y test--------------------")
    # print(Y_test)

    return X_train, X_test, Y_train, Y_test


def standardize(X_train, X_test):
    # Standardize features by removing the mean and scaling to unit variance
    # x = (x - m) / v
    train_variance = np.var(X_train, dtype=np.float64)
    train_mean = np.mean(X_train)
    X_train = (X_train - train_mean) / train_variance

    test_variance = np.var(X_test, dtype=np.float64)
    test_mean = np.mean(X_test)
    X_test = (X_test - test_mean) / test_variance

    # print("--------------------Standardized X train--------------------")
    # print(X_train)
    # print("--------------------Standardized X test--------------------")
    # print(X_test)

    return X_train, X_test


if __name__ == "main":
    # read data
    dataset = pd.read_csv('./archive/IRIS.csv')
    # print("--------------------input data--------------------")
    # print(dataset)

    X_train, X_test, Y_train, Y_test = my_train_test_split(dataset, test_size=0.25, random_state=0)
    X_train, X_test = standardize(X_train, X_test)

    classifier = KNN.KNN(5)
    # fit the training data
    classifier.fit(X_train, Y_train)
    # make prediction
    Y_prediction = classifier.predict(X_test)
    # print("--------------------compare prediction with true values--------------------")
    # print(Y_test == Y_prediction)
    print("Accuracy = ", KNN.compute_accuracy(Y_test, Y_prediction), "%")

    print("--------------------result for [6, 3, 5, 2]--------------------")
    print(classifier.predict([[6, 3, 5, 2]]))

    # parameter tuning with results
    accuracies = classifier.tune_parameter(X_test, Y_test, 100)
    plt.plot(accuracies, color="skyblue")
    plt.show()

    # k_fold
    k_fold = 20
    folds = KNN.get_fold_indices(k_fold, X, Y)
    scores = []
    for [X_train, Y_train, X_test, Y_test] in folds:
        classifier = KNN.KNN(4)
        # fit the training data
        classifier.fit(X_train, Y_train)
        # make prediction
        Y_prediction = classifier.predict(X_test)
        scores.append(KNN.compute_accuracy(Y_test, Y_prediction))

    mean_accuracy = np.mean(scores)
    print("K-Fold Cross-Validation Scores:", scores)
    print("Mean Accuracy:", mean_accuracy)
