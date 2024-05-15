import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import KNN


def extract_data_random_state(data, random_state):
    # to get the same shuffle each time
    # if you change the random_state the result will be different
    np.random.seed(random_state)
    # shuffle the DataFrame rows
    shuffled_data = np.random.permutation(data)
    # extract features and labels
    features = shuffled_data[:, :-1]
    labels = shuffled_data[:, -1]

    return features, labels


def extract_data_random(data):
    # another way but not the same every time
    shuffled_data = data.sample(frac=1).reset_index(drop=True)
    # up_down, left_right
    features = shuffled_data.iloc[:, :-1].values
    labels = shuffled_data.iloc[:, -1].values

    return features, labels


def extract_data_no_shuffle(data):
    # without shuffle
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values

    return features, labels


def split_data(features, labels, test_ratio):
    # Splitting the dataset into the Training set and Test set
    test_size = math.floor(test_ratio * len(features))
    f_test, f_train = features[:test_size], features[test_size:]
    l_test, l_train = labels[:test_size], labels[test_size:]

    return f_train, f_test, l_train, l_test


def standardize(f_train, f_test):
    # Standardize features by removing the mean and scaling to unit variance
    # x = (x - m) / v
    train_variance = np.var(f_train, dtype=np.float64)
    train_mean = np.mean(f_train)
    f_train = (f_train - train_mean) / train_variance

    test_variance = np.var(f_test, dtype=np.float64)
    test_mean = np.mean(f_test)
    f_test = (f_test - test_mean) / test_variance

    return f_train, f_test


def apply_knn(k_neighbor, f_train, l_train, f_test, l_test):
    classifier = KNN.KNN(k_neighbor)
    # fit the training data
    classifier.fit(f_train, l_train)
    # make prediction
    l_prediction = classifier.predict(f_test)
    accuracy = KNN.compute_accuracy(l_test, l_prediction)

    return classifier, l_prediction, accuracy


def apply_k_fold(k_neighbor, k_fold, features, labels):
    # k_fold
    folds = KNN.get_fold_indices(k_fold, features, labels)
    scores = []
    for [f_train, l_train, f_test, l_test] in folds:
        _, l_prediction, accuracy = apply_knn(k_neighbor, f_train, l_train, f_test, l_test)
        scores.append(accuracy)
    mean_accuracy = np.mean(scores)

    return scores, mean_accuracy


if __name__ == "__main__":
    # read data
    dataset = pd.read_csv('./archive/IRIS.csv')

    # 3 options:
    # X, Y = extract_data_random_state(dataset, random_state=42)
    X, Y = extract_data_random(dataset)
    # X, Y = extract_data_no_shuffle(dataset)

    # # just KNN
    # X_train, X_test, Y_train, Y_test = split_data(X, Y, test_ratio=0.25)
    # X_train, X_test = standardize(X_train, X_test)
    # knn, Y_prediction, k_accuracy = apply_knn(5, X_train, Y_train, X_test, Y_test)
    # print("Accuracy = ", k_accuracy, "%")
    #
    # # test
    # print(knn.predict([[6, 3, 5, 2]]))
    #
    # # parameter tuning with results
    # accuracies = knn.tune_parameter(X_test, Y_test, 100)
    # plt.plot(accuracies, color="skyblue")
    # plt.show()

    # k_fold
    fold_scores, accuracy_mean = apply_k_fold(10, 10, X, Y)
    print("K-Fold Cross-Validation Scores:", fold_scores)
    print("Mean Accuracy:", accuracy_mean)
