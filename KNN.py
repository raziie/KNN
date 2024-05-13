import numpy as np


def compute_euclidean(x1, x2):
    return np.sqrt(np.sum(np.power((x1-x2), 2)))


class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def find_neighbors(self, point):
        # compute distances with all points in X_train
        distances = np.array([])
        for x in self.X_train:
            distance = compute_euclidean(x, point)
            distances = np.append(distances, distance)
        # sort by distances index
        sorted_dists = distances.argsort()
        # return the k nearest ones
        return sorted_dists[:self.k]

    def predict(self, X_test):
        Y_out = np.array([])
        # for all test values find the most common value between k nearest neighbors
        # and add it to the result
        for x in X_test:
            neighbors = self.find_neighbors(x)
            unique, counts = np.unique(self.Y_train[neighbors], return_counts=True)
            Y_out = np.append(Y_out, unique[counts.argsort()[-1]])
        return Y_out
