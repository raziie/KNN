import numpy as np


def compute_euclidean(x1, x2):
    return np.sqrt(np.sum(np.power((x1-x2), 2)))


class KNN:

    def __init__(self, k):
        self.k = k
        # self.X_test = X_test
        # self.Y_test = Y_test

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    # def find_neighbors(self):

    def predict(self, X_test):
        pass
