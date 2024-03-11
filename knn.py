import numpy as np
from scipy.stats import mode


class KNN():
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def predict_proba(self, X):
        probabilities = [self._predict_proba(x) for x in X]
        return np.array(probabilities)

    def _predict(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = mode(k_nearest_labels)
        return most_common.mode if np.isscalar(most_common.mode) else most_common.mode[0]

    def _predict_proba(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        proba = np.mean(np.array(k_nearest_labels) == 1), np.mean(np.array(k_nearest_labels) == 0)
        return proba

    @staticmethod
    def _euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_params(self, deep=True):
        return {'k': self.k}

    def set_params(self, **parameters):
        self.k = parameters.get('k', self.k)
        return self
