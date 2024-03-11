import numpy as np
from scipy.special import expit as sigmoid_function

class Perceptron():
    def __init__(self, learning_rate=0.01, n_iterations=1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None
        self.classes_ = None
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = 1 if linear_output >= 0 else 0
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
        
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)
    
    def predict_proba(self, X):
        decision_function = np.dot(X, self.weights) + self.bias
        proba = sigmoid_function(decision_function)
        return np.vstack((1 - proba, proba)).T
    
    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)
    
    def get_params(self, deep=True):
        return {
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations
        }
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / len(y)