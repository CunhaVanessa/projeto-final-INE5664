import numpy as np

class LossFunction:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    @staticmethod
    def mse_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

    @staticmethod
    def cross_entropy(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def cross_entropy_derivative(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred)) / y_true.shape[0]