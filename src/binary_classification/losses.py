import numpy as np

def mse_loss(y_true, y_pred): return np.mean((y_true - y_pred)**2)
def mse_loss_deriv(y_true, y_pred): return (y_pred - y_true)

def bce_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bce_loss_deriv(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * len(y_true))
