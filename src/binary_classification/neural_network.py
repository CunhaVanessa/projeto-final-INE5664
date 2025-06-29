import numpy as np
from activations import sigmoid, sigmoid_deriv, relu, relu_deriv, tanh, tanh_deriv
from losses import mse_loss, mse_loss_deriv, bce_loss, bce_loss_deriv

class NeuralNetwork:
    def __init__(self, layers, activation='sigmoid', loss='bce', lr=0.01):
        self.layers = layers
        self.lr = lr
        self.weights = [np.random.randn(layers[i], layers[i+1]) * 0.1 for i in range(len(layers)-1)]
        self.biases = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]

        activations = {
            'sigmoid': (sigmoid, sigmoid_deriv),
            'relu': (relu, relu_deriv),
            'tanh': (tanh, tanh_deriv)
        }
        self.act, self.act_deriv = activations[activation]

        losses = {
            'mse': (mse_loss, mse_loss_deriv),
            'bce': (bce_loss, bce_loss_deriv)
        }
        self.loss_func, self.loss_deriv = losses[loss]

    def forward(self, X):
        a = X
        self.zs, self.activations = [], [X]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            a = self.act(z)
            self.zs.append(z)
            self.activations.append(a)
        return a

    def backward(self, y):
        m = len(y)
        delta = self.loss_deriv(y, self.activations[-1]) * self.act_deriv(self.zs[-1])
        deltas = [delta]

        for i in range(len(self.layers)-2, 0, -1):
            z = self.zs[i-1]
            delta = np.dot(deltas[0], self.weights[i].T) * self.act_deriv(z)
            deltas.insert(0, delta)

        for i in range(len(self.weights)):
            dw = np.dot(self.activations[i].T, deltas[i]) / m
            db = np.sum(deltas[i], axis=0, keepdims=True) / m
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db

    def train(self, X, y, epochs=100):
        for epoch in range(1, epochs+1):
            y_pred = self.forward(X)
            loss = self.loss_func(y, y_pred)
            self.backward(y)
            if epoch % 10 == 0 or epoch == 1:
                print(f"Época {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)
