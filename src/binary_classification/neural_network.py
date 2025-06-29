import numpy as np
from activations import sigmoid, sigmoid_deriv, relu, relu_deriv, tanh, tanh_deriv
from losses import bce_loss, bce_loss_deriv, mse_loss, mse_loss_deriv

activation_functions = {
    'sigmoid': (sigmoid, sigmoid_deriv),
    'relu': (relu, relu_deriv),
    'tanh': (tanh, tanh_deriv)
}

loss_functions = {
    'bce': (bce_loss, bce_loss_deriv),
    'mse': (mse_loss, mse_loss_deriv)
}

class NeuralNetwork:
    def __init__(self, layers, activation, loss, lr):
        self.layers = layers
        self.lr = lr
        self.act, self.act_deriv = activation_functions[activation]
        self.loss, self.loss_deriv = loss_functions[loss]
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            weight = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i])
            bias = np.zeros((1, layers[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, X):
        a = X
        self.zs, self.activations = [], [X]
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            if i == len(self.weights) - 1:
                a = sigmoid(z)
            else:
                a = self.act(z)
            self.zs.append(z)
            self.activations.append(a)
        return a

    def backward(self, y):
        m = len(y)
        delta = self.activations[-1] - y
        deltas = [delta]
        for i in range(len(self.layers)-3, -1, -1):
            delta = np.dot(deltas[0], self.weights[i+1].T) * self.act_deriv(self.activations[i+1])
            deltas.insert(0, delta)
        for i in range(len(self.weights)):
            dw = np.dot(self.activations[i].T, deltas[i]) / m
            db = np.sum(deltas[i], axis=0, keepdims=True) / m
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db


    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.loss(y, y_pred)
            print(f"Epoch {epoch+1}, Loss: {loss:.6f}")
            self.backward(y)

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)
