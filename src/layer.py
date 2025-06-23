import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation, activation_derivative):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.input = None
        self.output = None

    def forward(self, input_data):
        self.input = input_data
        z = np.dot(input_data, self.weights) + self.biases
        self.output = self.activation(z)
        return self.output

    def backward(self, output_error, learning_rate):
        activation_error = output_error * self.activation_derivative(self.output)
        input_error = np.dot(activation_error, self.weights.T)
        weights_error = np.dot(self.input.T, activation_error)

        # Atualização dos pesos e biases
        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * np.sum(activation_error, axis=0, keepdims=True)
        return input_error