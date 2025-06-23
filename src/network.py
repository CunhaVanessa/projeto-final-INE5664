class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, loss_grad, learning_rate):
        output_error = loss_grad
        for layer in reversed(self.layers):
            output_error = layer.backward(output_error, learning_rate)

    def train(self, X, y, epochs, learning_rate, loss_fn, loss_derivative):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = loss_fn(y, output)
            loss_grad = loss_derivative(y, output)

            self.backward(loss_grad, learning_rate)

            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch+1}/{epochs} - Loss: {loss:.6f}')

    def predict(self, X):
        output = self.forward(X)
        return output