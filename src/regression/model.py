import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        # Inicialização de He melhorada
        self.W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size1))
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2. / hidden_size1)
        self.b2 = np.zeros((1, hidden_size2))
        self.W3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2. / hidden_size2)
        self.b3 = np.zeros((1, output_size))
        
        # Histórico para monitoramento
        self.loss_history = []
        self.val_loss_history = []
        self.best_weights = None
        self.best_loss = float('inf')

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def forward(self, X):
        # Camada oculta 1
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        
        # Camada oculta 2
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.relu(self.Z2)
        
        # Camada de saída (sem ativação para regressão)
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        return self.Z3

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        error = self.Z3 - y.reshape(-1, 1)
        
        # Gradientes da camada de saída
        dZ3 = error
        dW3 = np.dot(self.A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m
        
        # Gradientes da camada oculta 2
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * self.relu_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Gradientes da camada oculta 1
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Atualização dos pesos
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3

    def compute_loss(self, y_pred, y_real):
        return np.mean((y_pred - y_real.reshape(-1, 1)) ** 2)

    def train_model(self, X_train, y_train, X_val, y_val, num_epochs=2000, 
                   learning_rate=0.01, patience=100, min_delta=0.001):
        print(f"Iniciando treinamento com {len(X_train)} amostras...")
        
        for epoch in range(num_epochs):
            # Forward e backward pass
            y_pred = self.forward(X_train)
            loss = self.compute_loss(y_pred, y_train)
            self.backward(X_train, y_train, learning_rate)
            
            # Cálculo da perda de validação
            val_pred = self.forward(X_val)
            val_loss = self.compute_loss(val_pred, y_val)
            
            # Armazenar histórico
            self.loss_history.append(loss)
            self.val_loss_history.append(val_loss)
            
            # Verificar melhoria e early stopping
            if val_loss < self.best_loss - min_delta:
                self.best_loss = val_loss
                self.best_weights = [
                    self.W1.copy(), self.b1.copy(),
                    self.W2.copy(), self.b2.copy(),
                    self.W3.copy(), self.b3.copy()
                ]
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping na época {epoch} | Loss: {loss:.4f} | Val Loss: {val_loss:.4f}")
                    self.restore_best_weights()
                    break
            
            # Log periódico
            if epoch % 100 == 0:
                print(f"Época {epoch:4d}/{num_epochs} | Loss: {loss:.4f} | Val Loss: {val_loss:.4f}")

        print("Treinamento concluído!")
        self.plot_training_history()

    def restore_best_weights(self):
        if self.best_weights:
            self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = self.best_weights

    def predict(self, X):
        return self.forward(X)

    def plot_training_history(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.loss_history, label='Training Loss')
        plt.plot(self.val_loss_history, label='Validation Loss')
        plt.xlabel('Épocas')
        plt.ylabel('Loss (MSE)')
        plt.title('Evolução da Loss durante o Treinamento')
        plt.legend()
        plt.grid(True)
        plt.savefig("regression/graphics/training_history.png")
        plt.close()