import numpy as np

class WineQualityClassifier:
    """
    Classe que implementa uma Rede Neural Artificial (RNA) do zero para
    classificação multiclasse da qualidade de vinhos, usando apenas NumPy.

    Recursos:
    - Arquitetura customizada com número variável de camadas
    - Suporte a múltiplas funções de ativação por camada
    - Treinamento com Gradiente Descendente via Backpropagation
    - Camada de saída com Softmax para classificação multiclasse
    """

    def __init__(self, layers):
        """
        Inicializa a rede neural com camadas customizadas.

        :param layers: lista de dicionários, cada um representando uma camada.
                       Cada dicionário deve conter:
                       - 'neurons': número de neurônios
                       - 'activation': função de ativação ('relu', 'tanh', 'sigmoid')
        """
        self.layers = layers
        self.weights = []
        self.biases = []
        self.activations = [layer['activation'] for layer in layers[:-1]] + ['softmax']
        self._init_weights()

    def _init_weights(self):
        for i in range(len(self.layers) - 1):
            input_size = self.layers[i]['neurons']
            output_size = self.layers[i + 1]['neurons']
            W = np.random.randn(input_size, output_size) * 0.01
            b = np.zeros((1, output_size))
            self.weights.append(W)
            self.biases.append(b)

    def _activate(self, z, func):
        if func == 'relu':
            return np.maximum(0, z)
        elif func == 'tanh':
            return np.tanh(z)
        elif func == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif func == 'softmax':
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            raise ValueError(f"Função de ativação inválida: {func}")

    def _activate_deriv(self, a, func):
        if func == 'relu':
            return (a > 0).astype(float)
        elif func == 'tanh':
            return 1 - np.square(a)
        elif func == 'sigmoid':
            return a * (1 - a)
        else:
            raise ValueError(f"Derivada não implementada para função: {func}")

    def forward(self, X):
        self.Z = []
        self.A = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.A[-1], self.weights[i]) + self.biases[i]
            a = self._activate(z, self.activations[i])
            self.Z.append(z)
            self.A.append(a)
        return self.A[-1]

    def backward(self, y_true, learning_rate):
        m = y_true.shape[0]
        y_encoded = np.zeros_like(self.A[-1])
        y_encoded[np.arange(m), y_true] = 1

        dz = self.A[-1] - y_encoded

        for i in reversed(range(len(self.weights))):
            dw = np.dot(self.A[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m

            if i > 0:
                da = np.dot(dz, self.weights[i].T)
                dz = da * self._activate_deriv(self.A[i], self.activations[i - 1])

            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

    def train(self, X, y, epochs, learning_rate):
        loss_history = []
        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = self._loss(y, predictions)
            self.backward(y, learning_rate)
            if epoch % 100 == 0:
                print(f"Época {epoch} - Perda: {loss:.4f}")
            loss_history.append(loss)
        return loss_history

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def _loss(self, y_true, y_pred):
        m = y_true.shape[0]
        clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -np.mean(np.log(clipped[np.arange(m), y_true]))

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        acc = np.mean(y_pred == y_true)

        classes = np.unique(np.concatenate([y_true, y_pred]))
        num_classes = len(classes)
        metrics = {}
        cm = np.zeros((num_classes, num_classes), dtype=int)
        class_to_index = {cls: idx for idx, cls in enumerate(classes)}

        for t, p in zip(y_true, y_pred):
            cm[class_to_index[t], class_to_index[p]] += 1

        for cls in classes:
            idx = class_to_index[cls]
            tp = cm[idx, idx]
            fp = cm[:, idx].sum() - tp
            fn = cm[idx, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            support = cm[idx, :].sum()

            metrics[cls] = {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1-score': round(f1, 4),
                'support': int(support)
            }

        return {
            "accuracy": round(acc, 4),
            "metrics_per_class": metrics,
            "confusion_matrix": cm,
            "class_labels": classes.tolist()
        }
