import numpy as np

class WineQualityClassifier:
    """
    Classe que implementa uma Rede Neural Artificial (RNA) do zero para
    classificação multiclasse da qualidade de vinhos, usando apenas NumPy.

    Recursos:
    - Arquitetura customizada com número variável de camadas
    - Suporte a múltiplas funções de ativação (ReLU, Tanh, Sigmoid)
    - Treinamento com Gradiente Descendente via Backpropagation
    - Camada de saída com Softmax para classificação multiclasse
    """

    def __init__(self, architecture, activation='relu'):
        """
        Inicializa a rede neural com a arquitetura e função de ativação desejadas.

        :param architecture: lista com o número de neurônios por camada
        :param activation: função de ativação ('relu', 'tanh', 'sigmoid')
        """
        self.architecture = architecture
        self.activation = activation
        self.weights = []  # Lista com os pesos de cada camada
        self.biases = []   # Lista com os vieses de cada camada
        self._init_weights()

    def _init_weights(self):
        """
        Inicializa os pesos com valores pequenos aleatórios e vieses com zeros.
        """
        for i in range(len(self.architecture) - 1):
            W = np.random.randn(self.architecture[i], self.architecture[i + 1]) * 0.01
            b = np.zeros((1, self.architecture[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

    def _activate(self, z):
        """
        Aplica a função de ativação na camada oculta.
        """
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        else:
            raise ValueError("Função de ativação inválida.")

    def _activate_deriv(self, a):
        """
        Calcula a derivada da função de ativação escolhida.
        """
        if self.activation == 'relu':
            return (a > 0).astype(float)
        elif self.activation == 'tanh':
            return 1 - np.square(a)
        elif self.activation == 'sigmoid':
            return a * (1 - a)

    def _softmax(self, z):
        """
        Função softmax para saída multiclasse.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """
        Realiza a propagação para frente e armazena os valores intermediários.
        """
        self.Z = []  # Valores antes da ativação
        self.A = [X] # Saídas após ativação
        for i in range(len(self.weights)):
            z = np.dot(self.A[-1], self.weights[i]) + self.biases[i]
            a = self._softmax(z) if i == len(self.weights) - 1 else self._activate(z)
            self.Z.append(z)
            self.A.append(a)
        return self.A[-1]

    def backward(self, y_true, learning_rate):
        """
        Executa retropropagação e atualiza pesos e vieses.
        """
        m = y_true.shape[0]
        y_encoded = np.zeros_like(self.A[-1])
        y_encoded[np.arange(m), y_true] = 1  # One-hot encoding

        dz = self.A[-1] - y_encoded  # Gradiente da saída

        for i in reversed(range(len(self.weights))):
            dw = np.dot(self.A[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m

            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * self._activate_deriv(self.A[i])

            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

    def train(self, X, y, epochs, learning_rate):
        """
        Treina a rede usando gradiente descendente.
        """
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
        """
        Prediz a classe para os dados de entrada.
        """
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def _loss(self, y_true, y_pred):
        """
        Calcula a perda cross-entropy.
        """
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



