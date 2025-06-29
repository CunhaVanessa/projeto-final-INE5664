import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score
from neural_network import NeuralNetwork

# Carregar dados
df = pd.read_csv('spam.csv')
X = df[['num_links', 'num_words', 'has_offer', 'sender_score', 'all_caps']].values
y = df['is_spam'].astype(int).values.reshape(-1, 1)

# Normalizar
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Separar treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Instanciar e treinar rede
nn = NeuralNetwork(layers=[5, 8, 4, 1], activation='sigmoid', loss='bce', lr=0.05)
nn.train(X_train, y_train, epochs=100)

# Avaliação
y_pred = nn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nMatriz de Confusão:")
print(cm)
print(f"F1 Score: {f1:.4f}")
