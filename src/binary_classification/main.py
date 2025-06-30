import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score
from neural_network import NeuralNetwork

valid_activations = ['sigmoid', 'tanh', 'relu']
valid_losses = ['bce', 'mse']

def main():
    if len(sys.argv) < 3:
        print("Uso: python main.py <activation> <loss>")
        print(f"Ativações válidas: {valid_activations}")
        print(f"Loss válidas: {valid_losses}")
        sys.exit(1)

    activation = sys.argv[1]
    loss = sys.argv[2]

    if activation not in valid_activations:
        print(f"Erro: ativação inválida '{activation}'. Use uma das {valid_activations}")
        sys.exit(1)

    if loss not in valid_losses:
        print(f"Erro: loss inválida '{loss}'. Use uma das {valid_losses}")
        sys.exit(1)

    df = pd.read_csv('spam.csv')
    X = df[['num_links', 'num_words', 'has_offer', 'sender_score', 'all_caps']].values
    y = df['is_spam'].astype(int).values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    nn = NeuralNetwork(layers=[5, 16, 8, 1], activation=activation, loss=loss, lr=0.5)
    nn.train(X_train, y_train, epochs=100)

    y_pred = nn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nMatriz de Confusão:")
    print(cm)
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
