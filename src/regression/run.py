from time import sleep
import numpy as np
from .data import load_data
from .model import NeuralNetwork
from .utils import evaluate_model

def run():
    print('--------------------------- Regressão - Desempenho Estudantil [INICIO] ---------------------------\n\n\n')
    
    # 1. Carregar e preparar os dados
    print("Carregando e preparando os dados...")
    X_train, X_val, X_test, y_train, y_val, y_test, y_train_mean, y_train_std, feature_names = load_data(
        train_size=0.7,
        val_size=0.15
    )
    print(f"Dados carregados e normalizados:")
    print(f"  - Treino: {len(X_train)} amostras")
    print(f"  - Validação: {len(X_val)} amostras")
    print(f"  - Teste: {len(X_test)} amostras")
    
    # 2. Inicializar a rede neural
    print("\nInicializando a rede neural...")
    input_size = X_train.shape[1]  # Número de características
    hidden_size1 = 128  # Primeira camada oculta
    hidden_size2 = 64   # Segunda camada oculta
    output_size = 1     # Saída única para regressão
    
    model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)
    
    # 3. Treinar o modelo
    print("\nIniciando o treinamento...")
    num_epochs = 3000
    learning_rate = 0.02
    training_patient = 150  # Paciência para early stopping
    
    model.train_model(
        X_train, y_train,
        X_val, y_val,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        patience=training_patient,
        min_delta=0.0005
    )
    print("Treinamento finalizado.")
    
    # 4. Avaliar o modelo no conjunto de teste
    print("\nAvaliando o modelo no conjunto de teste...")
    sleep(2)
    evaluate_model(X_test, y_test, model, y_train_mean, y_train_std)
    
    print('\n--------------------------- Regressão - Desempenho Estudantil [FIM] ---------------------------\n\n\n')

if __name__ == "__main__":
    run()