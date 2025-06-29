import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder

def exploracao_dados(X, y, feature_names, dataset_name="Dataset"):

    # Convertendo para DataFrame para fácil visualização
    df = pd.DataFrame(X, columns=feature_names)
    df["Target"] = y

    # Gráfico de distribuição dos dados de entrada (features)
    plt.figure(figsize=(16, 10))
    df.drop("Target", axis=1).hist(bins=20, grid=False, layout=(4, 4))
    plt.suptitle(f"Distribuição das Variáveis - {dataset_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("regression/histograma_distribuicao.png")
    plt.close()

    # Gráfico de dispersão entre a variável target e as variáveis
    num_features = X.shape[1]
    rows = (num_features + 2) // 3  # Calcular o número de linhas necessárias

    plt.figure(figsize=(18, 5 * rows))
    for i in range(num_features):
        plt.subplot(rows, 3, i + 1)
        plt.scatter(df.iloc[:, i], df["Target"], alpha=0.5)
        plt.xlabel(f"{feature_names[i]}")
        plt.ylabel("Exam Score")
        plt.title(f"Relação entre {feature_names[i]} e Desempenho")

    plt.tight_layout()
    plt.savefig("regression/dispersao_features_target.png")
    plt.close()

def load_data(train_size=0.7, val_size=0.15):
    # Carregando o dataset de estudantes
    df = pd.read_csv('regression/cleaned_student_performance.csv')
    
    # Pré-processamento dos dados
    if 'student_id' in df.columns:
        df = df.drop('student_id', axis=1)
    
    # Converter variáveis categóricas para numéricas
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    # Separar features e target
    X = df.drop('exam_score', axis=1).values
    y = df['exam_score'].values
    feature_names = df.drop('exam_score', axis=1).columns.tolist()

    # Criando índices aleatórios para separação
    n = len(X)
    indices = np.random.permutation(n)
    
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    
    # Separando os dados
    X_train = X[indices[:train_end]]
    X_val = X[indices[train_end:val_end]]
    X_test = X[indices[val_end:]]
    
    y_train = y[indices[:train_end]]
    y_val = y[indices[train_end:val_end]]
    y_test = y[indices[val_end:]]

    # Normalizando os dados manualmente
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    X_train_std[X_train_std == 0] = 1
    
    X_train = (X_train - X_train_mean) / X_train_std
    X_val = (X_val - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std

    # Normalizando os valores de y (target)
    y_train_mean = y_train.mean()
    y_train_std = y_train.std()
    y_train = (y_train - y_train_mean) / y_train_std
    y_val = (y_val - y_train_mean) / y_train_std
    y_test = (y_test - y_train_mean) / y_train_std

    return X_train, X_val, X_test, y_train, y_val, y_test, y_train_mean, y_train_std, feature_names