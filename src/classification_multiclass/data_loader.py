# src/classification_multiclass/data_loader.py

import os
import pandas as pd
import numpy as np
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class WineQualityDataLoader:
    """
    Classe responsável por baixar, carregar e preparar o dataset de qualidade do vinho
    (red + white) para uma tarefa de classificação multiclasse.

    Dataset original: https://archive.ics.uci.edu/ml/datasets/Wine+Quality
    """

    def __init__(self):
        # Define diretório fixo onde os dados serão armazenados
        self.data_dir = "/Users/vanessacunha/Documents/Estudo/UFSC/2025.1/Aprendizado de máquina/projeto-final-INE5664/data"
        os.makedirs(self.data_dir, exist_ok=True)

        # URLs de onde os dados serão baixados
        self.red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        self.white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

        # Caminhos onde os arquivos serão salvos
        self.red_path = os.path.join(self.data_dir, "winequality-red.csv")
        self.white_path = os.path.join(self.data_dir, "winequality-white.csv")

    def download_if_needed(self):
        """
        Faz download dos arquivos se ainda não existirem no diretório local.
        """
        if not os.path.exists(self.red_path):
            print("📥 Baixando dataset do vinho tinto...")
            urlretrieve(self.red_url, self.red_path)
            print("✅ Vinho tinto salvo.")

        if not os.path.exists(self.white_path):
            print("📥 Baixando dataset do vinho branco...")
            urlretrieve(self.white_url, self.white_path)
            print("✅ Vinho branco salvo.")

    def load_and_prepare_data(self):
        """
        Carrega os arquivos CSV, une os dados, normaliza os atributos e aplica
        transformação para que os rótulos fiquem de 0 a 6 (7 classes).

        :return: X_train, X_test, y_train, y_test, y_all
        """
        # Garante que os arquivos estejam disponíveis
        self.download_if_needed()

        # Lê os arquivos CSV separados por ponto e vírgula
        red_df = pd.read_csv(self.red_path, sep=";")
        white_df = pd.read_csv(self.white_path, sep=";")

        # Adiciona coluna "type" para diferenciar (opcional, descartada depois)
        red_df['type'] = 0
        white_df['type'] = 1

        # Une os dois datasets
        df = pd.concat([red_df, white_df], ignore_index=True)

        # Normaliza os rótulos para que comecem de zero (de 3 a 9 → 0 a 6)
        df['quality'] = df['quality'] - df['quality'].min()

        # Divide em atributos (X) e rótulos (y)
        X = df.drop(columns=['quality', 'type']).values
        y = df['quality'].values

        # Normaliza os atributos
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Separa em treino e teste estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        print("✅ Dados carregados e preparados com sucesso.")
        print(f"🔢 Total de amostras: {len(y)} | Treino: {len(y_train)} | Teste: {len(y_test)}")

        return X_train, X_test, y_train, y_test, y