import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import os

def evaluate_model(X_test, y_test, model, y_train_mean, y_train_std):
    """
    Avalia o modelo de regressão com as métricas MAE, MSE, RMSE e R².
    Gera dois gráficos: Previsões vs Reais e Histograma de Erros.
    """
    print("\nAvaliando o modelo...")
    
    # 1. Fazer previsões
    y_pred = model.predict(X_test)
    
    # 2. Desnormalizar valores
    y_pred_denorm = y_pred.flatten() * y_train_std + y_train_mean
    y_test_denorm = y_test * y_train_std + y_train_mean
    
    # 3. Calcular métricas
    errors = y_test_denorm - y_pred_denorm
    abs_errors = np.abs(errors)
    
    mae = np.mean(abs_errors)
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum(errors**2) / np.sum((y_test_denorm - np.mean(y_test_denorm))**2))
    
    # 4. Exibir métricas em tabela
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    }
    
    print("\nMétricas de Desempenho:")
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Métrica', 'Valor'])
    print(tabulate(metrics_df, headers='keys', tablefmt='pretty', showindex=False))
    
    # 5. Gerar visualizações
    
    # a. Gráfico de dispersão: Valores Reais vs Preditos
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_denorm, y_pred_denorm, alpha=0.5)
    plt.plot([y_test_denorm.min(), y_test_denorm.max()], 
             [y_test_denorm.min(), y_test_denorm.max()], 
             'r--', lw=2)
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Preditos')
    plt.title('Valores Reais vs. Valores Preditos')
    plt.grid(True)
    plt.savefig('regression/graphics/real_vs_pred.png', bbox_inches='tight')
    plt.close()
    
    # b. Histograma de erros
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Erro de Predição')
    plt.ylabel('Frequência')
    plt.title('Distribuição dos Erros de Predição')
    plt.grid(True)
    plt.savefig('regression/graphics/error_distribution.png', bbox_inches='tight')
    plt.close()
    
    print("\nAvaliação concluída! Resultados salvos em 'regression/evaluation/'")
    
    return metrics