# Projeto Final - Implementação de uma Rede Neural Artificial
A implementação foi desenvolvida como parte do Projeto Final da disciplina de aprendizado de máquina, no semestre 2025.1 da UFSC.

## Integrantes
- Vanessa Cunha (17100926)
- Gabriel Oliveira Reis (19205056)
- Pedro Ruschel Bressan (22100920) 

## Funcionalidades Implementadas
- Estrutura modular de rede com múltiplas camadas
- Algoritmo de retropropagação (backpropagation)
- Otimização via gradiente descendente
- Funções de ativação: Sigmoid, ReLU e Tanh
- Funções de perda: MSE e Cross-Entropy
- Suporte a tarefas de:
  - Regressão
  - Classificação Binária
  - Classificação Multiclasse

## Estrutura do Projeto

- data              -> Bases de dados
- notebooks         -> Notebooks para os três problemas
- src               -> Código fonte
    - activations.py
    - losses.py
    - layer.py
    - network.py
    -  utils.py
-  README.txt         -> Este arquivo
- requirements.txt   -> Dependências

## Instalação
1. Clone este repositório:
   git clone https://github.com/CunhaVanessa/projeto-final-INE5664.git
   cd repositorio

2. Instale as dependências:
   pip install -r requirements.txt

## Como Executar
1. Acesse os notebooks na pasta /notebooks.
2. Execute as células para carregar os dados, treinar os modelos e visualizar os resultados.

## Dados Utilizados
- Concrete Strength (Regressão)
- Breast Cancer Wisconsin (Classificação Binária)
- Iris (Classificação Multiclasse)

Tag de Avaliação
-----------------
Tag utilizada para avaliação: v1.0

Observações
-----------
Este repositório está privado até o término das apresentações. Acesso concedido para:
- GitHub: ProfCamiloUFSC

Checklist de Entrega
-----------

| Item                                  | Status |
| ------------------------------------- | ------ |
| Estrutura da rede (camadas, pesos)    | ⬜      |
| 3 funções de ativação                 | ⬜      |
| 2 funções de perda                    | ⬜      |
| Backpropagation                       | ⬜      |
| Gradiente descendente                 | ⬜      |
| Notebook de regressão                 | ⬜      |
| Notebook de classificação binária     | ⬜      |
| Notebook de classificação multiclasse | ⬜      |
| README organizado                     | ⬜      |
| Repositório no GitHub com tag         | ⬜      |
