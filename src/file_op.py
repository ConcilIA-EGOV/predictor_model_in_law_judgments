###
import sys
import os
# Obtém o diretório atual do script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Adiciona o diretório base do projeto ao caminho de busca do Python
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
###
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from src.training import split_train_test
from util.parameters import DM_FOLDS, TARGET, TEST_SIZE


def stratify(y, N=DM_FOLDS) -> pd.Series:
    """
    retorna um y_bin, para estratificação, que divide o y em N partes
    """
    bins = np.linspace(y.min(), y.max(), N + 1)
    labes = [f"{i}" for i in range(N)]
    y_bin = pd.cut(y, bins=bins, labels=labes, include_lowest=True)
    return y_bin


def balance_data(X, y, y_bins, strategy='not majority',
                 random_state=15) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Realiza oversampling em um problema de regressão com variáveis categóricas,
    usando discretização do alvo contínuo em faixas (bins).

    Parâmetros:
    -----------
    X : pd.DataFrame
        DataFrame com variáveis categóricas.
    y : pd.Series
        Série com alvo contínuo (ex: atraso em minutos).
    bins : list or None
        Lista de limites de faixas.
    labels : list or None
        Lista de rótulos dos bins.
    strategy : str
        Estratégia de oversampling (ex: 'not majority', 'auto', 'minority').
    random_state : int
        Semente aleatória para reprodutibilidade.

    Retorna:
    --------
    X_resampled : pd.DataFrame
        Conjunto de entrada balanceado.
    y_resampled : pd.Series
        Alvo contínuo balanceado.
    y_bins_resampled : pd.Series
        Alvo discretizado balanceado.
    """

    # 1. Guardar os índices originais
    X_temp = X.copy()
    X_temp["__index__"] = X.index

    # 2. Realizar oversampling com base nos bins
    ros = RandomOverSampler(sampling_strategy=strategy, random_state=random_state)
    X_resampled, y_bins_resampled = [i for i in ros.fit_resample(X_temp, y_bins)]

    # 3. Criar cópia do DataFrame original com y contínuo
    df_original = X.copy()
    df_original['y_continuo'] = y

    # 4. Recuperar os índices dos dados originais usados
    idx_resampled = X_resampled["__index__"].values


    # 5. Recuperar os valores contínuos de y
    y_resampled = y.loc[idx_resampled].reset_index(drop=True)

    # 6. Limpar coluna de índice temporário
    X_resampled = X_resampled.drop(columns=["__index__"]).reset_index(drop=True)
    X_resampled = pd.DataFrame(X_resampled)
    
    return X_resampled, y_resampled, y_bins_resampled # type: ignore


def load_data(split=True) -> list[pd.DataFrame]:
    """
    Carrega os dados do arquivo CSV, divide em treino e teste, e aplica balanceamento.

    Retorna (se split for True):
    - X_train: DataFrame de treino
    - X_test: DataFrame de teste
    - y_train: Série de treino
    - y_test: Série de teste
    - y_test_bin: Série de folds de y_test para estratificação
    - sentences_train: Série de sentenças de treino
    - sentences_test: Série de sentenças de teste

    Se split for False, retorna:
    - X_bal: DataFrame balanceado
    - y_bal: Série balanceada
    - y_bin: Série de bins para estratificação
    - sentences_bal: Série de sentenças balanceadas
    """
    # Load the dataset
    data = pd.read_csv('data/main.csv')
    
    # Split the data into features and target variable
    y = data[TARGET]
    X = data.drop([TARGET], axis=1)
    y_bin = stratify(y, N=DM_FOLDS)
    X_bal, y_bal, y_bin = balance_data(X, y, y_bin, strategy='not majority', random_state=15)
    # Adiciona a coluna de sentenca ao DataFrame balanceado
    sentences_bal = X_bal['sentenca'].copy()
    # sentences_bal.to_csv('data/sentences_bal.csv', index=False)

    # Treinar/testar
    if split:
        X_train, X_test, y_train, y_test = split_train_test(X_bal, y_bal, test_size=TEST_SIZE, y_bin=y_bin) # type: ignore
        y_test_bin = stratify(y_test, N=DM_FOLDS)
        # Adiciona a coluna de sentenca ao DataFrame de teste
        sentences_train = X_train['sentenca'].copy()
        sentences_test = X_test['sentenca'].copy()

        # sentences_train.to_csv('data/sentences_train.csv', index=False)
        # sentences_test.to_csv('data/sentences_test.csv', index=False)
        X_train = X_train.drop(columns=['sentenca'])
        X_test = X_test.drop(columns=['sentenca'])
        output = [pd.DataFrame(i) for i in (X_train, X_test, y_train, y_test, y_test_bin, sentences_train, sentences_test)]
        return output

    X_bal = X_bal.drop(columns=['sentenca'])
    output = [pd.DataFrame(i) for i in [X_bal, y_bal, y_bin, sentences_bal]]
    return output

if __name__ == '__main__':
    # Load the dataset without splitting
    X, y, bin, sentences = load_data(split=False)
    print("Data loaded without splitting.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y_bin shape: {bin.shape}")
    print(f"sentences shape: {sentences.shape}")
    
    data = pd.concat([sentences, X, y], axis=1)
    data['fold'] = bin
    data.to_csv('data/non_split_data.csv', index=False)

    # Load the dataset and split into train and test sets
    X_train, X_test, y_train, y_test, y_test_bin, sent_train, sent_test = load_data(split=True)
    print("Data loaded and split successfully.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"y_test_bin shape: {y_test_bin.shape}")
    print(f"sent_train shape: {sent_train.shape}")
    print(f"sent_test shape: {sent_test.shape}")

    # Save the split data to CSV files
    data_train = pd.concat([sent_train, X_train, y_train], axis=1)
    data_test = pd.concat([sent_test, X_test, y_test], axis=1)
    data_train['fold'] = y_test_bin
    data_train.to_csv('data/train_data.csv', index=False)
    data_test.to_csv('data/test_data.csv', index=False)
    print("Split Data saved to CSV files.")

    print("All operations completed successfully.")

