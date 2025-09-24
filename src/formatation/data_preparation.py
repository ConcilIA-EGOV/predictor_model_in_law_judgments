import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

def stratify(y: pd.Series, N) -> pd.Series:
    """
    retorna um y_bin, para estratificação, que divide o y em N partes
    """
    bins = np.linspace(y.min(), y.max(), N + 1)
    labes = [f"{i}" for i in range(N)]
    y_bin = pd.cut(y, bins=bins, labels=labes, include_lowest=True)
    return y_bin


def split_data(X:pd.DataFrame, y:pd.Series, test_size:float,
               y_bin, n_folds) -> tuple[
                   pd.DataFrame, pd.DataFrame,
                   pd.Series, pd.Series, pd.Series]:
    """
    Dividir em conjuntos de treino e teste
    Parâmetros:
    -----------
    X : pd.DataFrame
        DataFrame com variáveis categóricas.
    y : pd.Series
        Série com alvo contínuo (ex: atraso em minutos).
    test_size : float
        Proporção do conjunto de teste.
    y_bin : pd.Series or None
        Série com o alvo discretizado para estratificação.
    STORE : bool
        Se True, armazena informações de debug e da coluna de sentenças.
    """
    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y,
                        test_size=test_size,
                        stratify=y_bin,
                        random_state=42)
    y_test_bin = stratify(y_test, n_folds)

    # Para verificar o balanceamento das sentenças
    sentences_test = X_test['sentenca'].copy()
    sentences_train = X_train['sentenca'].copy()
    
    sentences_test.to_csv('logs/sentences/test.csv', index=False)
    sentences_train.to_csv('logs/sentences/train.csv', index=False)
    
    X_test.to_csv('logs/data/X_test.csv', index=False)
    y_test.to_csv('logs/data/y_test.csv', index=False)
    X_train.to_csv('logs/data/X_train.csv', index=False)
    y_train.to_csv('logs/data/y_train.csv', index=False)
    y_test_bin.to_csv('logs/data/y_test_bin.csv', index=False)

    X_train = X_train.drop(columns=['sentenca'])
    X_test = X_test.drop(columns=['sentenca'])
    return X_train, X_test, y_train, y_test, y_test_bin


def balance_data(X: pd.DataFrame, y: pd.Series, strategy,
                 random_state, n_folds)->tuple[
                     pd.DataFrame, pd.Series, pd.Series]:
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
    STORE : bool
        Se True, armazena informações de debug e da coluna de sentenças.

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
    X_resampled, y_bins_resampled = [i for i in ros.fit_resample(X_temp, stratify(y, n_folds))]

    # 3. Criar cópia do DataFrame original com y contínuo
    df_original = X.copy()
    df_original['y_continuo'] = y

    # 4. Recuperar os índices dos dados originais usados
    idx_resampled = (X_resampled["__index__"].values)


    # 5. Recuperar os valores contínuos de y
    # Converter idx_resampled para uma lista/Index para satisfazer o tipo aceito por .loc
    idx_list = list(idx_resampled)
    y_resampled = y.loc[idx_list].reset_index(drop=True)

    # 6. Limpar coluna de índice temporário
    X_resampled = X_resampled.drop(columns=["__index__"]).reset_index(drop=True)
    X_resampled = pd.DataFrame(X_resampled)
    
    # 7. Adiciona a coluna de sentenca ao DataFrame balanceado
    sentences_bal = X_resampled['sentenca'].copy()
    sentences_bal.to_csv('logs/sentences/bal.csv', index=False)
    
    return X_resampled, y_resampled, pd.Series(y_bins_resampled)
