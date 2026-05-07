import pandas as pd
import numpy as np
import math
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

from src.util.parameters import LOG_DATA_PATH, TARGET, FOLD_SIZE
from src.util.log_aux import update_data_log, log_file_preparation

def stratify(y: pd.Series, n_folds:int=0) -> tuple[pd.Series, float, int]:
    """
    retorna um y_bin, para estratificação, que divide o y em N partes
    """
    if n_folds == 0:
        n_folds = math.ceil((y.max() - y.min()) / FOLD_SIZE)
    bins, step = np.linspace(y.min(), y.max(), n_folds + 1, retstep=True)
    labels = [f"{i}" for i in range(n_folds)]
    y_bin = pd.cut(y, bins=bins, labels=labels, include_lowest=True)
    return y_bin, step, n_folds


def split_data(X:pd.DataFrame, y:pd.Series,
               test_size:float, random_state:int
               ) -> tuple[pd.DataFrame, pd.DataFrame,
                          pd.Series, pd.Series,
                          pd.Series, pd.Series]:
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
    y_bin, _, n_folds = stratify(y)
    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y,
                        test_size=test_size,
                        stratify=y_bin,
                        random_state=random_state)
    # Recalcula y_bin para os conjuntos de treino e teste
    y_test_bin, _, _ = stratify(y_test, n_folds)
    y_train_bin, _, _ = stratify(y_train, n_folds)

    # Para verificar o balanceamento das sentenças
    X_train[TARGET] = y_train.values
    X_train['bin'] = y_train_bin.values
    X_test[TARGET] = y_test.values
    X_test['bin'] = y_test_bin.values
    X_train.to_csv(f"{LOG_DATA_PATH}Train.csv", index=False)
    X_test.to_csv(f"{LOG_DATA_PATH}Test.csv", index=False)

    X_train = X_train.drop(columns=['sentenca', 'bin', TARGET])
    X_test = X_test.drop(columns=['sentenca', 'bin', TARGET])

    log_file_preparation.write(f"\n----\nTamanho original do conjunto de treino: {len(y_train)}\n")
    log_file_preparation.write(f"Tamanho do conjunto de teste: {len(y_test)}\n")


    return X_train, X_test, y_train, y_test, y_train_bin, y_test_bin


def balance_data(X: pd.DataFrame, y: pd.Series,
                 strategy: str | None, random_state: int
                 )-> tuple[pd.DataFrame, pd.Series, pd.Series]:
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

    # Guardar os índices originais
    X_temp = X.copy()
    X_temp["__index__"] = X.index

    # Determinar o número de faixas (bins) com base no intervalo do alvo
    update_data_log("Limite de tamanho para cada Faixa", FOLD_SIZE)
    strat, step, n_folds = stratify(y)
    update_data_log("Numero de Faixas de Valor", n_folds)
    # Realizar oversampling com base nos bins
    if strategy is not None:
        ros = RandomOverSampler(sampling_strategy=strategy, random_state=random_state)
        X_resampled, y_bins_resampled = [i for i in ros.fit_resample(X_temp, strat)]
    else:
        X_resampled, y_bins_resampled = X_temp, strat

    # Log dos dados balanceados
    log_file_preparation.write(f"\n----\nBalanceando os dados usando RandomOverSampler com a estratégia '{strategy}'\n")
    update_data_log("Tamanho de cada Faixa", round(step, 2))
    update_data_log("Bibliteca de Balanceamento", 'imblearn.over_sampling.RandomOverSampler')
    update_data_log("Metodo de Balanceamento", "fit_resample")
    update_data_log("Numero de Instancias Pre-Balanceamento", len(y))
    update_data_log("Numero de Instancias Apos Balanceamento", len(y_bins_resampled))
    update_data_log("Numero de Instancias Adicionadas pelo Balanceamento", len(y_bins_resampled) - len(y))
    update_data_log("Valor Medio Apos Balanceamento", round(y.mean(), 2))
    update_data_log("Valor Minimo Apos Balanceamento", int(y.min()))
    update_data_log("Valor Maximo Apos Balanceamento", int(y.max()))

    log_file_preparation.write(f"\n----\nDiscretizando o alvo contínuo em {n_folds
                   } faixas com step = {step}\n")
    log_file_preparation.write(f"Numero de instancias Apos Balanceamento: {len(y_bins_resampled)}\n")
    log_file_preparation.write(f"Valor Medio Apos Balanceamento: {round(y.mean(), 2)}\n")
    log_file_preparation.write(f"Valor Minimo Apos Balanceamento: {y.min()}\n")
    log_file_preparation.write(f"Valor Maximo Apos Balanceamento: {y.max()}\n")


    # Recuperar os índices dos dados originais usados
    idx_resampled = list(X_resampled["__index__"].values)

    # Converter idx_resampled para uma lista/Index para satisfazer o tipo aceito por .loc
    y_resampled = y.loc[idx_resampled].reset_index(drop=True)

    # Limpar coluna de índice temporário
    X_resampled = X_resampled.drop(columns=["__index__"]).reset_index(drop=True)
    X_resampled = pd.DataFrame(X_resampled)

    # Salvar dados balanceados para debug
    X_resampled[TARGET] = y_resampled
    X_resampled['Faixa'] = y_bins_resampled
    X_resampled.to_csv(f'{LOG_DATA_PATH}Balanced-Data.csv', index=False)
    X_resampled = X_resampled.drop(columns=[TARGET, "Faixa"])

    return X_resampled, y_resampled, pd.Series(y_bins_resampled)
