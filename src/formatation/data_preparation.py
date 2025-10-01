from os import SEEK_END
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

from src.util.parameters import update_data_log
from src.util.parameters import LOG_DATA_PATH, log_file, TARGET
log_file.seek(0, SEEK_END)  # Move the cursor to the end of the file for appending new logs

def stratify(y: pd.Series, N) -> tuple[pd.Series, float]:
    """
    retorna um y_bin, para estratificação, que divide o y em N partes
    """
    bins, step = np.linspace(y.min(), y.max(), N + 1, retstep=True)
    labels = [f"{i}" for i in range(N)]
    y_bin = pd.cut(y, bins=bins, labels=labels, include_lowest=True)
    return y_bin, step


def split_data(X:pd.DataFrame, y:pd.Series, test_size:float,
               y_bin: pd.Series, n_folds: int) -> tuple[
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
    y_test_bin, _ = stratify(y_test, n_folds)
    y_train_bin, _ = stratify(y_train, n_folds)

    # Log dos tamanhos dos conjuntos
    update_data_log('Tamanho do Conjunto de Treino', len(y_train))
    update_data_log('Tamanho do Conjunto de Teste', len(y_test))
    log_file.write(f"\n----\nTamanho do conjunto de treino: {len(y_train)}\n")
    log_file.write(f"Tamanho do conjunto de teste: {len(y_test)}\n")

    # Para verificar o balanceamento das sentenças
    X_train[TARGET] = y_train.values
    X_train['bin'] = y_train_bin.values
    X_test[TARGET] = y_test.values
    X_test['bin'] = y_test_bin.values
    X_train.to_csv(f"{LOG_DATA_PATH}Train.csv", index=False)
    X_test.to_csv(f"{LOG_DATA_PATH}Test.csv", index=False)

    X_train = X_train.drop(columns=['sentenca', 'bin', TARGET])
    X_test = X_test.drop(columns=['sentenca', 'bin', TARGET])
    
    
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

    # Guardar os índices originais
    X_temp = X.copy()
    X_temp["__index__"] = X.index

    # Realizar oversampling com base nos bins
    log_file.write(f"\n----\nBalanceando os dados usando RandomOverSampler com a estratégia '{strategy}'\n")
    ros = RandomOverSampler(sampling_strategy=strategy, random_state=random_state)
    strat, step = stratify(y, n_folds)
    X_resampled, y_bins_resampled = [i for i in ros.fit_resample(X_temp, strat)]
    
    # Log dos dados balanceados
    update_data_log("Valor de Intervalo das Faixas", step)
    update_data_log("Bibliteca de Balanceamento", 'imblearn.over_sampling.RandomOverSampler')
    update_data_log("Metodo de Balanceamento", "fit_resample")
    update_data_log("Numero de Instancias Apos Balanceamento", len(y_bins_resampled))
    update_data_log("Valor Medio Apos Balanceamento", round(y.mean(), 2))
    update_data_log("Valor Minimo Apos Balanceamento", int(y.min()))
    update_data_log("Valor Maximo Apos Balanceamento", int(y.max()))

    log_file.write(f"\n----\nDiscretizando o alvo contínuo em {n_folds
                   } faixas com step = {step}\n")
    log_file.write(f"Número de instâncias Apos Balanceamento: {len(y_bins_resampled)}\n")
    log_file.write(f"Valor Medio Apos Balanceamento: {round(y.mean(), 2)}\n")
    log_file.write(f"Valor Minimo Apos Balanceamento: {y.min()}\n")
    log_file.write(f"Valor Maximo Apos Balanceamento: {y.max()}\n")
    

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
