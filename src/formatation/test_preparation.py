# Adiciona o diretório base do projeto ao caminho de busca do Python
import sys, os
this_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not this_path in sys.path:
    sys.path.append(this_path)

import pandas as pd
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold

from util.parameters import TARGET, FOLD_SIZE, BIN_COL
from util.log_aux import update_data_log, log_file_preparation

def stratify(y: pd.Series, n_folds:int=0) -> tuple[pd.Series, float, int]:
    """
    retorna um y_bin, para estratificação, que divide y em classes iguais pelo seu valor
    bem como o intervalo das classes (step) e o número de classes n_folds
    """
    if n_folds == 0:
        n_folds = math.ceil((y.max() - y.min()) / FOLD_SIZE)
    bins, step = np.linspace(y.min(), y.max(), n_folds + 1, retstep=True)
    labels = [f"{i}" for i in range(n_folds)]
    y_bin = pd.cut(y, bins=bins, labels=labels, include_lowest=True)
    return y_bin, step, n_folds


def split_data(X:pd.DataFrame, n_splits:int, random_state:int
               ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    cria múltiplas divisões de treino/teste (0.8/0.2)
    Parâmetros:
    -----------
    X : pd.DataFrame
        DataFrame com com features, target e coluna estratificada do target.
    n_splits : int
        Número de cross-validation splits, cada um contendo sua própria divisão distinta de treino/teste
    random_state : int
        Seed de aleatorieadade para permitir a reprodução de experimentos
    """
    # Determinar o número de faixas (bins) com base no intervalo do alvo
    update_data_log("Limite de tamanho para cada Faixa", FOLD_SIZE)
    y_bin, step, nf = stratify(X[TARGET])
    update_data_log("Tamanho de cada Faixa", round(step, 2))
    update_data_log("Numero de Faixas de Valor", nf)
    log_file_preparation.write(f"\n----\nDiscretizando o alvo contínuo em {nf} faixas com step = {step}\n")
    X[BIN_COL] = y_bin.values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    split_idxs = list(skf.split(X, y_bin))
    output = []
    for fold, (train_idx, test_idx) in enumerate(split_idxs):
        train = X.iloc[train_idx]
        test = X.iloc[test_idx]

        log_file_preparation.write(f"\n----\nTamanho original do conjunto de treino {fold}: {len(train)}\n")
        log_file_preparation.write(f"Tamanho do conjunto de teste {fold}: {len(test)}\n")
        output.append((train, test))

    return output


def balance_data(data: pd.DataFrame, strategy: str | None,
                 random_state: int)-> pd.DataFrame:
    """
    Realiza oversampling em um problema de regressão com variáveis categóricas,
    usando discretização do alvo contínuo em faixas (bins).

    Parâmetros:
    -----------
    data : pd.DataFrame
        DataFrame com features e target.
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
    resampled : pd.DataFrame
        Conjunto balanceado.
    """
    if BIN_COL not in data.columns:
        update_data_log("Limite de tamanho para cada Faixa", FOLD_SIZE)
        y_bin, step, nf = stratify(data[TARGET])
        update_data_log("Tamanho de cada Faixa", round(step, 2))
        update_data_log("Numero de Faixas de Valor", nf)
        log_file_preparation.write(f"\n----\nDiscretizando o alvo contínuo em {nf} faixas com step = {step}\n")
        data[BIN_COL] = y_bin.values

    if strategy is not None:
        ros = RandomOverSampler(sampling_strategy=strategy, random_state=random_state)
        log_file_preparation.write(f"\n----\nBalanceando os dados usando RandomOverSampler com a estratégia '{strategy}'\n")
        update_data_log("Bibliteca de Balanceamento", 'imblearn.over_sampling.RandomOverSampler')
        # Realizar oversampling com base nos bins
        resampled, _ = [i for i in ros.fit_resample(data, data[BIN_COL])]
        update_data_log("Metodo de Balanceamento", "fit_resample")
        resampled = pd.DataFrame(resampled)
    else:
        resampled = data

    # Log dos dados balanceados
    update_data_log("Numero de Instancias Pre-Balanceamento", len(data))
    update_data_log("Valor Medio Pre-Balanceamento", round(data[TARGET].mean(), 2))
    update_data_log("Numero de Instancias Apos Balanceamento", len(resampled))
    update_data_log("Numero de Instancias Adicionadas pelo Balanceamento", len(resampled) - len(data))
    update_data_log("Valor Medio Pos-Balanceamento", round(resampled[TARGET].mean(), 2))

    return resampled
