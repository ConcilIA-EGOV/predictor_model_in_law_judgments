import os
import pandas as pd

from util.log_aux import append_to_data_log_list, log_file_preprocessing
from util.parameters import TARGET, ID_COL, BIN_COL

def feature_selection(
        df: pd.DataFrame,
        log_data_path:str,
        rnd_state:int,
        var_threshold:float=0.05,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parâmetros:
        - df: dataset que será alterado
        - log_data_path: o caminho para registrar as alterações
        - rnd_state: random seed for reproductibility
        - var_threshold: Variância mínima tolerada para menter um feature.
        - features_to_ignore: nomes das features/colunas que serão desconsideradas (isto é, as colunas em si serão removidas, perdendo-se a informação de quais entradas/linhas possuiam valores não 0 nessas features)
        - features_to_eliminate: nomes das features/coluns que, além de serem removidas, qualquer caso/linha onde possuam valor diferente de 0 será excluído,
    Remove colunas não relacionadas ao experimento e retorna os casos removidos
    """
    # Eliminating features anf any entry that has a non-zero value for them
    features_to_keep = feature_exploration(df, log_data_path, var_threshold, rnd_state, verbose=False)
    remove_columns = [col for col in df.columns if col not in features_to_keep]
    df, con = feature_elimination(df, log_data_path, remove_columns)

    append_to_data_log_list('Features Eliminadas', remove_columns)
    return df, con

def feature_elimination(df: pd.DataFrame, log_data_path:str, remove_cols:list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove confatores do DataFrame e as retorna separadamente
    1. Identifica as linhas/entradas onde o valor dos features a serem removidos é diferente de 0
    2. Salva o dataset parcial só com as entradas removidas para cada feature em um csv em `log_data_path/_Discarded_Features/`
    3. Remove todos as entradas com esses features != 0 na variável `pro`
    4. Concatena todos os dados removidos na variável `con`
    5. retorna `pro`e `con`
    """

    remove_cols = [c for c in remove_cols if c in df.columns]
    pro = df
    con = pd.DataFrame(columns=df.columns)
    conf_dir = f"{log_data_path}_Discarded_Features/"
    st_size = pro.shape[0]
    os.makedirs(conf_dir, exist_ok=True)
    for col in remove_cols:
        tmp = df[df[col] != 0]
        tmp.to_csv(f"{conf_dir}{col}.csv")
        con = pd.concat([con, tmp])
        pro = pro[pro[col] == 0]

    end_size = pro.shape[0] - st_size
    log_file_preprocessing.write(f"Removendo colunas: {remove_cols}.\n   --> Resultando em {pro.shape} instancias sem confactors e {end_size} instancias com confactors.\n")
    append_to_data_log_list('Alteracoes nas Features', f"Removidas {end_size} instancias que continham confactors: {remove_cols}")
    pro = pro.drop(columns=remove_cols)

    return pro, con


def print_scores(scores_df: pd.DataFrame, col: str, reversed) -> None:
    scores = scores_df[col].tolist()
    cols = scores_df["Feature"].tolist()
    print(f"\n{col} Scores:")
    # formats a string to have 30 chars
    match = [(f"{cols[i]:30}:", v) for i, v in enumerate(scores)]
    match = sorted(match, key=lambda x: float(x[1]), reverse=reversed)
    match = [f"{m[0]} {m[1]}"  for m in match]
    log = " - " + "\n - ".join(match)
    print(log)
    # log_file.write(f"{log}\n")


def append_scores_log(scores_df: pd.DataFrame, scores, cols: list[str], col: str) -> None:
    cols.append(col)
    scores_df[col] = [f"{abs(i):.15f}" if i else "0.0" for i in scores] # type: ignore


import warnings
from scipy.stats import ConstantInputWarning
warnings.filterwarnings("ignore", category=ConstantInputWarning)
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import pointbiserialr
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_regression

from test_preparation import stratify

def feature_exploration(
        X_df: pd.DataFrame,
        log_data_path:str,
        var_threshold:float,
        rnd_state:int|None,
        verbose:bool=False,
        ) -> list[str]:
    """
    Executes Multiple exploration methods on the data
    """
    # Input must be a simple 2D array
    y_df = X_df[TARGET]
    tmp = X_df.drop(columns=[TARGET, ID_COL])
    col_names = tmp.columns.tolist()
    X = tmp.values
    # Target must be 1D array
    y = y_df.to_numpy()
    y_bin, _, _ = stratify(y_df)
    y_bin = y_bin.to_numpy()
    cols = []
    reversed = []


    # Creates a dataframe to store the scores
    scores_df = pd.DataFrame(columns=["Feature"])
    scores_df["Feature"] = col_names

    # Feature Variance
    vt = VarianceThreshold(threshold=var_threshold).fit(X)
    var = vt.variances_.tolist()

    # Sorts the feature names by their variance on the dataframe
    cols_sorted = sorted([(col_names[i], var[i]) for i in range(len(col_names))], key=lambda x: x[1], reverse=True)
    col_names = [n for n, _ in cols_sorted]
    scores_df["Feature"] = col_names
    X = X_df[scores_df["Feature"].tolist()].values

    append_scores_log(scores_df, [v for _, v in cols_sorted], cols, "Variance")
    reversed.append(1)

    # append_scores_log(scores_df, boruta(X_df, y_df, rnd_state), cols, "Boruta")
    # reversed.append(1)

    # Better when both input and target are nominal
    # calculates the reduction in entropy from the transformation of a dataset
    mir = lambda X, y: mutual_info_regression(X, y_bin, random_state=rnd_state, discrete_features=True, copy=True)
    fs00 = SelectKBest(score_func=mir, k='all').fit(X, y_bin)
    append_scores_log(scores_df, fs00.scores_, cols, "Mutual Information Regression")
    reversed.append(1)

    """
    a correlation measure especially designed to evaluate the relationship between a binary and a continuous variable.
    is just a special case of Pearson's correlation,
    Makes strong normality assumptions
    """
    pbs = [pointbiserialr(X[:, f], y) for f in range(X.shape[1])]

    """
    Spearman’s rank correlation is an alternative to Pearson correlation for ratio/interval variables.
    As the name suggests, it only looks at the rank values,
    i.e. it compares the two variables in terms of the relative positions of particular data points within the variables.
      measures the strength of a monotonic relationship (as one increases, the other tends to either increase too)
    It is able to capture non-line to only considering the rank instead of the exact data points.
    """
    rho = [spearmanr(X[:, f], y, nan_policy='omit') for f in range(X.shape[1])]
    """
    Also rank-based like Spearman's rho, but uses concordant and discordant pairs of values,
     as opposed to Spearman’s calculations based on deviations
    It searches for pairs where if a value of one variable is higher than another,
      the corresponding value of the other variable is also higher (concordant),
      or lower (discordant).re robust to outliers in the data.
    """
    tau = [kendalltau(X[:, f], y, nan_policy='omit') for f in range(X.shape[1])]

    append_scores_log(scores_df, [p[0] for p in pbs], cols, "Point-Biserial r Statistic")
    reversed.append(1)
    append_scores_log(scores_df, [r[0] for r in rho], cols, "Spearman's rho Statistic")
    reversed.append(1)
    append_scores_log(scores_df, [t[0] for t in tau], cols, "Kendall's tau Statistic")
    reversed.append(1)

    """
    A p-value < 0.05 means the linear relationship is "statistically significant."
    A p-value > 0.05 means the linear relationship is not statistically significant (it could be due to random chance).
    """
    append_scores_log(scores_df, [p[1] for p in pbs], cols, "Point-Biserial r P-Value")
    reversed.append(0)
    append_scores_log(scores_df, [r[1] for r in rho], cols, "Spearman's rho P-Value")
    reversed.append(0)
    append_scores_log(scores_df, [t[1] for t in tau], cols, "Kendall's tau P-Value")
    reversed.append(0)
    scores_df.to_csv(f'{log_data_path}_FeatureSelection_Scores.csv', index=False)


    if verbose:
        for i in range(len(cols)):
            col = cols[i]
            rev = reversed[i] == 1
            print_scores(scores_df, col, rev)

    kept_cols = [n for n, v in cols_sorted if v >= var_threshold ]
    kept_cols += [TARGET, ID_COL, BIN_COL]

    return kept_cols


from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy # https://github.com/scikit-learn-contrib/boruta_py

def boruta(X: pd.DataFrame, y: pd.Series, rnd_state: int|None) -> list[int]:
    # gets feature importance ranking
    return BorutaPy(
        estimator=RandomForestRegressor(random_state=rnd_state, n_jobs=-1),
        verbose=0, random_state=rnd_state
        ).fit(X, y).ranking_.tolist()