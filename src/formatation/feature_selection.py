import pandas as pd

from util.parameters import LOG_DATA_PATH, RANDOM_STATE
from util.parameters import append_to_data_log_list, log_file_preprocessing as log_file
from formatation.feature_formatation import FUNCTIONS

def trim_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas não relacionadas ao experimento
    """
    df = remove_confactors(df)
    remove_columns = [col for col in df.columns if col not in FUNCTIONS.keys()]
    log_file.write(f"Removendo colunas: {remove_columns}\n")
    append_to_data_log_list('Features Removidas', remove_columns)
    df = df.drop(columns=remove_columns)
    return df

def remove_confactors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove confatores do DataFrame
    1. Identifica as colunas de confatores: culpa_exclusiva_consumidor e fechamento_aeroporto
    2. Se ambas as colunas existirem, separa as instâncias em dois grupos:
       - Instâncias sem confatores (ambas as colunas iguais a 0)
       - Instâncias com confatores (pelo menos uma das colunas igual a 1)
    3. Registra o número de instâncias removidas e salva as instâncias com confatores em um arquivo CSV
    4. Remove as colunas de confatores do DataFrame
    5. Retorna o DataFrame sem as colunas de confatores
    6. Loga todas as mudanças feitas no arquivo de log
    7. Se uma das colunas de confatores já foi removida, utiliza a outra coluna para filtrar as instâncias
    8. Se ambas as colunas de confatores foram removidas, não faz nada
    """
    conf1 = 'culpa_exclusiva_consumidor'
    conf2 = 'fechamento_aeroporto'
    remove = [conf1, conf2]
    if conf1 in df.columns and conf2 in df.columns:
        pro = df[(df[conf1] == 0) & (df[conf2] == 0)]
        con = df[(df[conf1] == 1) | (df[conf2] == 1)]
        append_to_data_log_list('Alteracoes nas Features', f"Removidas {con.shape[0]} instâncias que continham confactors: {conf1}, {conf2}")
        log_file.write(f"Removendo colunas de co-fatores: {conf1}, {conf2}.\n   --> Resultando em {pro.shape} instâncias sem confactors e {con.shape} instâncias com confactors.\n")
    else:
        if all(conf not in df.columns for conf in remove):
            log_file.write("Ambas as colunas de confatores já foram removidas. Nenhuma ação necessária.\n")
            # an empty dataframe with the same columns as df
            con = pd.DataFrame(columns=df.columns)
            remove.clear()
        else:
            if conf1 in df.columns:
                log_file.write(f"Coluna confator {conf2} já removida.\n")
                remove.remove(conf2)
            elif conf2 in df.columns:
                log_file.write(f"Coluna confator {conf1} já removida.\n")
                remove.remove(conf1)
            pro = df[df[remove[0]] == 0]
            con = df[df[remove[0]] == 1]
            append_to_data_log_list('Alteracoes nas Features', f"Removidas {con.shape[0]} instancias que continham confactor: {remove[0]}")
            log_file.write(f"Removendo coluna de co-fator: {remove[0]}.\n   --> Resultando em {pro.shape} instâncias sem confator e {con.shape} instâncias com confator.\n")
    con.to_csv(f'{LOG_DATA_PATH}_Confactors.csv', index=False)
    df = df.drop(columns=remove)
    return df


# Feature Selection Methods

# Statistical correlation methods
from scipy.stats import kendalltau, spearmanr, pointbiserialr
from sklearn.feature_selection import mutual_info_regression
import numpy as np
from sklearn.ensemble import RandomForestRegressor
# Filtering Methods
from sklearn.feature_selection import SelectKBest #, SelectPercentile
from boruta import BorutaPy # https://github.com/scikit-learn-contrib/boruta_py

def print_scores(scores_df: pd.DataFrame, col: str) -> None:
    scores = scores_df[col].tolist()
    cols = scores_df["Feature"].tolist()
    print(f"\n{col} Scores:")
    # formats a string to have 30 chars
    match = [(f"{cols[i]:30}:", v) for i, v in enumerate(scores)]
    match = sorted(match, key=lambda x: float(x[1]), reverse=True)
    match = [(f"{m[0]} {m[1]}" if float(m[1]) < 0.0 else f"{m[0]} 0{m[1]}") if abs(float(m[1])) < 10 else f"{m[0]} {m[1]}"  for m in match]
    log = " - " + "\n - ".join(match)
    print(log)
    # log_file.write(f"{log}\n")

def append_scores_log(scores_df: pd.DataFrame, scores, cols: list[str], col: str) -> None:
    cols.append(col)
    scores_df[col] = [f"{i:.20f}" if i else "0.0" for i in scores] # type: ignore

def filter_methods(X_df: pd.DataFrame, y_df: pd.Series):
    # Input must be a simple 2D array
    X = X_df.values
    # Target must be 1D array
    y = y_df.to_numpy()

    # Creates a dataframe to store the scores
    scores_df = pd.DataFrame(columns=["Feature"])
    scores_df["Feature"] = X_df.columns.tolist()
    
    # gets feature importance ranking
    ranking, _, _ = feature_importance(X, y)
    
    cols = X_df.columns.tolist()
    cols = sorted([(cols[i], ranking[i]) for i in range(len(cols))], key=lambda x: x[1])
    # Adds the feature names to the dataframe
    n_r = [c[1] for c in cols]
    scores_df["Feature"] = [c[0] for c in cols]
    X = X_df[scores_df["Feature"].tolist()].values

    cols = []
    append_scores_log(scores_df, n_r, cols, "Boruta Ranking")

    cols = []
    

    # Better when both input and target are nominal
    # calculates the reduction in entropy from the transformation of a dataset
    mir = lambda X, y: mutual_info_regression(X, y, random_state=RANDOM_STATE, discrete_features=True, copy=True)
    fs00 = SelectKBest(score_func=mir, k='all').fit(X, y)
    append_scores_log(scores_df, fs00.scores_, cols, "Mutual Information Regression")

    # a correlation measure especially designed to evaluate the relationship between a binary and a continuous variable.
    # is just a special case of Pearson's correlation,
    # Makes strong normality assumptions
    pbs = [pointbiserialr(X[:, f], y) for f in range(X.shape[1])]

    # Spearman’s rank correlation is an alternative to Pearson correlation for ratio/interval variables.
    # As the name suggests, it only looks at the rank values,
    # i.e. it compares the two variables in terms of the relative positions of particular data points within the variables.
    #   measures the strength of a monotonic relationship (as one increases, the other tends to either increase too)
    # It is able to capture non-linear relations, but there are no free lunches:
    #   we lose some information due to only considering the rank instead of the exact data points.
    rho = [spearmanr(X[:, f], y, nan_policy='omit') for f in range(X.shape[1])]

    # Also rank-based like Spearman's rho,
    # but uses concordant and discordant pairs of values, as opposed to Spearman’s calculations based on deviations
    # It searches for pairs where if a value of one variable is higher than another,
    #   the corresponding value of the other variable is also higher (concordant),
    #   or lower (discordant).
    # Kendall is often regarded as more robust to outliers in the data.
    tau = [kendalltau(X[:, f], y, nan_policy='omit') for f in range(X.shape[1])]

    append_scores_log(scores_df, [p[0] for p in pbs], cols, "Point-Biserial r Statistic")
    append_scores_log(scores_df, [r[0] for r in rho], cols, "Spearman's rho Statistic")
    append_scores_log(scores_df, [t[0] for t in tau], cols, "Kendall's tau Statistic")
    
    # A p-value < 0.05 means the linear relationship is "statistically significant."
    # A p-value > 0.05 means the linear relationship is not statistically significant (it could be due to random chance).
    append_scores_log(scores_df, [p[1] for p in pbs], cols, "Point-Biserial r P-Value")
    append_scores_log(scores_df, [r[1] for r in rho], cols, "Spearman's rho P-Value")
    append_scores_log(scores_df, [t[1] for t in tau], cols, "Kendall's tau P-Value")
    

    # # Uses pearson-r internally
    # # Only for Linear Relationships, Assuming Normality on input and target
    # fs11 = f_regression(X, y)
    # append_scores_log(scores_df, fs11[0], cols, "F Regression (F-Statistic)")
    # append_scores_log(scores_df, fs11[1], cols, "F Regression (P-value)")

    scores_df.to_csv(f'{LOG_DATA_PATH}_FeatureSelection_Scores.csv', index=False)
    
    for col in cols:
        print_scores(scores_df, col)
    

def feature_importance(X: np.ndarray, y: np.ndarray) -> tuple[list, list, list]:
    """
    Feature importance techniques assign a score to each feature based on its contribution
    to the predictive model's performance.
    ---
    Common methods include:
    - Tree-based models (e.g., Random Forest, Gradient Boosting)
    - Permutation Importance
    - SHAP Values (SHapley Additive exPlanations)
    """
    est = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    bor = BorutaPy(estimator=est, verbose=0, random_state=RANDOM_STATE).fit(X, y)
    ranking = bor.ranking_.tolist()
    support = bor.support_.tolist()
    sur_weak = bor.support_weak_.tolist()
    return ranking, support, sur_weak

# chi2 and f_classif are for classification tasks
# r_regression just uses pearson correlation for all features
# from sklearn.feature_selection import chi2, f_classif, r_regression, f_regression
# association is for nominal data
# from scipy.stats.contingency import association

# Remove unsupervised methods, due to being unsuited to find the best features alone
'''
from sklearn.feature_selection import VarianceThreshold
def unsupervised_methods(X: pd.DataFrame) -> pd.DataFrame:
    """
    Unsupervised methods use the intrinsic properties of the data to select predictors
    without involving any predictive model.
    ---
    This function currently implements the Variance Threshold method,
    which removes features with low variance across samples.
    Features with variance below the specified threshold are considered less informative
    and are removed from the dataset.
    """
    vt = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    vt.fit(X)
    var = vt.variances_.tolist()
    print(f"VarianceThreshold variances: {var}\n")
    X_vt = vt.fit_transform(X)
    print(f"VarianceThreshold reduced features from {X.shape[1]} to {X_vt.shape[1]}")
    removed_cols = [X.columns[i] for i in range(len(var)) if var[i] < VARIANCE_THRESHOLD]
    print(f"VarianceThreshold removed columns: {removed_cols}\n")
    X_vt = pd.DataFrame(X_vt, columns=vt.get_feature_names_out())
    return X_vt
'''
