from joblib import dump
import json
import numpy as np
import pandas as pd

# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
from sklearn.ensemble import RandomForestRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
from sklearn.linear_model import LinearRegression
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
from sklearn.ensemble import GradientBoostingRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
from sklearn.svm import SVR
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
from sklearn.neural_network import MLPRegressor
# https://scikit-learn.org/stable/modules/naive_bayes.html
from sklearn.naive_bayes import GaussianNB
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
from sklearn.model_selection import GridSearchCV

from src.util.parameters import PARAM_GRIDS
from util.log_aux import get_data_log

MODELS = {
    "DecisionTree": DecisionTreeRegressor,
    "RandomForest": RandomForestRegressor,
    "GradientBoost": GradientBoostingRegressor,
    "LinearRegression": LinearRegression,
    "NeuralNetork": MLPRegressor,
    "SVM": SVR,
    "NaiveBayes": GaussianNB,
}

def grid_search(model_name: str, X, y) -> tuple:
    """
    Realiza um treinamento e cross-validação
    usando os dados entregues, instanciando o modelo, se seu nome for suportado em `src.util.parameters.MODELS`
    Retorna o melhor modelo, os melhores parâmetros e o score desse modelo
    """
    if not model_name in MODELS.keys():
        raise Exception("Model name not suported for custom grid_search function")

    # Realizar a busca em grade
    grid_search = GridSearchCV(
        verbose=0,
        n_jobs=-1,
        refit=True,
        cv=5,
        estimator=MODELS[model_name](),
        param_grid=PARAM_GRIDS[model_name],
        scoring='neg_root_mean_squared_error',
    )
    grid_search.fit(X, y)

    # Melhor combinação de hiperparâmetros
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_model, best_params, abs(best_score)


def test_model(model, X:pd.DataFrame,
               y:pd.Series, y_bin:pd.Series
    ) -> tuple[float, float, float, list[str], np.ndarray]:
    '''
    Testar o modelo usando o conjunto de teste
    Retorna RMSE, MAE, MAPE e resultados por faixa de valores
    '''
    # Make predictions
    predictions = model.predict(X)

    errors = (y - predictions)
    # Calculate the RMSE, MAE and proportional MAE overall
    rmse_all = np.sqrt(np.mean(errors**2))

    # Calculate the percentual MAE
    abs_error = np.abs(errors)
    mae_all = np.mean(abs_error)
    mape_all = np.mean(abs_error / y) * 100

    # Calculate the RMSE, and MAE for each fold
    df = pd.DataFrame({"y_true": y, "y_pred": predictions, "fold": y_bin})
    resultados = []
    for faixa, grupo in df.groupby('fold', observed=False):
        if len(grupo) == 0:
            continue
        faixa = int(faixa) # type: ignore
        y_true = grupo['y_true']
        y_pred = grupo['y_pred']
        mininmo = y_true.min()
        maximo = y_true.max()
        group_error = (y_true - y_pred)
        abs_error = np.abs(group_error)
        rmse = np.sqrt(np.mean(group_error**2))
        mae = np.mean(abs_error)
        mape = np.mean(abs_error / y_true) * 100
        resultados.append(f"""Faixa {
            faixa + 1
            }:\n\t - RMSE:  {round(rmse,2
            )}\n\t - MAE: {round(mae,2
            )}\n\t - MAPE: {round(mape,2
            )}%\n\t - N Amostras: {len(grupo
            )}\n\t - Valores: {mininmo} a {maximo}""")
    return (rmse_all, mae_all, mape_all, resultados, errors)


def save_model(model, model_name:str, model_file:str, save_path:str,
               bs_rmse:float, bs_mae:float, bs_mape:float, gs_score:float,
               folds:list[str], model_params: dict[str, str]):
    """
    Salvar o modelo treinado em um arquivo e os logs de performance em outro
    """
    log_file = open(f'{save_path}Model-log.txt', 'w')
    log_file.write('Parametros da Pipeline:\n')
    for key, value in get_data_log().items():
        if isinstance(value, list):
            if len(value) == 0:
                log_file.write(f' - {key}: []\n')
                continue
            log_file.write(f' - {key}: \n\t - {"\n\t - ".join(map(str, value))}\n')
            continue
        log_file.write(f' - {key}: {value}\n')
    log_file.write(f'\nParametros do Modelo ({model_name}):\n')
    for param, value in model_params.items():
        log_file.write(f' - {param}: {value}\n')
    with open(f"{save_path}Parametros.json", "w") as f:
        json.dump(model_params, f, indent=4)


    log_file.write('\nPerformance do Modelo Base:\n')

    log_file.write(f' - RMSE:     {bs_rmse:.2f}\n')
    log_file.write(f' - MAE:      {bs_mae:.2f}\n')
    log_file.write(f' - GS SCORE: {gs_score:.2f}\n')
    log_file.write(f' - MAPE:       {bs_mape:.2f}%\n')

    [log_file.write(f' - {folds[i]}\n') for i in range(len(folds))]

    # Log dos dados usados
    log_file.close()
    # Save the base model
    dump(model, model_file)
