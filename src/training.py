from joblib import dump
import numpy as np
import pandas as pd
# https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning

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

from util.parameters import MODELS_PARAMS
from util.log_aux import get_data_log


MODELS_CLS = {
    "DecisionTree": DecisionTreeRegressor,
    "RandomForest": RandomForestRegressor,
    "GradientBoost": GradientBoostingRegressor,
    "LinearRegression": LinearRegression,
    "NeuralNetork": MLPRegressor,
    "NaiveBayes": GaussianNB,
    "SVM": SVR,
}


def get_model_instance(model_name: str):
    try:
        return MODELS_CLS[model_name](**MODELS_PARAMS[model_name])
    except:
        raise ValueError(f"Modelo desconhecido: {model_name}")


def train_model(model, X, y):
    """
    Treinar o modelo usando o conjunto de treino.
    """
    model.fit(X, y)
    return model


def test_model(model, X:pd.DataFrame, y:pd.Series, y_bin:pd.Series) -> tuple[float, float, float, float, list[str], np.ndarray]:
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
    mape_x_all = np.mean(abs_error / predictions) * 100

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
        mape_x = np.mean(abs_error / y_pred) * 100
        resultados.append(f"""Faixa {
            faixa + 1
            }:\n\t - RMSE:  {round(rmse,2
            )}\n\t - MAE: {round(mae,2
            )}\n\t - MAPE: {round(mape,2
            )}%\n\t - MAPE X: {round(mape_x,2
            )}%\n\t - N Amostras: {len(grupo
            )}\n\t - Valores: {mininmo} a {maximo}""")
    return (rmse_all, mae_all, mape_all, mape_x_all, resultados, errors)


def save_model(model, model_name:str, model_file:str, save_path:str, bs_rmse:float, bs_mae:float,
               bs_mape:float, bs_mape_x:float, folds:list[str]):
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
    for param, value in MODELS_PARAMS[model_name].items():
        log_file.write(f' - {param}: {value}\n')
    log_file.write('\nPerformance do Modelo Base:\n')
    print('\nPerformance do Modelo Base:')
    log_file.write(f' - RMSE:   {bs_rmse:.2f}\n')
    print(f' - RMSE:   {bs_rmse:.2f}')
    log_file.write(f' - MAE:    {bs_mae:.2f}\n')
    print(f' - MAE:    {bs_mae:.2f}')
    log_file.write(f' - MAPE:   {bs_mape:.2f}%\n')
    print(f' - MAPE:   {bs_mape:.2f}%')
    log_file.write(f' - MAPE X: {bs_mape_x:.2f}%\n')
    print(f' - MAPE X: {bs_mape_x:.2f}%')

    [log_file.write(f' - {folds[i]}\n') for i in range(len(folds))]

    # Log dos dados usados
    log_file.close()
    # Save the base model
    dump(model, model_file)
