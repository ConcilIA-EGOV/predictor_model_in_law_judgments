###
import sys
import os

# Obtém o diretório atual do script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Adiciona o diretório base do projeto ao caminho de busca do Python
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
###
from joblib import dump, load
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from util.parameters import BEST_SCORE_STORAGE, MAIN_MODEL_FILE, REFIT
from custom_models import MODELS

def split_train_test(X, y, test_size, y_bin=None):
    """
    Dividir em conjuntos de treino e teste
    """
    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y, test_size=test_size, stratify=y_bin, random_state=15)
    #return X, X, y, y
    return X_train, X_test, y_train, y_test


def train_model(model, X, y):
    """
    Treinar o modelo usando o conjunto de treino.
    """    
    model.fit(X, y)        
    return model


def test_model(model, X, y, y_bin):
    '''
    Testar o modelo usando o conjunto de teste
    '''
    #score = classification_report(y, model.predict(X), output_dict=True)
    # score = cross_val_score(model, X, y, cv=FOLDS, n_jobs=-1)
    # Make predictions
    predictions = model.predict(X)
    
    # Calculate the RMSE, and MAE overall
    rmse_all = root_mean_squared_error(y, predictions)
    mae_all = mean_absolute_error(y, predictions)

    # Calculate the RMSE, and MAE for each fold
    df = pd.DataFrame({"y_true": y, "y_pred": predictions, "fold": y_bin})
    resultados = []
    for faixa, grupo in df.groupby('fold', observed=False):
        if len(grupo) == 0:
            continue
        mae = mean_absolute_error(grupo['y_true'], grupo['y_pred'])
        rmse = root_mean_squared_error(grupo['y_true'], grupo['y_pred'])
        resultados.append(f"""Faixa {int(faixa) + 1}:\n\t\tMAE: {round(mae, 2)}\n\t\tRMSE: {round(rmse, 2)}\n\t\tN Amostras: {len(grupo)}""")
    return (rmse_all, mae_all, resultados)


def feature_importance(model, X):
    importances = model.feature_importances_
    features = X.columns
    sorted_idx = importances.argsort()

    plt.figure(figsize=(10,6))
    plt.barh(features[sorted_idx], importances[sorted_idx])
    plt.title('Importância das Features')
    plt.tight_layout()
    # since FigureCanvasAgg is non-interactive, and thus cannot be show
    plt.savefig("feature_importance.png")
    plt.close()


def save_model(model, score):
    """
    Salvar o modelo treinado em um arquivo e o score em outro
    """
    dict_score = {"best_score": score}
    json.dump(dict_score, open(BEST_SCORE_STORAGE, "w"))
    dump(model, MAIN_MODEL_FILE)
    return

def get_best_score():
    return json.load(open(BEST_SCORE_STORAGE, "r"))["best_score"]

def get_models() -> dict[str, object]:
    if not REFIT:
        return MODELS
    model = load(MAIN_MODEL_FILE)
    name = 'DecisionTree'
    #name = model.__str__
    return {name: model}

def is_best_model(score):
    best_score = get_best_score()
    scr = -1
    tipo = type(score)
    if tipo == float:
        scr = score
    elif tipo == dict:
        scr = score["accuracy"]
    elif tipo == np.ndarray:
        scr = score.mean()
    else:
        raise TypeError("Tipo de score não suportado")
    return scr > best_score

def print_results(key, cv_score):
    print(f"Model: {key}")
    prt_cv = " - ".join([f"{(score)*100:.2f}%" for score in cv_score])
    print(f"Cross Validation Scores: {prt_cv}")
    print(f"Cross Validation Mean: {(cv_score.mean())*100:.2f}%\n")


if __name__ == "__main__":
    print(get_models())
