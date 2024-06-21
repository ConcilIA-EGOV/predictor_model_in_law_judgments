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
from util.parameters import CV, BEST_SCORE_STORAGE, MAIN_MODEL_FILE, REFIT
from custom_models import MODELS
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split


def split_train_test(X, y):
    """
    Dividir em conjuntos de treino e teste
    """
    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def train_model(model, X, y):
    """
    Treinar o modelo usando o conjunto de treino.
    """    
    model.fit(X, y)        
    return model


def test_model(model, X, y):
    '''
    Testar o modelo usando o conjunto de teste
    '''
    #score = classification_report(y, model.predict(X), output_dict=True)
    score = cross_val_score(model, X, y, cv=CV)
    return score


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
    name = model.__str__
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
