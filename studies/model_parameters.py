###
import sys
import os
# Obtém o diretório atual do script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Adiciona o diretório base do projeto ao caminho de busca do Python
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
###
import numpy as np
import json
# -----------
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from util.parameters import CV
from util.param_grids import param_grid
###
from formatation.input_formatation import load_data, separate_features_labels
from src.preprocessing import preprocessing
from src.training import get_models

def test_best_model(grid_search:GridSearchCV, X_test, y_test):
    '''
    Print the best parameters and best score
    '''
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

def grid_search(X, y, classifier, param_grid):

    # Realizar a busca em grade
    grid_search = GridSearchCV(estimator=classifier,
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=CV, n_jobs=-1, refit=True)
    grid_search.fit(X, y)

    # Melhor combinação de hiperparâmetros
    best_params = grid_search.best_params_
    # Melhor score
    test_best_model(grid_search, X, y)
    return best_params

if __name__ == "__main__":
    data = load_data()
    X, y = separate_features_labels(data)
    X = preprocessing(X)
    best_params_all = dict()
    models = get_models()
    for key, model in models.items():
        print(f"Testing {key}")
        try:
            best_params_all[key] = grid_search(X, y, model, param_grid[key])
        except Exception as e:
            print(e)
            best_params_all[key] = [str(e)]
        with open("logs/best_parameters__"+key+".json", "w") as f:
            json.dump(best_params_all[key], f, indent=4)
    
