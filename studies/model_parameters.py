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
from sklearn.calibration import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Perceptron
# -----------
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from util.parameters import FILE_PATH, CV
from util.param_grids import param_grid
from formatation.input_formatation import load_data, separate_features_labels
from src.preprocessing import preprocessing

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

def grid_search(X_train, y_train, classifier, param_grid, cv_=5):

    # Realizar a busca em grade
    grid_search = GridSearchCV(estimator=classifier,
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=cv_, n_jobs=-1, refit=True)
    grid_search.fit(X_train, y_train)

    # Melhor combinação de hiperparâmetros
    best_params = grid_search.best_params_
    # Melhor score
    test_best_model(grid_search, X_train, y_train)
    return best_params

models = {
    'SVC': SVC(),
    'GradientBoosting': GradientBoostingClassifier(),
    'Perceptron': Perceptron()
}

if __name__ == "__main__":
    data = load_data(FILE_PATH)
    X, y = separate_features_labels(data)
    X = preprocessing(X)
    best_params_all = dict()
    for key, model in models.items():
        print(f"Testing {key}")
        try:
            best_params_all[key] = grid_search(X, y, model, param_grid[key], CV)
            json.dump(best_params_all[key], open("best_parameters__"+key+".txt", "w"))
        except Exception as e:
            print(e)
            best_params_all[key] = str(e)
            json.dump(best_params_all[key], open("best_parameters__"+key+".txt", "w"))
    
    json.dump(best_params_all, open("best_parameters.txt", "w"))
    
