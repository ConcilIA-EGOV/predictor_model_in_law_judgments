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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib  # Para salvar o modelo
from util.param_grids import param_grid
###
from src.file_op import load_data

def grid_search(X, y, model, param_grid):

    # Realizar a busca em grade
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               scoring='neg_root_mean_squared_error',
                               n_jobs=-1, refit=True)
    grid_search.fit(X, y)

    # Melhor combinação de hiperparâmetros
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_
    # Melhor score
    print("Best parameters found: ", best_params)
    print("Best cross-validation score: {:.2f}".format(best_score))
    return best_params, best_score, best_model

if __name__ == "__main__":
    X, y, _, _ = load_data(split=False)
    best_params_all = dict()
    models = {
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor()
    }
    for key, model in models.items():
        print(f"Testing {key}")
        try:
            best_params, best_score, best_model = grid_search(X, y, model, param_grid[key])
            best_params_all[key] = dict()
            best_params_all[key]['params'] = best_params
            best_params_all[key]['score'] = best_score
            # Save the base model
            joblib.dump(best_model, f'logs/best_{key}.pkl')
        except Exception as e:
            print(e)
            best_params_all[key] = [str(e)]
        with open("logs/best_parameters__"+key+".json", "w") as f:
            json.dump(best_params_all[key], f, indent=4)
    
