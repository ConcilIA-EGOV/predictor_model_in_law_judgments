# Adiciona o diretório base do projeto ao caminho de busca do Python
import sys, os
this_path = os.path.dirname(os.path.abspath(__file__))
if not this_path in sys.path:
    sys.path.append(this_path)
this_path = os.path.dirname(this_path)
if not this_path in sys.path:
    sys.path.append(this_path)
###
import json
# -----------
from sklearn.model_selection import GridSearchCV
import joblib  # Para salvar o modelo

from util.parameters import FILE_PATH, LOG_PATH, LOG_DATA_PATH, MODELS_FOLDERS, param_grids
from src.formatation.preprocessing import load_data

from training import MODELS_CLS

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
    train, test = load_data(FILE_PATH, LOG_DATA_PATH)[0]
    X_train, y_train, _ = train
    X_test, y_test, _ = test
    X = X_train + X_test
    y = y_train + y_test
    best_params_all = dict()
    for key, model in MODELS_CLS.items():
        print(f"Testing {key}")
        try:
            best_params, best_score, best_model = grid_search(X, y, model, param_grids[key])
            best_params_all[key] = dict()
            best_params_all[key]['params'] = best_params
            best_params_all[key]['score'] = best_score
            # Save the base model
            joblib.dump(best_model, f'{MODELS_FOLDERS[key]}/best_{key}.pkl')
            with open(f"{MODELS_FOLDERS[key]}/best_parameters__{key}.json", "w") as f:
                json.dump(best_params_all[key], f, indent=4)
        except Exception as e:
            print(e)
            best_params_all[key] = [str(e)]
