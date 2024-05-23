###
import sys
import os




# Obtém o diretório atual do script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Adiciona o diretório base do projeto ao caminho de busca do Python
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
###
from sklearn.calibration import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import SGDClassifier
# -----------
from sklearn.model_selection import GridSearchCV
from util.parameters import FILE_PATH, RESULTS_COLUMN
from formatation.input_formatation import load_data, separate_features_labels
from src.preprocessing import preprocessing

def grid_search(X_train, y_train, classifier, param_grid):

    # Realizar a busca em grade
    grid_search = GridSearchCV(classifier, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Melhor combinação de hiperparâmetros
    best_params = grid_search.best_params_
    return best_params

# Definir os hiperparâmetros a serem ajustados
param_grid = {
    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'eta0': [0.01, 0.1, 1],
    'max_iter': [10000, 15000],
    'tol': [1e-3, 1e-4, 1e-5]
}



models = {'SDG': SGDClassifier(),
          'LinearSVC': LinearSVC(),
          'KNN': KNeighborsClassifier(),
          'SVC': SVC(),
          'GradientBoosting': GradientBoostingClassifier()
        }

if __name__ == "__main__":
    data = load_data(FILE_PATH)
    X, y = separate_features_labels(data, RESULTS_COLUMN)
    X, y = preprocessing(X, y)
    best_params_all = dict()
    grid_search(X, y, SGDClassifier(), param_grid)
    '''for key, model in models.items():
        best_params_all[key] = grid_search(X, y, model, param_grid)
    '''
