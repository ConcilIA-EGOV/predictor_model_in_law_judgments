###
import sys
import os

# Obtém o diretório atual do script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Adiciona o diretório base do projeto ao caminho de busca do Python
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
###
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.kernel_approximation import SkewedChi2Sampler
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
PARAM_GRID_SDG = {
            'loss': ['hinge', 'log', 'modified_huber',
                    'squared_hinge', 'perceptron',
                    'squared_loss', 'huber',
                    'epsilon_insensitive',
                    'squared_epsilon_insensitive'],
            'max_iter': [0, 1000, 10000],
            'tol': [1e-3, 1e-4, 1e-5]
}


models = {'SDG': SGDClassifier(),
          'MultinomialNB': MultinomialNB(),
          'RBFSampler': RBFSampler(),
          'Nystroem': Nystroem(),
          'AdditiveChi2Sampler': AdditiveChi2Sampler(),
          'PolynomialCountSketch': PolynomialCountSketch(),
          'SkewedChi2Sampler': SkewedChi2Sampler()
        }

if __name__ == "__main__":
    data = load_data(FILE_PATH)
    X, y = separate_features_labels(data, RESULTS_COLUMN)
    X, y = preprocessing(X, y)
    best_params_all = dict()
    for key, model in models.items():
        best_params_all[key] = grid_search(X, y, model, PARAM_GRID_SDG)
