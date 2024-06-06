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
from sklearn.calibration import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
# -----------
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from util.parameters import FILE_PATH, CV
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
                               cv=cv_, n_jobs=-1,
                               verbose=1, refit=True)
    grid_search.fit(X_train, y_train)

    # Melhor combinação de hiperparâmetros
    best_params = grid_search.best_params_
    return best_params


param_grid_LSVC = {
    'penalty': ['l1', 'l2'],
    'loss': ['squared_hinge', 'hinge'],
    'dual': [True, False],  # 'l1' penalty is not supported with dual=False
    'tol': [1e-4, 1e-3, 1e-2],
    'C': [0.01, 0.1, 1, 10, 100],
    'multi_class': ['ovr', 'crammer_singer'],
    'fit_intercept': [True, False],
    'intercept_scaling': [0.1, 0.5, 1.0, 2.0, 5.0],
    'class_weight': [None, 'balanced'],  # or a dictionary {class_label: weight}
    'verbose': [0, 1, 2],
    'random_state': [None, 42, 100, 200, np.random.RandomState(42)],  # values for reproducibility
    'max_iter': [1000, 2000, 3000, 4000, 5000]
}


param_grid_KNN = {
    'n_neighbors': [3, 5, 10, 15, 20, 30, 50],
    'weights': ['uniform', 'distance', lambda x: 1 / (x + 1e-5), lambda x: x ** 2, 'custom'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 20, 30, 40, 50, 60],
    'p': [1, 2],  # 1 for Manhattan distance, 2 for Euclidean distance
    'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev', 'hamming'],
    'metric_params': [None, {'p': 2}, {'w': 0.5}, {'p': 3}, {'w': 0.7}],
    'n_jobs': [-1]  # use all processors
}

param_grid_SVC = {
    'C': [0.1, 1, 10, 100, 1000],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    'degree': [2, 3, 4, 5],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'coef0': [0.0, 0.1, 0.5, 1],
    'shrinking': [True, False],
    'probability': [True, False],
    'tol': [1e-4, 1e-3, 1e-2],
    'cache_size': [200, 500, 1000],
    'class_weight': [None, 'balanced', {0: 1, 1: 10}, {0: 1, 1: 50}],
    'verbose': False,  # Geralmente mantido False para evitar log excessivo
    'max_iter': [-1, 1000, 5000],  # -1 para sem limite
    'decision_function_shape': ['ovo', 'ovr'],
    'break_ties': [True, False],
    'random_state': [None, 42, 100, 200, np.random.RandomState(42)]
}

param_grid_GB = {
    'loss': ['log_loss', 'exponential'],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 400, 500],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'criterion': ['friedman_mse', 'squared_error', 'mae'],
    'min_samples_split': [2, 10, 50, 100, 500, 1000],
    'min_samples_leaf': [1, 10, 50, 100, 500, 1000],
    'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
    'max_depth': [None, 3, 5, 10, 20, 50, 100],
    'min_impurity_decrease': [0.0, 0.01, 0.05, 0.1],
    'init': [None],  # ou uma instância de estimador, geralmente None
    'random_state': [42, 100, 200, np.random.RandomState(42), ],  # valores comuns para garantir replicabilidade
    'verbose': False,  # Geralmente mantido False para evitar log excessivo
    'max_features': [None, 'auto', 'sqrt', 'log2', 0.5, 0.7],
    'max_leaf_nodes': [None, 10, 20, 50, 100, 200],
    'warm_start': [True, False],
    'validation_fraction': [0.1, 0.2, 0.3],
    'n_iter_no_change': [None, 5, 10, 20],
    'tol': [1e-4, 1e-3, 1e-2],
    'ccp_alpha': [0.0, 0.01, 0.1]
}


param_grid = {
    'LinearSVC': param_grid_LSVC,
    'KNN': param_grid_KNN,
    'SVC': param_grid_SVC,
    'GradientBoosting': param_grid_GB
}


models = {
    'LinearSVC': LinearSVC(),
    'KNN': KNeighborsClassifier(),
    'SVC': SVC(),
    'GradientBoosting': GradientBoostingClassifier()
}

main_model = GradientBoostingClassifier()

if __name__ == "__main__":
    data = load_data(FILE_PATH)
    X, y = separate_features_labels(data)
    X, y = preprocessing(X, y)
    best_params_all = dict()
    # grid_search(X, y, GradientBoostingClassifier(), param_grid['GradientBoosting'], CV)
    for key, model in models.items():
        best_params_all[key] = grid_search(X, y, model, param_grid[key], CV)
    
    print(best_params_all)
    
