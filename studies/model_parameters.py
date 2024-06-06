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
    grid_search = GridSearchCV(classifier, param_grid, scoring='accuracy', cv=cv_)
    grid_search.fit(X_train, y_train)

    # Melhor combinação de hiperparâmetros
    best_params = grid_search.best_params_
    return best_params


param_grid_LSVC = {
    'penalty': ['l1', 'l2'],
    'loss': ['squared_hinge', 'hinge'],
    'dual': [True, False],
    'tol': [0.0001],
    'C': [1],
    'multi_class': ['ovr', 'crammer_singer'],
    'fit_intercept': [True, False],
    'intercept_scaling': [1.0],
    'class_weight': ['Mapping', 'str', None],
    'verbose': [0],
    'random_state': ['Int', 'RandomState', None],
    'max_iter': [1000]
}

param_grid_KNN = {
    'n_neighbors': [5],
    'weights': ["((...) -> Any)", ['uniform', 'distance'], None],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [30],
    'p': [2],
    'metric': ['str', '((...) -> Any)', "minkowski"],
    'metric_params': [dict(), None],
    'n_jobs': 'Int | None = None'
}

param_grid_SVC = {
    'C': [1],
    'kernel': ['((...) -> Any)', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']],
    'degree': [3],
    'gamma': [float, ['scale', 'auto']],
    'coef0': [0],
    'shrinking': [True, False],
    'probability': [True, False],
    'tol': [0.001],
    'cache_size': [200],
    'class_weight': ['Mapping', 'str', None],
    'verbose': [True, False],
    'max_iter': [1000],
    'decision_function_shape': ['ovo', 'ovr'],
    'break_ties': [True, False],
    'random_state': ['Int', 'RandomState', None]
}

param_grid_GB = {
    'loss': ['log_loss', 'exponential'],
    'learning_rate': [0.1],
    'n_estimators': [100],
    'subsample': [1.0],
    'criterion': ['friedman_mse', 'squared_error'],
    'min_samples_split': [2, 10, 100, 1000],
    'min_samples_leaf': [1, 10, 100, 1000],
    'min_weight_fraction_leaf': [0.0],
    'max_depth': [None, 3, 10, 100],
    'min_impurity_decrease': [0.0],
    'init': ['str', 'BaseEstimator', None],
    'random_state': ['Int', 'RandomState', None],
    'max_features': ['float', 'int', ['auto', 'sqrt', 'log2'], None],
    'verbose': [0],
    'max_leaf_nodes': ['Int', None],
    'warm_start': [True, False],
    'validation_fraction': [0.1],
    'n_iter_no_change': ['Int', None],
    'tol': [0.0001],
    'ccp_alpha': [0.0]
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
    grid_search(X, y, GradientBoostingClassifier(), param_grid['GradientBoosting'], CV)
    '''
    for key, model in models.items():
        best_params_all[key] = grid_search(X, y, model, param_grid[key], CV)
    '''
