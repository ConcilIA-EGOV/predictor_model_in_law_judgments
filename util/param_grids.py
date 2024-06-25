param_grid_LSVC = {
    'penalty': ['l1', 'l2'],
    'loss': ['squared_hinge', 'hinge'],
    'dual': [True, False],  # 'l1' penalty is not supported with dual=False
    'tol': [1e-4, 1e-3, 1e-2],
    'C': [0.1, 1, 10],
    'multi_class': ['ovr', 'crammer_singer'],
    'fit_intercept': [True, False],
    'intercept_scaling': [0.1, 1.0, 5.0],
    'class_weight': [None, 'balanced'],  # or a dictionary {class_label: weight}
    'verbose': [0],
    'random_state': [None, 42, 100],  # values for reproducibility
    'max_iter': [1000]
}

param_grid_KNN = {
    'n_neighbors': [5, 10, 30, 50, 100],
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 30, 60, 100],
    'p': [1, 2],  # 1 for Manhattan distance, 2 for Euclidean distance
    'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev'],
    'metric_params': [None],
    'n_jobs': [-1]
}

param_grid_SVC = {
    'C': [0.1, 5, 10, 50, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4, 5],
    'gamma': ['scale', 0.5, 0.1, 1],
    'coef0': [0.0, 0.5, 1],
    'shrinking': [True, False],
    'probability': [True, False],
    'tol': [1e-4, 1e-3, 1e-5],
    'cache_size': [1, 5, 100, 200, 300],
    'class_weight': [None, 'balanced'],
    'verbose': [False],  # Geralmente mantido False para evitar log excessivo
    'max_iter': [-1, 1000, 10000],  # -1 para sem limite
    'decision_function_shape': ['ovr', 'ovo'],
    'break_ties': [False, True],
    'random_state': [None, 42, 100]
}

param_grid_GB = {
    'loss': ['log_loss'],
    'learning_rate': [0.1, 0.05, 0.2],
    'n_estimators': [100, 50, 200],
    'subsample': [0.8, 0.5, 1.0],
    'criterion': ['friedman_mse', 'squared_error'],
    'min_samples_split': [2, 10, 5],
    'min_samples_leaf': [0.5, 1, 5, 10],
    'min_weight_fraction_leaf': [0.0, 0.5, 0.3],
    'max_depth': [1, 3, 5, 10],
    'min_impurity_decrease': [0.0, 0.1, 0.5],
    'init': [None],
    'random_state': [0, 100, 200],
    'verbose': [False],
    'max_features': [None, 'sqrt', 'log2'],
    'max_leaf_nodes': [None, 2, 5, 10],
    'warm_start': [True, False],
    'validation_fraction': [0.1, 0.05, 0.3],
    'n_iter_no_change': [10, 20, 50],
    'tol': [1e-4, 1e-5, 1e-3],
    'ccp_alpha': [0.0, 0.01, 0.1]
}

param_grid_Perceptron = {
    'penalty': ['elasticnet', None],
    'alpha': [0.0001, 0.001, 0.00001, 0.005],
    'l1_ratio': [0, 0.15, 0.5, 0.75, 1],
    'fit_intercept': [True, False],
    'max_iter': [10000, 100000],
    'tol': [0.001, 0.0001, 0.005],
    'shuffle': [True, False],
    'eta0': [1.0, 0.5, 2.0],
    'n_jobs': [-1],
    'random_state': [0, 100, 200, 300, None],
    'early_stopping': [False],
    'validation_fraction': [0.1, 0.2, 0.3],
    'n_iter_no_change': [10, 5, 20],
    'class_weight': [None, 'balanced'],
    'warm_start': [True, False]
}



param_grid = {
    'LSVC': param_grid_LSVC,
    'KNN': param_grid_KNN,
    'SVC': param_grid_SVC,
    'GBC': param_grid_GB,
    'Perceptron': param_grid_Perceptron
}
