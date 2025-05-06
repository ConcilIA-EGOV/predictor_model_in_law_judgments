# -*- coding: utf-8 -*-

param_grid_RandForest = {
    'n_estimators': [200, 330, 400],
    'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
    'max_depth': [None, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
    'max_features': ['sqrt', 'log2', None, 1.0],
    'max_leaf_nodes': [None, 10, 20, 30],
    'min_impurity_decrease': [0.0, 0.1, 0.2],
    'bootstrap': [True, False],
    'oob_score': [True, False],
    'n_jobs': [None, -1],
    'random_state': [None, 15],
    'verbose': [0],
    'warm_start': [False, True],
    'ccp_alpha': [0.0, 0.1, 0.2],
    'max_samples': [None, 0.5, 0.75, 1.0]
}

param_grid_DecisionTree = {
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
    'max_features': [None, 'auto', 'sqrt', 'log2', 1.0],
    'random_state': [None, 15],
    'max_leaf_nodes': [None, 10, 20, 30],
    'min_impurity_decrease': [0.0, 0.1, 0.2],
    'ccp_alpha': [0.0, 0.1, 0.2]
}

param_grid = {
    'RandomForest': param_grid_RandForest,
    'DecisionTree': param_grid_DecisionTree
}
