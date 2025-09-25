from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# -*- coding: utf-8 -*-
from src.util.parameters import RANDOM_STATE, MODEL_NAME


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
    'n_jobs': [-1],
    'random_state': [RANDOM_STATE],
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
    'max_features': [None, 'sqrt', 'log2', 1.0],
    'random_state': [RANDOM_STATE],
    'max_leaf_nodes': [None, 10, 20, 30],
    'min_impurity_decrease': [0.0, 0.1, 0.2],
    'ccp_alpha': [0.0, 0.1, 0.2]
}


param_grid = {
    'RandomForest': param_grid_RandForest,
    'DecisionTree': param_grid_DecisionTree
}


DT_PARAMS = {
    'random_state': RANDOM_STATE,
    'criterion': 'poisson',
    'min_samples_split': 2,
    'max_features': 1.0,
    'max_depth': 15,
}


RF_PARAMS = {
    'random_state': RANDOM_STATE,
    'min_samples_split': 2,
    'criterion': 'poisson',
    'n_estimators': 330,
    'max_features': 1.0,
    'max_depth': 15,
    'n_jobs': -1,
}

MODEL_PARAMS = "DecisionTree"

if MODEL_NAME == "DecisionTree":
    MODEL_PARAMS = DT_PARAMS
elif MODEL_NAME == "RandomForest":
    MODEL_PARAMS = RF_PARAMS
else:
    raise ValueError(f"Modelo {MODEL_NAME} não suportado.")

def get_model_instance(model_name: str):
    if model_name == "DecisionTree":
        return DecisionTreeRegressor(**DT_PARAMS)
    elif model_name == "RandomForest":
        return RandomForestRegressor(**RF_PARAMS)
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")
