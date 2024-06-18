from sklearn.calibration import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Perceptron

KNN_PARAMS = {
    "n_neighbors": 10,
    "weights": "uniform",
    "algorithm": "ball_tree",
    "leaf_size": 30,
    "p": 2,
    "metric": "minkowski",
    "metric_params": None,
    "n_jobs": -1
}

LINEARSVC_PARAMS = {
    'C': 0.1,
    'class_weight': None,
    'dual': True,
    'fit_intercept': True,
    'intercept_scaling': 0.1,
    'loss': 'squared_hinge',
    'max_iter': 1000,
    'multi_class': 'ovr',
    'penalty': 'l2',
    'random_state': None,
    'tol': 0.0001,
    'verbose': 0
}

SVC_PARAMS = {
    "C": 10, 
    "break_ties": False,
    "cache_size": 1,
    "class_weight": None,
    "coef0": 0.0,
    "decision_function_shape": "ovr",
    "degree": 2,
    "gamma": 2,
    "kernel": "rbf",
    "max_iter": 1000,
    "probability": True,
    "random_state": None,
    "shrinking": True,
    "tol": 0.0001,
    "verbose": False
}

MODELS = {
    'KNN': KNeighborsClassifier(**KNN_PARAMS),
    'LinearSVC': LinearSVC(**LINEARSVC_PARAMS),
}
