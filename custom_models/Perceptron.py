from sklearn.linear_model import Perceptron as pcn
from util.parameters import NUM_EPOCHS
import numpy as np


PERCEPTON_PARAMS = {
    "penalty": "elasticnet",
    "alpha": 0.0001,
    "l1_ratio": 0.75,
    "fit_intercept": True,
    "max_iter": 10000,
    "tol": 0.001,
    "shuffle": True,
    "eta0": 1.0,
    "n_jobs": -1,
    "random_state": 200,
    "early_stopping": True,
    "validation_fraction": 0.2,
    "n_iter_no_change": 10,
    "class_weight": None,
    "warm_start": False
}


class Perceptron(pcn):
    def __init__(self, **kwargs):
        if not kwargs:
            params = PERCEPTON_PARAMS
        else:
            params = kwargs
        try: 
            super().__init__(**params)
        except Exception as e:
            print(f"Erro ao instanciar o modelo {self.name}: {e}")
            super().__init__(**PERCEPTON_PARAMS)
        self.name = "Perceptron"
        self.params = params
    
    def fit(self, X, y):
        output = None
        try:
            classes = np.unique(y)
            output = super().fit(X, y)
            for _ in range(NUM_EPOCHS):
                output.partial_fit(X, y, classes=classes)
        except Exception as e:
            print(f"Erro ao treinar o modelo {self.name}: {e}")
        return output
