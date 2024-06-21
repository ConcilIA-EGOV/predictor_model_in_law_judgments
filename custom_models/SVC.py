from sklearn.svm import SVC as svc


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


class SVC(svc):
    def __init__(self, **kwargs):
        if not kwargs:
            params = SVC_PARAMS
        else:
            params = kwargs
        try: 
            super().__init__(**params)
        except Exception as e:
            print(f"Erro ao instanciar o modelo {self.name}: {e}")
            super().__init__(**SVC_PARAMS)
        self.name = "Support Vector Classifier"
        self.params = params
    
    def fit(self, X, y):
        output = None
        try:
            output = super().fit(X, y)
        except Exception as e:
            print(f"Erro ao treinar o modelo {self.name}: {e}")
        return output
