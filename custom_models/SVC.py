from sklearn.svm import SVC as svc


PARAMS = {
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
    def __init__(self, C=10, break_ties=False,cache_size=1,
                 class_weight=None, coef0=0.0, decision_function_shape='ovr',
                 degree=2, gamma=2, kernel='rbf', max_iter=1000,
                 probability=True, random_state=None, shrinking=True,
                 tol=0.0001, verbose=False):
        try: 
            super().__init__(C=C, break_ties=break_ties, cache_size=cache_size,
                             class_weight=class_weight, coef0=coef0,
                             decision_function_shape=decision_function_shape,
                             degree=degree, gamma=gamma, kernel=kernel,max_iter=max_iter,
                             probability=probability, random_state=random_state,
                             shrinking=shrinking, tol=tol, verbose=verbose)
        except Exception as e:
            print(f"Erro ao instanciar o modelo SVC: {e}")
            super().__init__(**PARAMS)
        self.name = "Support Vector Classifier"
    
    def fit(self, X, y):
        output = self
        try:
            output = super().fit(X, y)
        except Exception as e:
            print(f"Erro ao treinar o modelo {self.name}: {e}")
        return output
