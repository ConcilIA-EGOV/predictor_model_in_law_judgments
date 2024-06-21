from sklearn.calibration import LinearSVC


LSVC_PARAMS = {
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


class LSVC(LinearSVC):
    def __init__(self, params=LSVC_PARAMS):
        super().__init__(**params)
        self.name = "Linear Support Vector Classifier"
        self.params = params
    
    def fit(self, X, y):
        output = None
        try:
            output = super().fit(X, y)
        except Exception as e:
            print(f"Erro ao treinar o modelo {self.name}: {e}")
        return output
