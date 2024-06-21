from sklearn.calibration import LinearSVC


# Default parameters
PARAMS = {
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
    def __init__(self, C=0.1, class_weight=None, dual=True, fit_intercept=True,
                 intercept_scaling=0.1, loss='squared_hinge', max_iter=1000,
                 multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                 verbose=0, **kwargs):
        try: 
            super().__init__(C=C, class_weight=class_weight, dual=dual,
                             fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                             loss=loss, max_iter=max_iter, multi_class=multi_class,
                             penalty=penalty, random_state=random_state, tol=tol,
                             verbose=verbose)
        except Exception as e:
            print(f"Erro ao instanciar o modelo LSVC: {e}")
            super().__init__(**PARAMS)
        self.name = "Linear Support Vector Classifier"
    
    def fit(self, X, y):
        output = None
        try:
            output = super().fit(X, y)
        except Exception as e:
            print(f"Erro ao treinar o modelo {self.name}: {e}")
        return output
