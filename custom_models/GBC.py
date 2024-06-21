from sklearn.ensemble import GradientBoostingClassifier


GB_PARAMS = {
    "loss": "log_loss",
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 0.8,
    "criterion": "friedman_mse",
    "min_samples_split": 2,
    "min_samples_leaf": 10,
    "min_weight_fraction_leaf": 0.0,
    "max_depth": 5,
    "min_impurity_decrease": 0.0,
    "init": None,
    "random_state": 0,
    "verbose": False,
    "max_features": None,
    "max_leaf_nodes": 2,
    "warm_start": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
    "tol": 0.0001,
    "ccp_alpha": 0.0
}


class GBC(GradientBoostingClassifier):
    def __init__(self, **kwargs):
        print(**kwargs)
        if not kwargs:
            params = GB_PARAMS
        else:
            params = kwargs
        try: 
            super().__init__(**params)
        except Exception as e:
            print(f"Erro ao instanciar o modelo {self.name}: {e}")
            super().__init__(**GB_PARAMS)
        self.name = "Gradient Boosting Classifier"
        self.params = params
    
    def fit(self, X, y):
        output = None
        try:
            output = super().fit(X, y)
        except Exception as e:
            print(f"Erro ao treinar o modelo {self.name}: {e}")
        return output
