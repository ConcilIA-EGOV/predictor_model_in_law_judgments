from sklearn.ensemble import GradientBoostingClassifier

# Default parameters
PARAMS = {
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
    def __init__(self, loss='log_loss', learning_rate=0.1, n_estimators=100, subsample=0.8, 
                 criterion='friedman_mse', min_samples_split=2, min_samples_leaf=10, 
                 min_weight_fraction_leaf=0.0, max_depth=5, min_impurity_decrease=0.0, 
                 init=None, random_state=0, verbose=False, max_features=None, 
                 max_leaf_nodes=2, warm_start=True, validation_fraction=0.1,
                 n_iter_no_change=10, tol=0.0001, ccp_alpha=0.0):
        try: 
            super().__init__(
                loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, 
                subsample=subsample, criterion=criterion, min_samples_split=min_samples_split, 
                 min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, 
                 max_depth=max_depth, min_impurity_decrease=min_impurity_decrease, 
                 init=init, random_state=random_state, verbose=verbose, max_features=max_features, 
                 max_leaf_nodes=max_leaf_nodes, warm_start=warm_start, validation_fraction=validation_fraction,
                 n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)
        except Exception as e:
            print(f"Erro ao instanciar o modelo: {e}")
            super().__init__(**PARAMS)
        self.name = "Gradient Boosting Classifier"
    
    def fit(self, X, y):
        output = self
        try:
            output = super().fit(X, y)
        except Exception as e:
            print(f"Erro ao treinar o modelo {self.name}: {e}")
        return output


    def get_params(self, deep=True):
        return super().get_params(deep)


    def set_params(self, **params):
        super().set_params(**params)
        return self

