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
        self.name = "Gradient Boosting Classifier"
        # Use GB_PARAMS if no parameters are provided
        params = GB_PARAMS.copy()
        if kwargs:
            params.update(kwargs)
        try:
            super().__init__(**params)
            self.params = params
        except Exception as e:
            print(f"Erro ao instanciar o modelo {self.name} com parâmetros fornecidos: {e}")
            # Tente inicializar com GB_PARAMS caso os parâmetros fornecidos sejam inválidos
            try:
                super().__init__(**GB_PARAMS)
                self.params = GB_PARAMS
            except Exception as e2:
                print(f"Erro ao instanciar o modelo {self.name} com parâmetros padrão: {e2}")
                raise e2  # Relevante para interromper a execução caso ambos falhem

    def fit(self, X, y):
        output = None
        try:
            output = super().fit(X, y)
        except Exception as e:
            print(f"Erro ao treinar o modelo {self.name}: {e}")
        return output

# Exemplo de uso
if __name__ == "__main__":
    # Parâmetros incorretos para testar a recuperação
    wrong_params = {"max_depth": "deep"}
    model = GBC(**wrong_params)

    # Parâmetros corretos
    model = GBC(**GB_PARAMS)
