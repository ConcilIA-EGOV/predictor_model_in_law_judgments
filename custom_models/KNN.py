from sklearn.neighbors import KNeighborsClassifier

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

class KNN(KNeighborsClassifier):
    def __init__(self, **kwargs):
        if not kwargs:
            params = KNN_PARAMS
        else:
            params = kwargs
        try: 
            super().__init__(**params)
        except Exception as e:
            print(f"Erro ao instanciar o modelo {self.name}: {e}")
            super().__init__(**KNN_PARAMS)
        self.name = "K-Nearest Neighbors"
        self.params = params
    
    def fit(self, X, y):
        output = None
        try:
            output = super().fit(X, y)
        except Exception as e:
            print(f"Erro ao treinar o modelo {self.name}: {e}")
        return output
