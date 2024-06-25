from sklearn.neighbors import KNeighborsClassifier

# Default parameters
PARAMS = {
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
    def __init__(self, n_neighbors=10, weights='uniform', algorithm='ball_tree',
                 leaf_size=30, p=2, metric='minkowski', metric_params=None,
                 n_jobs=-1):
        try: 
            super().__init__(
                n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params,
                 n_jobs=n_jobs)
        except Exception as e:
            print(f"Erro ao instanciar o modelo KNN: {e}")
            super().__init__(**PARAMS)
        self.name = "K-Nearest Neighbors"
    
    def fit(self, X, y):
        output = self
        try:
            output = super().fit(X, y)
        except Exception as e:
            print(f"Erro ao treinar o modelo {self.name}: {e}")
        return output
