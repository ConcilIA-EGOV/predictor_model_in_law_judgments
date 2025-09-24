from sklearn.tree import DecisionTreeRegressor

from src.util.parameters import RANDOM_STATE

def get_model():
    model = DecisionTreeRegressor(min_samples_split=2, max_features=1.0,
                                  criterion='poisson', max_depth=15,
                                  random_state=RANDOM_STATE)
    return model
