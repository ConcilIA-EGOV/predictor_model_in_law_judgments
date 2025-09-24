from sklearn.ensemble import RandomForestRegressor

from src.util.parameters import RANDOM_STATE

def get_model():
    model = RandomForestRegressor(n_estimators=330, min_samples_split=2,
                                  max_features=1.0, criterion='poisson',
                                  random_state=RANDOM_STATE, n_jobs=-1, max_depth=15)
    return model
