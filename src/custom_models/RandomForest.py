from sklearn.ensemble import RandomForestRegressor

def get_model():
    model = RandomForestRegressor(n_estimators=330, min_samples_split=2,
                                  max_features=1.0, criterion='poisson',
                                  random_state=15, n_jobs=-1, max_depth=15)
    return model
