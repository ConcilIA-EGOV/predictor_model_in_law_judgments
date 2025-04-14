###
import sys
import os
# Obtém o diretório atual do script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Adiciona o diretório base do projeto ao caminho de busca do Python
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
###
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from src.training import split_train_test, test_model, stratify
import joblib  # Para salvar o modelo

def main():
    # Load the dataset
    data = pd.read_csv('data/main.csv')
    
    # Split the data into features and target variable
    X = data.drop('Dano-Moral', axis=1)
    y = data['Dano-Moral']
    y_bin = None #stratify(y, N=14)

    # Treinar/testar
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, y_bin=y_bin)


    # Modelo base do AdaBoost, a random forest regressor
    base_model = RandomForestRegressor(n_estimators=330, min_samples_split=2,
                                       max_features=1.0, criterion='poisson',
                                       random_state=15, n_jobs=-1, warm_start=True)

    # Fit the base model
    base_model.fit(X_train, y_train)
    # Make predictions with the base model
    (bs_rmse, bs_mae, folds) = test_model(base_model, X_test, y_test)
    print(f'RMSE: {bs_rmse}')
    print(f'MAE: {bs_mae}')
    print(f'Folds: \n{folds}\n')
    # Save the base model
    joblib.dump(base_model, 'models_storage/RandomForest.pkl')


    adb = False
    if adb:
        # Initialize the AdaBoost Regressor
        model = AdaBoostRegressor(estimator=base_model, n_estimators=200,
                                loss='square', random_state=15,
                                learning_rate=0.000001)
        
        # Fit the model to the data
        model.fit(X_train, y_train)
        
        # Make predictions
        (rmse, mae, folds) = test_model(model, X_test, y_test, CV=14)
        print(f'RMSE: {rmse}')
        print(f'MAE: {mae}')
        print(f'Folds: \n{folds}\n')
        # Save the model
        joblib.dump(model, 'models_storage/AdaBoost.pkl')

if __name__ == "__main__":
    main()
