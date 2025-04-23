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
import joblib  # Para salvar o modelo
from sklearn.ensemble import RandomForestRegressor
from src.training import test_model
from src.file_op import load_data

def main():
    # Carregar os dados
    X_train, X_test, y_train, y_test, y_test_bin, _, _ = load_data()

    base_model = RandomForestRegressor(n_estimators=330, min_samples_split=2,
                                       max_features=1.0, criterion='poisson',
                                       random_state=15, n_jobs=-1, max_depth=15)

    # Fit the base model
    base_model.fit(X_train, y_train)
    # Make predictions with the base model
    (bs_rmse, bs_mae, folds) = test_model(base_model, X_test, y_test, y_test_bin)
    print(f'RMSE: {bs_rmse}')
    print(f'MAE: {bs_mae}')
    print('\nPor Faixa:')
    [print(f'\t{folds[i]}') for i in range(len(folds))]
    # Save the base model
    joblib.dump(base_model, 'models_storage/RandomForest.pkl')


if __name__ == "__main__":
    main()
