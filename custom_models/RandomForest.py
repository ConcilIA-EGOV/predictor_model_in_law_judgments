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
from util.parameters import MODEL_PATH

def main():
    model_name = "RandomForest"
    # Carregar os dados
    X_train, X_test, y_train, y_test, y_test_bin, _, _ = load_data()

    base_model = RandomForestRegressor(n_estimators=330, min_samples_split=2,
                                       max_features=1.0, criterion='poisson',
                                       random_state=15, n_jobs=-1, max_depth=15)

    # Fit the base model
    base_model.fit(X_train, y_train)
    # Make predictions with the base model
    (bs_rmse, bs_mae, folds) = test_model(base_model, X_test, y_test, y_test_bin)
    log_file = open(f'{MODEL_PATH}/{model_name}-log.txt', 'w')
    log_file.write(f'RMSE: {bs_rmse}\n')
    log_file.write(f'MAE: {bs_mae}\n')
    log_file.write('\nPor Faixa:\n')
    [log_file.write(f'\t{folds[i]}\n') for i in range(len(folds))]
    log_file.close()
    # Save the base model
    joblib.dump(base_model, f'{MODEL_PATH}/{model_name}.pkl')


if __name__ == "__main__":
    main()
