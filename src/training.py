from joblib import dump
import numpy as np
import pandas as pd

from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from util.parameters import MODEL_PATH

def train_model(model, X, y):
    """
    Treinar o modelo usando o conjunto de treino.
    """    
    model.fit(X, y)        
    return model


def test_model(model, X:pd.DataFrame, y:pd.Series, y_bin:pd.Series) -> tuple:
    '''
    Testar o modelo usando o conjunto de teste
    Retorna RMSE, MAE, P-MAE e resultados por faixa de valores    
    '''
    #score = classification_report(y, model.predict(X), output_dict=True)
    # score = cross_val_score(model, X, y, cv=FOLDS, n_jobs=-1)
    # Make predictions
    predictions = model.predict(X)
    
    # Calculate the RMSE, MAE and proportional MAE overall
    rmse_all = root_mean_squared_error(y, predictions)
    # Calculate the percentual MAE
    pmae_all = np.mean(np.abs((y - predictions) / y)) * 100
    mae_all = mean_absolute_error(y, predictions)

    # Calculate the RMSE, and MAE for each fold
    df = pd.DataFrame({"y_true": y, "y_pred": predictions, "fold": y_bin})
    resultados = []
    for faixa, grupo in df.groupby('fold', observed=False):
        if len(grupo) == 0:
            continue
        faixa = int(faixa) # type: ignore
        mae = mean_absolute_error(grupo['y_true'], grupo['y_pred'])
        rmse = root_mean_squared_error(grupo['y_true'], grupo['y_pred'])
        pmae = np.mean(np.abs((grupo['y_true'] - grupo['y_pred']) / grupo['y_true'])) * 100
        resultados.append(f"""Faixa {
            faixa + 1}:\n\t\tMAE: {round(mae,2
            )}\n\t\tRMSE: {round(rmse,2
            )}\n\t\tP-MAE: {round(pmae,2
            )}\n\t\tN Amostras: {len(grupo
            )}""")
    return (rmse_all, mae_all, pmae_all, resultados)


def save_model(model, model_name, bs_rmse, bs_mae, bs_pmae, folds):
    """
    Salvar o modelo treinado em um arquivo e os logs de performance em outro
    """    
    log_file = open(f'{MODEL_PATH}/{model_name}-log.txt', 'w')
    log_file.write(f'RMSE: {bs_rmse}\n')
    print(f'RMSE: {bs_rmse}')
    log_file.write(f'MAE: {bs_mae}\n')
    print(f'MAE:  {bs_mae}')
    log_file.write(f'PMAE: {bs_pmae:.2f}%\n')
    # prints the PMAE with 2 decimal places
    print(f'PMAE: {bs_pmae:.2f}%')
    log_file.write('\nPor Faixa:\n')
    [log_file.write(f'\t{folds[i]}\n') for i in range(len(folds))]
    log_file.close()
    # Save the base model
    dump(model, f'{MODEL_PATH}/{model_name}.pkl')
