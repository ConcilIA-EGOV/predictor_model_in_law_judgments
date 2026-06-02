# Adiciona o diretório base do projeto ao caminho de busca do Python
import sys, os
this_path = os.path.dirname(os.path.abspath(__file__))
if not this_path in sys.path:
    sys.path.append(this_path)
this_path = os.path.dirname(this_path)
if not this_path in sys.path:
    sys.path.append(this_path)

import pandas as pd
from joblib import dump

from src.formatation.preprocessing import load_data
from src.formatation.visualization import export_tree_to_graphviz, plot_distribution
from src.util.parameters import FILE_PATH, LOG_DATA_PATH, OUT_PATH
from src.util.parameters import MODELS_FOLDERS, MODELS_FILES, BEST_MODEL_PATH
from src.shap_custom import explain_global, get_values
from src.training import MODELS, test_model, save_model, grid_search

# Função principal para executar o pipeline
def main(models_names: list[str]):
    # Load the dataset
    datasets = load_data(FILE_PATH, LOG_DATA_PATH, balance=False)
    N_features = datasets[0][0][0].shape[1]
    print("-> Dados carregados.")
    results_cols = ["Model", "RMSE", "MAE", "MAPE", "TRAIN_SCORE", "AVG_RMSE", "AVG_MAE", "AVG_MAPE", "AVG_TRAIN_SCORE"]
    model_results = []
    best_score = 2.0**31
    best_model = None

    # Fit the base model
    for mn in models_names:
        # Select the model
        print("\n##################################\n")
        print(f"-> Usando o Modelo:\n   -> {mn}")
        model = None
        scores = []
        avg_rmse, avg_mae, avg_mape, avg_train_score = 0, 0, 0, 0
        for i, (train, test) in enumerate(datasets):
            X_train, y_train, _ = train
            X_test, y_test, bin_test = test
            model, model_params, train_score = grid_search(mn, X_train, y_train)

            # Make predictions with the base model
            (var_rmse, var_mae, var_mape, folds, errors) = test_model(model, X_test, y_test, bin_test)
            avg_rmse += var_rmse
            avg_mae += var_mae
            avg_mape += var_mape
            avg_train_score += train_score
            print(f'    {i+1} -> RMSE: {var_rmse:.2f} - MAE: {var_mae:.2f} - MAPE: {var_mape:.2f}% - GS SCORE: {train_score:.2f}')
            scores.append((var_rmse, var_mae, var_mape, train_score, folds, model, errors, model_params))

        # Evaluate the best model
        print(f"\n-> Avaliando o modelo medio para {mn}...")
        n_folds = len(datasets)
        avg_rmse /= n_folds
        avg_mae /= n_folds
        avg_mape /= n_folds
        avg_train_score /= n_folds
        print(f'    --> AVG RMSE:     {avg_rmse:.2f}')
        print(f'    --> AVG GS SCORE: {avg_train_score:.2f}')
        print(f'    --> AVG MAE:      {avg_mae:.2f}')
        print(f'    --> AVG MAPE:       {avg_mape:.2f}%')

        lowest = avg_rmse
        chosen = -1
        for i, inst in enumerate(scores):
            var_rmse = inst[0]
            if abs(avg_rmse - var_rmse) < lowest:
                lowest = abs(avg_rmse - var_rmse)
                chosen = i

        rmse, mae, mape, train_score, folds, model, errors, model_params = scores[chosen]
        save_model(model, mn, MODELS_FILES[mn], MODELS_FOLDERS[mn], rmse, mae, mape, train_score, folds, model_params)

        print('\nPerformance do Modelo Base:')
        print(f' - RMSE: {rmse:.2f}')
        print(f' - MAE:  {mae:.2f}')
        print(f' - MAPE:   {mape:.2f}%')

        if rmse < best_score:
            best_score = rmse
            best_model = model

        model_results.append([mn] + [round(num, 2) for num in [rmse, mae, mape, train_score, avg_rmse, avg_mae, avg_mape, avg_train_score]])

        # plot error distribution
        plot_distribution(errors, mn, MODELS_FOLDERS[mn])
        print(f"\nModelo {mn} {chosen+1} salvo.")

        # SHAP explainability
        train, test = datasets[chosen]
        X_train = train[0]
        X_test = test[0]
        if mn in ["DecisionTree", "RandomForest", "GradientBoost", "LinearRegression"]:
            # Calculate SHAP values
            shap_values = get_values(model, X_train, X_test)
            # Global explanation
            explain_global(shap_values, N_features, MODELS_FOLDERS[mn])
            print("-> Gráfico SHAP global salvo.")
            # Visualizations
            if mn == "DecisionTree":
                # Plot the tree
                print("-> Exportando árvore de decisão para Graphviz...")
                export_tree_to_graphviz(model, X_train.columns, MODELS_FOLDERS[mn])

    df = pd.DataFrame(model_results, columns=results_cols).sort_values(by="RMSE")
    df.to_csv(f"{OUT_PATH}_Final_Results.csv", index=False)
    dump(best_model, f"{BEST_MODEL_PATH}")

if __name__ == "__main__":
    main(list(MODELS.keys()))
