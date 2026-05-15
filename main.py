# Adiciona o diretório base do projeto ao caminho de busca do Python
import sys, os
this_path = os.path.dirname(os.path.abspath(__file__))
if not this_path in sys.path:
    sys.path.append(this_path)
this_path = os.path.dirname(this_path)
if not this_path in sys.path:
    sys.path.append(this_path)

from src.formatation.preprocessing import load_data
from src.formatation.visualization import export_tree_to_graphviz, plot_distribution
from src.util.parameters import FILE_PATH, LOG_DATA_PATH
from src.util.parameters import MODELS, MODELS_FOLDERS, MODELS_FILES
from src.shap_custom import explain_global, get_values
from src.training import train_model, test_model, save_model, get_model_instance

# Função principal para executar o pipeline
def main(models_names: list[str]):
    # Load the dataset
    datasets = load_data(FILE_PATH, LOG_DATA_PATH)
    N_features = datasets[0][0][0].shape[1]
    print("-> Dados carregados.")

    # Fit the base model
    for mn in models_names:
        # Select the model
        print("\n##################################\n")
        print(f"-> Usando o Modelo:\n   -> {mn}")
        model = None
        scores = []
        avg_rmse, avg_mae, avg_mape, avg_mape_x = 0, 0, 0, 0
        for i, (train, test) in enumerate(datasets):
            model = get_model_instance(mn)
            X_train, y_train, _ = train
            X_test, y_test, bin_test = test
            train_model(model, X_train, y_train)

            # Make predictions with the base model
            (var_rmse, var_mae, var_mape, var_mape_x, folds, errors) = test_model(model, X_test, y_test, bin_test)
            avg_rmse += var_rmse
            avg_mae += var_mae
            avg_mape += var_mape
            avg_mape_x += var_mape_x
            print(f'    {i+1} -> RMSE: {var_rmse:.2f} - MAE: {var_mae:.2f} - MAPE: {var_mape:.2f}% - MAPE X: {var_mape_x:.2f}%')
            scores.append((var_rmse, var_mae, var_mape, var_mape_x, folds, model, errors))

        # Evaluate the best model
        print(f"\n-> Avaliando o modelo medio para {mn}...")
        n_folds = len(datasets)
        avg_rmse /= n_folds
        avg_mae /= n_folds
        avg_mape /= n_folds
        avg_mape_x /= n_folds
        print(f'    --> AVG RMSE:   {avg_rmse:.2f}')
        print(f'    --> AVG MAE:    {avg_mae:.2f}')
        print(f'    --> AVG MAPE:   {avg_mape:.2f}%')
        print(f'    --> AVG MAPE X: {avg_mape_x:.2f}%')

        lowest = avg_rmse
        chosen = -1
        for i, inst in enumerate(scores):
            var_rmse = inst[0]
            if abs(avg_rmse - var_rmse) < lowest:
                lowest = abs(avg_rmse - var_rmse)
                chosen = i

        rmse, mae, mape, mape_x, folds, model, errors = scores[chosen]
        save_model(model, mn, MODELS_FILES[mn], MODELS_FOLDERS[mn], rmse, mae, mape, mape_x, folds)

        # plot error distribution
        plot_distribution(errors, mn, MODELS_FOLDERS[mn])
        print(f"\nModelo {mn} {chosen+1} salvo.")

        # SHAP explainability
        train, test = datasets[chosen]
        X_train = train[0]
        X_test = test[0]
        if mn not in ["NeuralNetork", "NaiveBayes", "SVM"]:
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

if __name__ == "__main__":
    main(MODELS)
