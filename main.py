# Adiciona o diretório base do projeto ao caminho de busca do Python
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
###
from src.formatation.preprocessing import load_data
from src.formatation.visualization import export_tree_to_graphviz
from src.util.parameters import FILE_PATH, MODELS, MODELS_FOLDERS
from src.shap_custom import explain_global, get_values
from src.training import train_model, test_model, save_model, get_model_instance

# Função principal para executar o pipeline
def main(models_names: list[str]):
    # Load the dataset
    X_train, X_test, y_train, y_test, y_test_bin = load_data(FILE_PATH)
    print("-> Dados carregados.")

    # Fit the base model
    for mn in models_names:
        # Select the model
        print("\n##################################\n")
        model = get_model_instance(mn)
        print(f"-> Usando o Modelo:\n   -> {model}")
        train_model(model, X_train, y_train)
        print(f"-> Modelo {mn} treinado.")

        # Evaluate the base mnmodel
        print(f"-> Avaliando o modelo {mn}...")

        # Make predictions with the base model
        (bs_rmse, bs_mae, bs_mape, bs_mape_x, folds) = test_model(model, X_test, y_test, y_test_bin)

        # Log and save the model
        save_model(model, mn, bs_rmse, bs_mae, bs_mape, bs_mape_x, folds)
        print(f"-> Modelo {mn} salvo.")

        # SHAP explainability
        N_features = X_test.shape[1]
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
