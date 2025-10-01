# Adiciona o diretório base do projeto ao caminho de busca do Python
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
###
from src.formatation.data_preparation import split_data, balance_data
from src.formatation.data_formatation import load_data
from src.formatation.visualization import export_tree_to_graphviz
from src.util.param_grids import get_model_instance
from src.util.parameters import FILE_PATH, MODEL_NAME, TEST_SIZE
from src.util.parameters import BALANCE_STRATEGY, RANDOM_STATE
from src.shap_custom import explain_global, get_values
from src.training import train_model, test_model, save_model

# Função principal para executar o pipeline
def main(model, model_name: str):
    # Load the dataset
    X, y = load_data(FILE_PATH)
    print("-> Dados carregados.")
    
    # Balance the data
    X_bal, y_bal, y_bin = balance_data(X, y, BALANCE_STRATEGY, RANDOM_STATE)
    print("-> Dados balanceados.")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, y_test_bin = split_data(X_bal, y_bal, TEST_SIZE, y_bin)
    print("-> Dados divididos em treino e teste.")

    # Fit the base model
    train_model(model, X_train, y_train)
    print(f"-> Modelo {model_name} treinado.")

    # Evaluate the base model
    print(f"-> Avaliando o modelo {model_name}...")

    # Make predictions with the base model
    (bs_rmse, bs_mae, bs_pmae, folds) = test_model(model, X_test, y_test, y_test_bin)

    # Log and save the model
    save_model(model, model_name, bs_rmse, bs_mae, bs_pmae, folds)
    print(f"-> Modelo {model_name} salvo.")
    
    # SHAP explainability
    N_features = X_test.shape[1]
    # Calculate SHAP values
    shap_values = get_values(model, X_train, X_test)
    # Global explanation
    explain_global(shap_values, N_features)
    print("-> Gráfico SHAP global salvo.")
    
    # Visualizations
    if model_name == "DecisionTree":
        # Plot the tree
        print("-> Exportando árvore de decisão para Graphviz...")
        export_tree_to_graphviz(model, X_train.columns)

if __name__ == "__main__":    
    model_name = MODEL_NAME
    # Select the model
    model = get_model_instance(model_name)
    main(model, model_name)
