###
import sys
import os

# Obtém o diretório atual do script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Adiciona o diretório base do projeto ao caminho de busca do Python
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
###
import src.custom_models.DecisionTree as DT
import src.custom_models.RandomForest as RF
from src.formatation.data_formatation import load_data
from src.formatation.data_preparation import split_data, balance_data
from src.formatation.visualization import export_tree_to_graphviz
from src.training import train_model, test_model, save_model
from src.shap_custom import explain_global, get_values
from src.util.parameters import FILE_PATH, MODEL_NAME, TEST_SIZE, BALANCE_STRATEGY, RANDOM_STATE, DM_FOLDS

# Função principal para executar o pipeline
def main(model, model_name: str):
    # Load the dataset
    X, y = load_data(FILE_PATH)
    
    # Balance the data
    X_bal, y_bal, y_bin = balance_data(X, y, BALANCE_STRATEGY, RANDOM_STATE, DM_FOLDS)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, y_test_bin = split_data(X_bal, y_bal, TEST_SIZE, y_bin, DM_FOLDS)

    # Fit the base model
    train_model(model, X_train, y_train)
    print(f"Modelo {model_name} treinado.")
    # Evaluate the base model
    print(f"Avaliando o modelo {model_name}...")
    # Make predictions with the base model
    (bs_rmse, bs_mae, bs_pmae, folds) = test_model(model, X_test, y_test, y_test_bin)
    # Log and save the model
    save_model(model, model_name, bs_rmse, bs_mae, bs_pmae, folds)
    print(f"Modelo {model_name} salvo.")
    
    # SHAP explainability
    N_features = X_test.shape[1]
    # Calculate SHAP values
    shap_values = get_values(model, X_train, X_test)
    # Global explanation
    explain_global(shap_values, N_features)
    print("Gráfico SHAP global salvo.")
    
    # Visualizations
    if model_name == "DecisionTree":
        # Plot the tree
        print("Exportando árvore de decisão para Graphviz...")
        export_tree_to_graphviz(model, X_train.columns)

if __name__ == "__main__":    
    # Select the model
    model_name = MODEL_NAME
    model = None
    if model_name == "DecisionTree":
        model = DT.get_model()
    elif model_name == "RandomForest":
        model = RF.get_model()
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")
    main(model, model_name)
