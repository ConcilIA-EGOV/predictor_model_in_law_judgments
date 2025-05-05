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
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_graphviz
import matplotlib.pyplot as plt
import graphviz
from src.training import test_model
from src.file_op import load_data
from util.parameters import MODEL_PATH

def plot_decision_tree(model, feature_names):
    plt.figure(figsize=(20,10))
    plot_tree(model, feature_names=feature_names, filled=True,
              rounded=True, max_depth=3, fontsize=10)
    plt.title("Árvore de Decisão - Explicação")
    plt.savefig("DecisionTree.png", dpi=300)

def export_tree_to_graphviz(model, feature_names):
    # Exporta para o formato .dot
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=15  # ou mais
    )

    # Cria visualização
    graph = graphviz.Source(dot_data)
    graph.render("arvore_decisao", format="pdf", cleanup=True)

def main():
    model_name = "DecisionTree"
    # Load the dataset
    X_train, X_test, y_train, y_test, y_test_bin, _, _ = load_data()


    # Modelo base do AdaBoost, a random forest regressor
    base_model = DecisionTreeRegressor(min_samples_split=2, max_features=1.0,
                                       criterion='poisson', max_depth=15,
                                       random_state=15)

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
    # Plot the tree
    export_tree_to_graphviz(base_model, X_train.columns)
    # Save the base model
    joblib.dump(base_model, f'{MODEL_PATH}/{model_name}.pkl')


if __name__ == "__main__":
    main()
