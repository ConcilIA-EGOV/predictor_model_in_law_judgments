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
from src.training import split_train_test, test_model, stratify, balance_data

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
    graph.render("arvore_decisao", format="pdf", cleanup=True)  # também pode ser PNG
    graph.view()

def main():
    # Load the dataset
    data = pd.read_csv('data/main.csv')
    
    # Split the data into features and target variable
    y = data['Dano-Moral']
    X = data.drop(['sentenca'], axis=1)
    X.to_csv('data/X.csv', index=False)
    X = X.drop('Dano-Moral', axis=1)
    y_bin = stratify(y, N=14)
    X_bal, y_bal, y_bin = balance_data(X, y, y_bin, strategy='not majority', random_state=15, N=14)

    # Treinar/testar
    X_train, X_test, y_train, y_test = split_train_test(X_bal, y_bal, test_size=0.2, y_bin=y_bin)


    # Modelo base do AdaBoost, a random forest regressor
    base_model = DecisionTreeRegressor(min_samples_split=2, max_features=1.0,
                                       criterion='poisson', max_depth=15,
                                       random_state=15)

    # Fit the base model
    base_model.fit(X_train, y_train)
    # Make predictions with the base model
    (bs_rmse, bs_mae, folds) = test_model(base_model, X_test, y_test)
    print(f'RMSE: {bs_rmse}')
    print(f'MAE: {bs_mae}')
    print('\nPor Faixa:')
    [print(f'\t{folds[i]}') for i in range(len(folds))]
    # Plot the tree
    export_tree_to_graphviz(base_model, X_train.columns)
    # Save the base model
    joblib.dump(base_model, 'models_storage/DecisionTree.pkl')


if __name__ == "__main__":
    main()
