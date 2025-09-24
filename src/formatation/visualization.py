from sklearn.tree import plot_tree, export_graphviz
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import graphviz

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
    graph.render("models_storage/arvore_decisao", format="pdf", cleanup=True)



def plot_graphic_from_csv(data: pd.DataFrame,
        data_col:str,res_col:str,title:str): 
    """
    reads a csv file with 2 columns,
    and plot a boxplot or pairplot graphic comparing them 
    """
    if 'intervalo' in data_col and not 'faixa' in data_col:
        data.plot.scatter(x=data_col, y=res_col)
    else:
        data.boxplot(column=res_col, by=data_col)
    plt.xlabel(data_col)
    plt.ylabel(res_col)
    plt.title(title)


def plot_all_columns(df: pd.DataFrame, res_col: str, drop_cols: list):
    # df = df.drop(columns=['sentenca'])
    result_col = df[res_col].to_numpy()
    df = df.drop(columns=([res_col] + drop_cols))
    for col in df.columns:
        np_array = df[col].to_numpy()
        title = "{} x {}".format(col, res_col)
        # creates a new matrix with the ortogonal projection and the result
        new_matrix = np.column_stack((np_array, result_col))
        new_df = pd.DataFrame(new_matrix, columns=[col, res_col])
        plot_graphic_from_csv(new_df, col, res_col, title)
    plt.show()


def feature_importance(model, X):
    importances = model.feature_importances_
    features = X.columns
    sorted_idx = importances.argsort()

    plt.figure(figsize=(10,6))
    plt.barh(features[sorted_idx], importances[sorted_idx])
    plt.title('Importância das Features')
    plt.tight_layout()
    # since FigureCanvasAgg is non-interactive, and thus cannot be show
    plt.savefig("feature_importance.png")
    plt.close()
