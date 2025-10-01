from sklearn.tree import plot_tree, export_graphviz
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import graphviz

from util.parameters import MODEL_PATH

def plot_decision_tree(model, feature_names):
    plt.figure(figsize=(20,10))
    plot_tree(model, feature_names=feature_names, filled=True,
              rounded=True, max_depth=3, fontsize=10)
    plt.title("Árvore de Decisão - Explicação")
    plt.savefig(f"{MODEL_PATH}DecisionTree.png", dpi=300)


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
    graph.render(f"{MODEL_PATH}arvore_decisao", format="pdf", cleanup=True)


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
    plt.savefig(f"{MODEL_PATH}feature_importance.png")
    plt.close()


def associate_id_with_target(df1: pd.DataFrame, df2: pd.DataFrame, id_col: str, target_col: str) -> pd.DataFrame:
    """
    Associa os IDs do df1 com os valores alvo do df2
    Baseado em df2 também ter a coluna de IDs
    Retorna um novo DataFrame com as colunas de df1 mais a coluna alvo de df2
    """
    if id_col not in df1.columns or id_col not in df2.columns:
        raise ValueError(f"Coluna de ID '{id_col}' não encontrada em um dos DataFrames.")
    if target_col not in df2.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada no DataFrame df2.")
    print("df1 rows:", len(df1), "unique ids:", df1[id_col].nunique())
    print("df2 rows:", len(df2), "unique ids:", df2[id_col].nunique())

    # check duplicates in df2
    dups = df2[id_col].duplicated().sum()
    print("duplicate id_col in df2:", dups)

    # check per-key multiplicity
    counts = df2.groupby(id_col).size()
    print("max rows per id in df2:", counts.max())
    print("keys with >1 rows:", (counts>1).sum())
    
    # If df2 has multiple rows per id, reduce it to one row per id to avoid expanding df1 on a left merge.
    if dups > 0:
        print("Warning: df2 contains duplicate IDs; dropping duplicate rows and keeping the first occurrence for each ID.")
        df2_unique = df2.drop_duplicates(subset=id_col, keep='first')
        print("df2 reduced from", len(df2), "to", len(df2_unique), "rows after dropping duplicates on", id_col)
    else:
        df2_unique = df2

    merged_df = pd.merge(df1, df2_unique[[id_col, target_col]], on=id_col, how='left')

    # Fill missing target values (ids not found in df2) with 0
    merged_df[target_col] = merged_df[target_col].fillna(0)

    # If the original target in df2 was integer, convert to pandas nullable Int64 to avoid float upcast
    try:
        if pd.api.types.is_integer_dtype(df2_unique[target_col].dtype):
            merged_df[target_col] = merged_df[target_col].astype('Int64')
    except Exception:
        pass
    # Sanity check: merged should not have more rows than df1 after deduplication
    if len(merged_df) > len(df1):
        raise RuntimeError("Merged dataframe has more rows than df1 despite deduplication; inspect keys.")
    merged_df.to_csv("input/merged_with_target.csv", index=False)
    return merged_df

if __name__ == "__main__":
    df1 = pd.read_csv("input/ajudicada.csv")
    df2 = pd.read_csv("input/original.csv")
    merged = associate_id_with_target(df1, df2, 'sentenca', 'Dano-Moral')
    # print(merged.columns)
    print(merged.shape)
