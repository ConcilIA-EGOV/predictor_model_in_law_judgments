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
from util.parameters import DATA_VARS, FILE_PATH, USE_RANGES
from formatation.variable_formatation import FUNCTIONS


def format_data(df: pd.DataFrame):
    # Aplicar a função de reformatação aos valores float nas colunas
    for coluna in df.columns:
        df[coluna] = df[coluna].apply(FUNCTIONS[coluna])
    return df


def trim_columns(df: pd.DataFrame):
    """
    Remove colunas não relacionadas ao experimento
    """
    remove_columns = [col for col in df.columns if col not in DATA_VARS]
    df = df.drop(columns=remove_columns)
    results_columns = "dano_moral_individual"
    if USE_RANGES:
        results_columns = "faixa_dano_moral_individual"
    target_column = df.columns.get_loc(results_columns)
    if target_column != df.shape[1] - 1:
        # Mover a coluna alvo para a última posição
        tc = df.columns[target_column]
        x1 = list(df.columns[:target_column])
        x2 = list(df.columns[target_column + 1:])
        new_cols = x1 + x2 + [tc]
        df = df[new_cols]

    return df


def load_data(csv_file):
    """
    Carregar os dados de um arquivo CSV
    E formata-los
    """
    # Ler o arquivo CSV usando pandas
    data = pd.read_csv(csv_file)
    # Remover colunas não relacionadas ao experimento
    data = trim_columns(data)
    # Formatar os dados
    data = format_data(data)
    # Salvar o DataFrame modificado de volta ao arquivo CSV
    new_file = csv_file.replace(".csv", "__NEW.csv")
    data.to_csv(new_file, index=False)
    return data


def separate_features_labels(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separar features (X) dos labels (Y)
    """
    target_column = data.columns[-1]
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


if __name__ == "__main__":
    # load_data("data/dummy.csv")
    data = load_data(FILE_PATH)
    X, y = separate_features_labels(data)
