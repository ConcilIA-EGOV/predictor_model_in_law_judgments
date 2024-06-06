# Reading and writing files
import json
import os
import glob
import pandas as pd
import numpy as np
###
import sys
import os

# Obtém o diretório atual do script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Adiciona o diretório base do projeto ao caminho de busca do Python
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
###
from util.parameters import DATA_VARS, FILE_PATH, USE_RANGES

def hour_to_float(value, interval_values=[]):
    # print(" --> Valor com horas:", value)
    splits = value.split(":")
    last = len(splits) - 1
    minutes = float(splits[-last].strip()) / 60
    hours = float(splits[-(last + 1)].strip())
    f_value = hours + minutes
    if interval_values:
        for i, interval in enumerate(interval_values):
            if f_value < interval:
                return i
    return f_value

def format_comma_strings(value, interval_values=[]):
    f_value = float(value.replace(',', '.'))
    if interval_values:
        for i, interval in enumerate(interval_values):
            if f_value < interval:
                return i
    return f_value

# Função para reformatar valores float
# TODO: função específica para cada coluna
def format_cell(value):
    if type(value) == int:
        return value
    if type(value) == float:
        if np.isnan(value):
            return 0
        return value
    if value in ['S', 's', 'Y', 'y', 'Sim', 'sim', 'SIM', 'YES', 'Yes', 'yes', '1']:
        return 1
    if value in ['N', 'n', 'Não', 'não', 'NÃO', 'NO', 'No', 'no', '0']:
        return 0
    if value in ['-', '', ' ']:
        return -1
    if ',' in value:
        # print(" --> Valor com vírgula:", value)
        return format_comma_strings(value)
    if ':' in value:
        return hour_to_float(value)
    else:
        print("Valor não reconhecido:", value)
        return value

def trim_columns(df: pd.DataFrame):
    """
    Remove colunas não relacionadas ao experimento
    """
    df = df.drop(columns="sentença")
    df = df.drop(columns="número_do_processo")
    df = df.drop(columns="julgamento")
    df = df.drop(columns="julgamento(2)")
    df = df.drop(columns="data_do_julgamento")
    df = df.drop(columns="julgador(a)")
    df = df.drop(columns="tipo_julgador(a)")
    if USE_RANGES:
        df = df.drop(columns="dano_moral_individual")
        df = df.drop(columns="intervalo_do_extravio_(dias)")
        df = df.drop(columns="intervalo_do_atraso_(horas:minutos)")
    else:
        df = df.drop(columns="faixa_dano_moral_individual")

    target_column = df.columns.get_loc("dano_moral_individual")
    if target_column != df.shape[1] - 1:
        # Mover a coluna alvo para a última posição
        tc = df.columns[target_column]
        x1 = list(df.columns[:target_column])
        x2 = list(df.columns[target_column + 1:])
        new_cols = x1 + x2 + [tc]
        df = df[new_cols]

    return df

def format_data(df: pd.DataFrame):
    # Aplicar a função de reformatação aos valores float nas colunas
    for coluna in df.columns:
        df[coluna] = df[coluna].apply(format_cell)
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


def separate_features_labels(data, target_column=-1):
    """
    Separar features (X) dos labels (Y)
    """
    # Todas as colunas, menos a coluna alvo
    
    X = data.iloc[:, :target_column].values
    # A coluna alvo é o rótulo
    y = data.iloc[:, target_column].values
    return X, y


def get_set_of_files_path(base_path):
    folders = [os.path.join(base_path, f) for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    if len(folders) > 0:
        return folders
    raise Exception("No experiments found")


def get_list_of_prompts(prompt_base_path, ext="txt"):
    list_files = glob.glob(os.path.join(prompt_base_path, "*." + ext))
    if len(list_files) > 0:
        return list_files
    raise Exception("No prompts found")


def list_raw_files_in_folder(path_to_folder, ext="txt"):
    # List files
    list_files_paths = glob.glob(os.path.join(path_to_folder, "*." + ext))
    if len(list_files_paths) > 0:
        return sorted(list_files_paths)
    raise Exception("No raw files found")


def read_txt_file(target_file, enc="utf-8"):
    return open(target_file, "r", encoding=enc).read()


def ensure_directory_exists(path):
    os.makedirs(path, exist_ok=True)


def store_output_results(list_outputs, output_path, base_folder_name, output_type):
    print("Saving results to " + output_path + " using " + output_type.upper() + " format")

    final_output_path = os.path.join(output_path, base_folder_name)
    ensure_directory_exists(final_output_path)

    if output_type.lower() not in ["csv", "log", "json", "txt"]:
        raise Exception("Output type not recognized")

    if output_type.lower() == "csv":
        df = pd.DataFrame(list_outputs)
        path_to_csv = os.path.join(final_output_path, "csv")
        ensure_directory_exists(path_to_csv)
        file_path = os.path.join(path_to_csv, "output.csv")

        df.to_csv(file_path, index=False)

    elif output_type.lower() == "log":

        raise Exception("Output type 'log' not implemented")
    elif output_type.lower() == "txt":
        # Create a folder just the raw files
        raw_files_folder = os.path.join(final_output_path, "txt")
        ensure_directory_exists(raw_files_folder)

        # Just the text file is saved
        for output in list(list_outputs):
            raw_file_name = output["raw_file_path"].split(os.sep)[-1]
            text = output["output_text"]

            file_path = os.path.join(raw_files_folder, raw_file_name)
            with open(file_path, "w") as fp:
                fp.write(text)
    elif output_type.lower() == "json":
        file_path = os.path.join(final_output_path, "csv", "output.csv")
        with open(file_path, 'w') as json_file:
            json.dump(list_outputs, json_file, indent=4)


# Local onde será salvo o arquivo csv com o resultado das requisições
def get_results_path(target_files_paths, prompt_path, PATH_BASE_OUTPUT):
    prompt_name = prompt_path.split(os.sep)[-1].replace(".txt", "")
    documents_folder_name = target_files_paths[0].split(os.sep)[-2]

    base_dir_name = "_".join(["experiment", prompt_name, documents_folder_name]).replace(" ", "-")

    dir_path = os.path.join(PATH_BASE_OUTPUT, base_dir_name, "sem_formatacao")
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    results_path = os.path.join(dir_path, "resultados_sem_formatacao.csv")

    return results_path

# Local onde será salvo o log com as responses do experimento
def get_log_path(target_files_paths, prompt_path, PATH_LOG):
    prompt_name = prompt_path.split(os.sep)[-1].replace(".txt", "")
    documents_folder_name = target_files_paths[0].split(os.sep)[-2]

    base_dir_name = "_".join(["experiment", prompt_name, documents_folder_name]).replace(" ", "-")

    dir_path = os.path.join(PATH_LOG, base_dir_name)
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    log_path = os.path.join(dir_path, "log")

    return log_path


def convert_csv_to_xlsx(origin_csv_path, xlsx_file="resultados_sem_formatacao.xlsx"):    
    # Verificando se o arquivo existe
    try:
        if not os.path.exists(origin_csv_path):
            print("O arquivo de resultados não foi encontrado")
            return
        
        # Setando diretório do arquivo xlsx
        xlsx_dir_path = os.sep.join(origin_csv_path.split(os.sep)[:-1])

        # Setando arquivo xlsx
        xlsx_file = os.path.join(xlsx_dir_path, xlsx_file)

        # Transformando CSV em DataFrame
        data_frame = pd.read_csv(origin_csv_path)

        # Transformando Data Frame em XLSX
        data_frame.to_excel(xlsx_file, index=False)
    except:
        print("Erro na função convert_csv_to_xlsx")


def get_formatted_results_path(csv_origin_path):
    # Verificando se o arquivo existe
    if not os.path.exists(csv_origin_path):
        print("O arquivo de resultados (sem formatação) não foi encontrado")
        return
    
    # Criação do diretório de arquivos formatados
    base_dir_path = os.sep.join(csv_origin_path.split(os.sep)[:-2])
    res_dir_path = os.path.join(base_dir_path, "formatados")
    ensure_directory_exists(res_dir_path)

    return res_dir_path


if __name__ == "__main__":
    load_data("data/dummy.csv")
