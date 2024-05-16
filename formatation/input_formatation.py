# Reading and writing files
import json
import os
import glob
import pandas as pd
import numpy as np


import pandas as pd

# Função para reformatar valores float
def reformat_float(value):
    return float(value.replace(',', '.'))

def format_csv(caminho_arquivo):
    # Ler o arquivo CSV usando pandas
    df = pd.read_csv(caminho_arquivo)

    # Aplicar a função de reformatação aos valores float nas colunas
    for coluna in df.select_dtypes(include=['object']).columns:
        df[coluna] = df[coluna].apply(reformat_float)

    # Salvar o DataFrame modificado de volta ao arquivo CSV
    df.to_csv(caminho_arquivo, index=False)



def load_data(csv_file):
    """
    Carregar os dados de um arquivo CSV
    """
    format_csv(csv_file)
    data = pd.read_csv(csv_file)
    return data


def separate_features_labels(data, target_column=-1):
    """
    Separar features (X) dos labels (Y)
    """
    X = data.iloc[:, :target_column].values  # Todas as colunas, até a alvo
    y = data.iloc[:, target_column].values   # A coluna alvo é o rótulo
    return X, y


def process_time_delay(feature=np.array([])):

    delay_minutes = list()

    for time_delay in feature:
        time_delay = time_delay.replace("- (superior a 4)", "00:00:00")
        time_delay = time_delay.replace("-", "00:00:00")
        splits = time_delay.split(":")

        seconds = float(splits[-1].strip()) / 60
        minutes = float(splits[-2].strip())
        hours = float(splits[-3].strip()) * 60

        delay_minutes.append(hours + minutes + seconds)

    return np.array(delay_minutes)


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
