# Adiciona o diretório base do projeto ao caminho de busca do Python
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
###
import pandas as pd

from util.parameters import append_to_data_log_list, update_data_log
from util.parameters import LOG_PATH, LOG_DATA_PATH, FILE_PATH
from util.parameters import CANCELAMENTO, TARGET, log_file_preprocessing as log_file
import src.formatation.feature_formatation as ff
from formatation.feature_selection import trim_columns
from src.formatation.filtering import remove_outliers, separate_zeros

def feature_name_coherence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantir a coerência dos nomes das colunas
    1. Renomeia dano_moral_individual para Target
    1.5. Verifica se a coluna Target existe
    2. Renomeia faixa_intervalo_atraso para intervalo_atraso
    2.5. Se intervalo_atraso já existir, remove-a antes de renomear faixa_intervalo_atraso
    3. Renomeia faixa_intervalo_extravio_temporario para intervalo_extravio_temporario
    3.5. Se intervalo_extravio_temporario já existir, remove-a antes de renomear faixa_intervalo_extravio_temporario
    4. Renomeia assistencia_cia_aerea para desamparo
    5. Renomeia cancelamento/alteracao_destino para cancelamento
    6. Renomeia condicoes_climaticas/fechamento_aeroporto para fechamento_aeroporto
    7. Retorna o DataFrame com os nomes das colunas coerentes
    8. Loga todas as mudanças feitas no arquivo de log
    """
    # to avoid SettingWithCopyWarning
    pd.set_option('future.no_silent_downcasting', True)
    # renames dano_moral_individual to Target
    if 'dano_moral_individual' in df.columns:
        log_file.write("Renomeando coluna dano_moral_individual para {}\n".format(TARGET))
        df = df.rename(columns={'dano_moral_individual': TARGET})
    # check if target column exists
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in DataFrame columns.")
    # renames faixa_intervalo_atraso to intervalo_atraso
    if 'faixa_intervalo_atraso' in df.columns:
        if 'intervalo_atraso' in df.columns:
            log_file.write("Coluna faixa_intervalo_atraso já existe. Removendo intervalo_atraso.\n")
            df = df.drop(columns=['intervalo_atraso'])
        log_file.write("Renomeando coluna faixa_intervalo_atraso para intervalo_atraso\n")
        df.rename(columns={'faixa_intervalo_atraso': 'intervalo_atraso'}, inplace=True)
    # renames faixa_intervalo_extravio_temporario to intervalo_extravio_temporario
    if 'faixa_intervalo_extravio_temporario' in df.columns:
        if 'intervalo_extravio_temporario' in df.columns:
            log_file.write("Coluna faixa_intervalo_extravio_temporario já existe. Removendo intervalo_extravio_temporario.\n")
            df = df.drop(columns=['intervalo_extravio_temporario'])
        log_file.write("Renomeando coluna faixa_intervalo_extravio_temporario para intervalo_extravio_temporario\n")
        df = df.rename(columns={'faixa_intervalo_extravio_temporario': 'intervalo_extravio_temporario'})
    # Inverting the values of assistencia_cia_aerea to make it a profactor
    if 'assistencia_cia_aerea' in df.columns:
        log_file.write("Renomeando coluna assistencia_cia_aerea para desamparo\n")
        df.rename(columns={'assistencia_cia_aerea': 'desamparo'}, inplace=True)
    # renaming some columns for coherence
    if 'cancelamento/alteracao_destino' in df.columns:
        log_file.write("Renomeando coluna cancelamento/alteracao_destino para cancelamento\n")
        df.rename(columns={'cancelamento/alteracao_destino': 'cancelamento'}, inplace=True)
    if 'condicoes_climaticas/fechamento_aeroporto' in df.columns:
        log_file.write("Renomeando coluna condicoes_climaticas/fechamento_aeroporto para fechamento_aeroporto\n")
        df.rename(columns={'condicoes_climaticas/fechamento_aeroporto': 'fechamento_aeroporto'}, inplace=True)
    return df

def format_data(df: pd.DataFrame) -> pd.DataFrame:
    # Aplicar a função de reformatação aos valores float nas colunas
    for coluna in df.columns:
        ff.current_column = coluna  # setting the current column for logging
        ff.first_run = True  # indicating it's the first run for this column
        df[coluna] = df[coluna].apply(ff.FUNCTIONS[coluna])
    # to avoid SettingWithCopyWarning
    pd.set_option('future.no_silent_downcasting', True)
    # combining intervalo_atraso and cancelamento into intervalo_atraso
    if 'cancelamento' in df.columns and ('intervalo_atraso' in df.columns):
        log_file.write("Combinando as colunas cancelamento e intervalo_atraso em intervalo_atraso\n")
        canc = df['cancelamento'] == 1
        df.loc[canc, 'intervalo_atraso'] = CANCELAMENTO  # setting to the corresponding value
        df = df.drop(columns=['cancelamento'])
        # logging the change
        append_to_data_log_list("Alteracoes nas Features", f"cancelamento e combinado com intervalo_atraso, onde cancelamento=1 torna-se intervalo_atraso={CANCELAMENTO}")
    # Inverting the values of desamparo to make it a profactor
    des_col = 'desamparo'
    if des_col in df.columns:
        log_file.write("Invertendo os valores de Assistência para torná-la um profator\n")
        df.loc[(df['intervalo_atraso'] == 0), 'desamparo'] = -1
        df[des_col] = df[des_col].replace(1, -1)
        df[des_col] = df[des_col].replace(0, 1)
        df[des_col] = df[des_col].replace(-1, 0)
        # logging the change
        append_to_data_log_list("Alteracoes nas Features", "assistencia_cia_aerea e invertida (0 torna-se 1, 1 e -1 tornam-se 0) para tornar-se um profactor (desamparo)")
    return df

def separate_features_labels(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separar features (X) dos labels (Y)
    Baseado na constante TARGET definida em parameters.py
    Retorna features (X) e labels (y).
    """
    y = data[TARGET]
    X = data.drop(columns=[TARGET])
    log_file.write(f"Features shape: {X.shape}, Labels shape: {y.shape}\n")
    return X, y

def load_data(csv_file: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Carregar e preparar os dados do arquivo CSV;
    Lê o arquivo CSV usando pandas
    Remove colunas não relacionadas ao experimento
    Formata os dados conforme necessário
    Separa os dados em procedentes e não procedentes
    Remove outliers
    Salva o arquivo principal formatado
    Separa features (X) e labels (y)
    Retorna features (X) e labels (y).
    """
    steps = 0
    # Ler o arquivo CSV usando pandas
    data = pd.read_csv(csv_file)
    log_file.write(f"Dados carregados: {data.shape}\n")
    log_file.write(f"Colunas originais: {data.columns.tolist()}\n---\n")
    update_data_log("Numero de Instancias Originais", data.shape[0])
    data.to_csv(f'{LOG_PATH}data/{steps}-original_data.csv', index=False)
    steps += 1
    
    # Garantir a coerência dos nomes das colunas
    data = feature_name_coherence(data)
    log_file.write(f"\n---\nColunas após coerência de nomes: {data.columns.tolist()}\n")
    data.to_csv(f'{LOG_DATA_PATH}{steps}-coherent_names.csv', index=False)
    steps += 1

    # Remove colunas não relacionadas ao experimento
    log_file.write("\n-----\nRemovendo colunas não relacionadas...\n")
    data = trim_columns(data)
    log_file.write(f"\n-----\nColunas após remoção: {data.columns.tolist()}\n")
    data.to_csv(f'{LOG_DATA_PATH}{steps}-trimmed_data.csv', index=False)
    append_to_data_log_list("Features Usadas", list(data.columns[1:-1]))  # all except target
    steps += 1

    # Formata os features conforme necessário
    log_file.write("\n-----\nFormatando dados...\n")
    data = format_data(data)
    data.to_csv(f'{LOG_DATA_PATH}{steps}-formatted_data.csv', index=False)
    steps += 1

    # Separa os dados em procedentes e não procedentes
    log_file.write("\n-----\nSeparando dados procedentes e não procedentes...\n")
    ip, data = separate_zeros(data)
    ip.to_csv(f'{LOG_DATA_PATH}{steps}.5-Improcedentes.csv', index=False)
    data.to_csv(f'{LOG_DATA_PATH}{steps}-procedentes.csv', index=False)
    steps += 1
    
     # Remove outliers
    log_file.write("\n-----\nRemovendo outliers...\n")
    data = remove_outliers(data)

    # storing the main data
    prep_data_path = f'{LOG_DATA_PATH}{steps}-Preprocessed.csv'
    data.to_csv(prep_data_path, index=False)
    steps += 1
    log_file.write(f"\n-----\nDados principais salvos em {prep_data_path}\n")

    # Separa features (X) e labels (y)
    log_file.write("\n-----\nSeparando features e labels...\n")
    X, y = separate_features_labels(data)
    return X, y


if __name__ == "__main__":
    load_data(FILE_PATH)
