# Adiciona o diretório base do projeto ao caminho de busca do Python
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
###
import pandas as pd

from util.parameters import DATA_PATH, LOG_PATH, LOG_DATA_PATH, CANCELAMENTO
from util.parameters import FILE_PATH, TARGET, log_file, append_to_data_log_list, update_data_log
from formatation.variable_formatation import FUNCTIONS
import formatation.variable_formatation as vf
from formatation.data_filtering import separate_zeros, trim_confactors, remove_outliers


def feature_modification(df: pd.DataFrame) -> pd.DataFrame:
    # to avoid SettingWithCopyWarning
    pd.set_option('future.no_silent_downcasting', True)
    # combining intervalo_atraso and cancelamento into intervalo_atraso
    if 'cancelamento' in df.columns and 'intervalo_atraso' in df.columns:
        log_file.write("Combinando as colunas cancelamento e intervalo_atraso em intervalo_atraso\n")
        canc = df['cancelamento'] == 1
        df.loc[canc, 'intervalo_atraso'] = CANCELAMENTO  # setting to the corresponding value
        df = df.drop(columns=['cancelamento'])
        # logging the change
        append_to_data_log_list("Alteracoes nas Features", f"cancelamento é combinado com intervalo_atraso, onde cancelamento=1 torna-se intervalo_atraso={CANCELAMENTO}")
    # Inverting the values of assistencia_cia_aerea to make it a profactor
    if 'assistencia_cia_aerea' in df.columns:
        log_file.write("Invertendo os valores de assistencia_cia_aerea para torná-la um profator\n")
        df.loc[(df['intervalo_atraso'] == 0), 'assistencia_cia_aerea'] = -1
        df['assistencia_cia_aerea'] = df['assistencia_cia_aerea'].replace(1, -1)
        df['assistencia_cia_aerea'] = df['assistencia_cia_aerea'].replace(0, 1)
        df['assistencia_cia_aerea'] = df['assistencia_cia_aerea'].replace(-1, 0)
        df.rename(columns={'assistencia_cia_aerea': 'desamparo'}, inplace=True)
        # logging the change
        append_to_data_log_list("Alteracoes nas Features", "assistencia_cia_aerea é invertida (0 torna-se 1, 1 e -1 tornam-se 0) para tornar-se um profator (desamparo)")
    # renames dano_moral_individual to Target
    if 'dano_moral_individual' in df.columns:
        log_file.write("Renomeando coluna dano_moral_individual para {}\n".format(TARGET))
        df = df.rename(columns={'dano_moral_individual': TARGET})
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in DataFrame columns.")
    target_column = int(df.columns.get_loc(TARGET)) # type: ignore
    if target_column != df.shape[1] - 1:
        # Mover a coluna alvo para a última posição
        tc = df.columns[target_column]
        x1 = list(df.columns[:target_column])
        x2 = list(df.columns[target_column + 1:])
        new_cols = x1 + x2 + [tc]
        df = df[new_cols]
    if 'extravio_temporario' in df.columns:
        log_file.write("Removendo a coluna extravio_temporario\n")
         # removing extravio_temporario since intervalo_extravio_temporario is already present
        df = df.drop(columns=['extravio_temporario'])
        append_to_data_log_list('Features Removidas', 'extravio_temporario')
    if 'atraso' in df.columns:
        log_file.write("Removendo a coluna atraso\n")
        # removing atraso since intervalo_atraso is already present
        df = df.drop(columns=['atraso'])
        append_to_data_log_list('Features Removidas', 'atraso')
    return df


def format_data(df: pd.DataFrame) -> pd.DataFrame:
    # Aplicar a função de reformatação aos valores float nas colunas
    for coluna in df.columns:
        vf.current_column = coluna  # setting the current column for logging
        vf.first_run = True  # indicating it's the first run for this column
        df[coluna] = df[coluna].apply(FUNCTIONS[coluna])
    return df


def separate_features_labels(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separar features (X) dos labels (Y)
    """
    target_column = data.columns[-1]
    X = data.drop(columns=[target_column])
    y = data[target_column]
    log_file.write(f"Features shape: {X.shape}, Labels shape: {y.shape}\n")
    return X, y


def trim_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas não relacionadas ao experimento
    E transfere a coluna alvo para a última posição
    """
    remove_columns = [col for col in df.columns if col not in FUNCTIONS.keys()]
    log_file.write(f"Removendo colunas: {remove_columns}\n")
    append_to_data_log_list('Features Removidas', remove_columns)
    df = df.drop(columns=remove_columns)
    # renames faixa_intervalo_atraso to intervalo_atraso
    if 'faixa_intervalo_atraso' in df.columns:
        if 'intervalo_atraso' in df.columns:
            log_file.write("Coluna intervalo_atraso já existe. Removendo-a.\n")
            df = df.drop(columns=['intervalo_atraso'])
        log_file.write("Renomeando coluna faixa_intervalo_atraso para intervalo_atraso\n")
        df.rename(columns={'faixa_intervalo_atraso': 'intervalo_atraso'}, inplace=True)
    # renames faixa_intervalo_extravio_temporario to intervalo_extravio_temporario
    if 'faixa_intervalo_extravio_temporario' in df.columns:
        if 'intervalo_extravio_temporario' in df.columns:
            log_file.write("Coluna intervalo_extravio_temporario já existe. Removendo-a.\n")
            df = df.drop(columns=['intervalo_extravio_temporario'])
        log_file.write("Renomeando coluna faixa_intervalo_extravio_temporario para intervalo_extravio_temporario\n")
        df = df.rename(columns={'faixa_intervalo_extravio_temporario': 'intervalo_extravio_temporario'})
    return df


def load_data(csv_file: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Carregar e preparar os dados do arquivo CSV;
    1. Lê o arquivo CSV usando pandas
    2. Remove colunas não relacionadas ao experimento
    3. Formata os dados conforme necessário
    4. Altera as features conforme necessário
    5. Separa os dados em procedentes e não procedentes
    6. Remove Improcedentes com ou sem co-fatores
    7. Remove confatores dos procedentes
    8. Remove outliers
    9. Salva o arquivo principal formatado em data/main.csv
    10. Separa features (X) e labels (y)
    11. Retorna features (X) e labels (y).
    """
    # if there's already a DATA_PATH file, use it
    if os.path.exists(DATA_PATH):
        log_file.write(f"Arquivo {DATA_PATH} já existe. Carregando dados formatados...\n")
        data = pd.read_csv(DATA_PATH)
        log_file.write(f"Colunas carregadas: {data.columns.tolist()}\n")
        X, y = separate_features_labels(data)
        return X, y
    steps = 0
    # Ler o arquivo CSV usando pandas
    log_file.write("Carregando dados...\n")
    data = pd.read_csv(csv_file)
    log_file.write(f"Dados carregados: {data.shape}\n")
    log_file.write(f"Colunas originais: {data.columns.tolist()}\n---\n")
    data.to_csv(f'{LOG_PATH}data/{steps}-original_data.csv', index=False)
    update_data_log("Numero de Instancias Originais", data.shape[0])
    steps += 1

    # Remove colunas não relacionadas ao experimento
    log_file.write("\n-----\nRemovendo colunas não relacionadas...\n")
    data = trim_columns(data)
    log_file.write(f"\n-----\nColunas após trim: {data.columns.tolist()}\n")
    data.to_csv(f'{LOG_DATA_PATH}{steps}-trimmed_data.csv', index=False)
    steps += 1

    # Formata os features conforme necessário
    log_file.write("\n-----\nFormatando dados...\n")
    data = format_data(data)
    data.to_csv(f'{LOG_DATA_PATH}{steps}-formatted_data.csv', index=False)
    steps += 1

    # Formata features especiais (assistencia_cia_aerea e cancelamento)
    log_file.write("\n-----\nModificando features especiais...\n")
    data = feature_modification(data)
    data.to_csv(f'{LOG_DATA_PATH}{steps}-features_modified.csv', index=False)
    steps += 1

    # Remove confatores
    log_file.write("\n-----\nRemovendo confatores...\n")
    # Separa todos os confatores primeiro
    data, con = trim_confactors(data)
    con.to_csv(f'{LOG_DATA_PATH}{steps}.5-Confactors.csv', index=False)
    # saving the filtered data without Improcedentes or confactors
    data.to_csv(f'{LOG_DATA_PATH}{steps}-filtered_data.csv', index=False)
    steps += 1
    log_file.write(f"\n---\nColunas após remoção de confactors: {data.columns.tolist()}\n")
    append_to_data_log_list("Features Usadas", list(data.columns[1:-1]))  # all except target

    # Separa os dados em procedentes e não procedentes
    log_file.write("\n-----\nSeparando dados procedentes e não procedentes...\n")
    ip, data = separate_zeros(data)
    ip.to_csv(f'{LOG_DATA_PATH}{steps}.5-Improcedentes.csv', index=False)
    data.to_csv(f'{LOG_DATA_PATH}{steps}-Procedentes.csv', index=False)
    
    steps += 1
    
     # Remove outliers
    log_file.write("\n-----\nRemovendo outliers...\n")
    data, out  = remove_outliers(data)
    out.to_csv(f'{LOG_DATA_PATH}{steps}.5-Outliers.csv', index=False)
    data.to_csv(f'{LOG_DATA_PATH}{steps}-main.csv', index=False)
    steps += 1

    update_data_log("Valor Minimo", int(data[TARGET].min()))
    update_data_log("Valor Maximo", int(data[TARGET].max()))
    update_data_log("Valor Medio", round(data[TARGET].mean(), 2))
    update_data_log("Instancias Usadas", data.shape[0])

    # storing the main data
    data.to_csv(DATA_PATH, index=False)
    log_file.write(f"\n-----\nDados principais salvos em {DATA_PATH}\n")
    
    # Separa features (X) e labels (y)
    log_file.write("\n-----\nSeparando features e labels...\n")
    X, y = separate_features_labels(data)
    return X, y


if __name__ == "__main__":
    load_data(FILE_PATH)
