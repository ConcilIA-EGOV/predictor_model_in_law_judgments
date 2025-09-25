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
from util.parameters import DATA_VARS, DATA_PATH, FILE_PATH, TARGET
from formatation.variable_formatation import FUNCTIONS
from formatation.data_filtering import separate_zeros, trim_confactors, remove_outliers

def feature_formatation(df: pd.DataFrame) -> pd.DataFrame:
    # to avoid SettingWithCopyWarning
    pd.set_option('future.no_silent_downcasting', True)
    # Inverting the values of assistencia_cia_aerea to make it a profactor
    df.loc[(df['intervalo_atraso'] == 0), 'assistencia_cia_aerea'] = -1
    df['assistencia_cia_aerea'] = df['assistencia_cia_aerea'].replace(1, -1)
    df['assistencia_cia_aerea'] = df['assistencia_cia_aerea'].replace(0, 1)
    df['assistencia_cia_aerea'] = df['assistencia_cia_aerea'].replace(-1, 0)
    if 'extravio_temporario' in df.columns:
        df = df.drop(columns=['extravio_temporario'])
    if 'atraso' in df.columns:
        df = df.drop(columns=['atraso'])
    return df


def format_data(df: pd.DataFrame) -> pd.DataFrame:
    # Aplicar a função de reformatação aos valores float nas colunas
    for coluna in df.columns:
        print(f"Formatando coluna: {coluna}")
        df[coluna] = df[coluna].apply(FUNCTIONS[coluna])
    return df


def separate_features_labels(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separar features (X) dos labels (Y)
    """
    target_column = data.columns[-1]
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def trim_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas não relacionadas ao experimento
    E transfere a coluna alvo para a última posição
    """
    remove_columns = [col for col in df.columns if col not in DATA_VARS]
    print(f"Removendo colunas: {remove_columns}")
    df = df.drop(columns=remove_columns)
    # renames faixa_intervalo_atraso to intervalo_atraso
    if 'faixa_intervalo_atraso' in df.columns:
        if 'intervalo_atraso' in df.columns:
            print("Coluna intervalo_atraso já existe. Removendo-a.")
            df = df.drop(columns=['intervalo_atraso'])
        print("Renomeando coluna faixa_intervalo_atraso para intervalo_atraso")
        df.rename(columns={'faixa_intervalo_atraso': 'intervalo_atraso'}, inplace=True)
    # renames faixa_intervalo_extravio_temporario to intervalo_extravio_temporario
    if 'faixa_intervalo_extravio_temporario' in df.columns:
        if 'intervalo_extravio_temporario' in df.columns:
            print("Coluna intervalo_extravio_temporario já existe. Removendo-a.")
            df = df.drop(columns=['intervalo_extravio_temporario'])
        print("Renomeando coluna faixa_intervalo_extravio_temporario para intervalo_extravio_temporario")
        df = df.rename(columns={'faixa_intervalo_extravio_temporario': 'intervalo_extravio_temporario'})
    # renames dano_moral_individual to Target
    if 'dano_moral_individual' in df.columns:
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
        print(f"Arquivo {DATA_PATH} já existe. Carregando dados formatados...")
        data = pd.read_csv(DATA_PATH)
        # data = trim_columns(data)
        # data.to_csv(f'logs/trimmed_data.csv', index=False)
        X, y = separate_features_labels(data)
        print("Features e labels separados com sucesso!")
        print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
        return X, y
    steps = 1
    # Ler o arquivo CSV usando pandas
    data = pd.read_csv(csv_file)
    print("Dados carregados com sucesso!")
    print(f"Colunas originais: {data.columns.tolist()}")
    
    data = trim_columns(data)
    print(f"Colunas após trim: {data.columns.tolist()}")
    data.to_csv(f'logs/data/{steps}-trimmed_data.csv', index=False)
    steps += 1
    
    data = format_data(data)
    print("Dados formatados com sucesso!")
    data.to_csv(f'logs/data/{steps}-formatted_data.csv', index=False)
    steps += 1
    
    data = feature_formatation(data)
    print("Features formatadas com sucesso!")
    data.to_csv(f'logs/data/{steps}-feature_formatted_data.csv', index=False)
    steps += 1
    
    ip, p = separate_zeros(data)
    ip.to_csv('logs/IP/all.csv', index=False)
    p.to_csv('logs/P/all.csv', index=False)
    pro, con = trim_confactors(data)
    pro.to_csv('logs/all/Pro.csv', index=False)
    con.to_csv('logs/all/Con.csv', index=False)
    pro, con = trim_confactors(ip)
    pro.to_csv('logs/IP/Pro.csv', index=False)
    con.to_csv('logs/IP/Con.csv', index=False)
    data, con = trim_confactors(p)
    con.to_csv('logs/P/Con.csv', index=False)
    data.to_csv(f'logs/data/{steps}-filtered_data.csv', index=False)
    steps += 1
    print(f"Colunas após remover confactors: {data.columns.tolist()}")
    
     # Remove outliers
    data, out  = remove_outliers(data, TARGET)
    out.to_csv('logs/data/Outliers.csv', index=False)
    print("Outliers removidos com sucesso!")
    data.to_csv(DATA_PATH, index=False)
    
    X, y = separate_features_labels(data)
    print("Features e labels separados com sucesso!")
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    return X, y


if __name__ == "__main__":
    load_data(FILE_PATH)
