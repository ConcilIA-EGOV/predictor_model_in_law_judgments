import numpy as np
import pandas as pd
from parameters import FAIXAS_EXTRAVIO, FAIXAS_ATRASO, FAIXAS_DANO
from parameters import DATA_VARS, CREATE_RANGES
# from parameters import SOURCE, TEST, ARQUIVO_DE_SAIDA
from file_operations import log_file
current_column = ""

def hour_to_float(value):
    splits = value.split(":")
    last = len(splits) - 1
    try:
        minutes = float(splits[-last].strip()) / 60
        hours = float(splits[-(last + 1)].strip())
    except:
        global current_column
        log_file.write(f"\nColuna: {current_column}: Valor não reconhecido: {value}\n")
        return 0
    f_value = hours + minutes
    return f_value

def format_string(value):
    if '$' in value:
        value = value.replace('R$ ', '')
    if ',' in value:
        value = value.replace(',', '.')
    if value[-3] == '.':
        if value[-3:] == '.00':
            value = value[:-3]
            if '.' in value:
                value = value.replace('.', '')
            return float(value)
    try:
        f_value = float(value)   
    except:
        global current_column
        log_file.write(f"\nColuna: {current_column}: Valor não reconhecido: {value}\n")
        return 0
    return f_value

def format_binario(value, anomaly=0, yes=1, no=0):
    if type(value) == float:
        if np.isnan(value):
            return anomaly
    if value in ['S', 's', 'Y', 'y', 'Sim', 'sim', 'SIM', 'YES', 'Yes', 'yes', '1']:
        return yes
    elif value in ['N', 'n', 'Não', 'não', 'NÃO', 'NO', 'No', 'no', '0']:
        return no
    else:
        return anomaly

def generate_range(value, interval_values=[]):
    if not CREATE_RANGES:
        return float(value)
    if type(value) != int and type(value) != float:
        global current_column
        log_file.write(f"\nColuna: {current_column}: Valor não reconhecido: {value}\n")
        return 0
    if value == -1:
        return 0
    for i, interval in enumerate(interval_values):
        if value < interval:
            return i
    return len(interval_values)

def format_intervalo(value, interval_values=[]):
    if type(value) == int:
        pass
    elif type(value) == float:
        if np.isnan(value):
            value = 0
    elif value in ['-', '', ' ', '--', 'null', 'NULL', 'Null']:
        value = 0
    elif ':' in value:
        value = hour_to_float(value)
    elif ',' in value or '.' in value:
        value = format_string(value)
    elif type(value) == str and value.isnumeric():
        value = float(value)
    else:
        try:
            value = float(value)
        except:
            global current_column
            log_file.write(f"\nColuna: {current_column}: Valor não reconhecido: {value}\n")
            value = 0
    return generate_range(value, interval_values)

FUNCTIONS = {
    'sentenca': lambda x: x,
    'direito_de_arrependimento': lambda x: format_binario(x),
    'descumprimento_de_oferta': lambda x: format_binario(x),
    'extravio_definitivo': lambda x: format_binario(x),
    'extravio_temporario': lambda x: format_binario(x),
    'intervalo_extravio_temporario': lambda x: format_intervalo(x, FAIXAS_EXTRAVIO),
    'faixa_intervalo_extravio_temporario': lambda x: int(x),
    'violacao_furto_avaria': lambda x: format_binario(x),
    'cancelamento/alteracao_destino': lambda x: format_binario(x),
    'atraso': lambda x: format_binario(x),
    'intervalo_atraso': lambda x: format_intervalo(x, FAIXAS_ATRASO),
    'faixa_intervalo_atraso': lambda x: int(x),
    'culpa_exclusiva_consumidor': lambda x: format_binario(x),
    'condicoes_climaticas/fechamento_aeroporto': lambda x: format_binario(x),
    'noshow': lambda x: format_binario(x),
    'overbooking': lambda x: format_binario(x),
    'assistencia_cia_aerea': lambda x: format_binario(x, -1),
    'hipervulneravel': lambda x: format_binario(x),
    'dano_moral_individual': lambda x: format_intervalo(x, FAIXAS_DANO),
    'faixa_dano_moral_individual': lambda x: int(x)
}

def trim_columns(df: pd.DataFrame):
    """
    Remove colunas não relacionadas ao experimento
    """
    remove_columns = [col for col in df.columns if col not in DATA_VARS]
    log_file.write(f"Colunas removidas: {remove_columns}\n")
    df = df.drop(columns=remove_columns)
    # log_file.write(f"Colunas restantes: {df.columns.to_list()}\n")
    return df

def format_data(csv_file:str):
    try:
        log_file.write(f"\n---\nArquivo: {csv_file}\n")
        df = pd.read_csv(csv_file)
    except Exception as e:
        log_file.write(str(e) + "\n")
        log_file.write("Arquivo não encontrado ou mal formatado\n")
        return None
    # Remover colunas não relacionadas ao experimento
    df = trim_columns(df)
    # Aplicar a função de reformatação aos valores float nas colunas
    for coluna in df.columns:
        global current_column
        current_column = coluna
        try:
            df[coluna] = df[coluna].apply(FUNCTIONS[coluna])
        except Exception as e:
            log_file.write(str(e) + '\n')
            log_file.write(f"Erro ao formatar a coluna: {coluna}\n")
    return df

def main(csv_file):
    df = pd.read_csv(csv_file)
    # converts the values on result_col from money to float
    result_col = df['dano_moral_individual'].apply(format_string)
    # format the data
    df = format_data(csv_file)
    # df = df.drop(columns=['sentenca'])
    res_col = 'Dano-Moral'
    df[res_col] = result_col
    df.to_csv('projecao/AAA_formated.csv', index=False)

if __name__ == "__main__":
    try:
        csv_file = 'projecao/original.csv'
        main(csv_file)
    except Exception as e:
        print(e)

