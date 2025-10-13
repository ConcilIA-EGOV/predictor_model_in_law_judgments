import pandas as pd

from util.parameters import LOG_DATA_PATH, append_to_data_log_list, log_file_preprocessing as log_file
from src.formatation.feature_formatation import FUNCTIONS

def trim_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas não relacionadas ao experimento
    """
    df = remove_confactors(df)
    remove_columns = [col for col in df.columns if col not in FUNCTIONS.keys()]
    log_file.write(f"Removendo colunas: {remove_columns}\n")
    append_to_data_log_list('Features Removidas', remove_columns)
    df = df.drop(columns=remove_columns)
    return df

def remove_confactors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove confatores do DataFrame
    1. Identifica as colunas de confatores: culpa_exclusiva_consumidor e fechamento_aeroporto
    2. Se ambas as colunas existirem, separa as instâncias em dois grupos:
       - Instâncias sem confatores (ambas as colunas iguais a 0)
       - Instâncias com confatores (pelo menos uma das colunas igual a 1)
    3. Registra o número de instâncias removidas e salva as instâncias com confatores em um arquivo CSV
    4. Remove as colunas de confatores do DataFrame
    5. Retorna o DataFrame sem as colunas de confatores
    6. Loga todas as mudanças feitas no arquivo de log
    7. Se uma das colunas de confatores já foi removida, utiliza a outra coluna para filtrar as instâncias
    8. Se ambas as colunas de confatores foram removidas, não faz nada
    """
    conf1 = 'culpa_exclusiva_consumidor'
    conf2 = 'fechamento_aeroporto'
    remove = [conf1, conf2]
    if conf1 in df.columns and conf2 in df.columns:
        pro = df[(df[conf1] == 0) & (df[conf2] == 0)]
        con = df[(df[conf1] == 1) | (df[conf2] == 1)]
        append_to_data_log_list('Alteracoes nas Features', f"Removidas {con.shape[0]} instâncias que continham confactors: {conf1}, {conf2}")
        log_file.write(f"Removendo colunas de co-fatores: {conf1}, {conf2}.\n   --> Resultando em {pro.shape} instâncias sem confactors e {con.shape} instâncias com confactors.\n")
    else:
        if all(conf not in df.columns for conf in remove):
            log_file.write("Ambas as colunas de confatores já foram removidas. Nenhuma ação necessária.\n")
            # an empty dataframe with the same columns as df
            con = pd.DataFrame(columns=df.columns)
            remove.clear()
        else:
            if conf1 in df.columns:
                log_file.write(f"Coluna confator {conf2} já removida.\n")
                remove.remove(conf2)
            elif conf2 in df.columns:
                log_file.write(f"Coluna confator {conf1} já removida.\n")
                remove.remove(conf1)
            pro = df[df[remove[0]] == 0]
            con = df[df[remove[0]] == 1]
            append_to_data_log_list('Alteracoes nas Features', f"Removidas {con.shape[0]} instancias que continham confactor: {remove[0]}")
            log_file.write(f"Removendo coluna de co-fator: {remove[0]}.\n   --> Resultando em {pro.shape} instâncias sem confator e {con.shape} instâncias com confator.\n")
    con.to_csv(f'{LOG_DATA_PATH}_Confactors.csv', index=False)
    df = df.drop(columns=remove)
    return df

