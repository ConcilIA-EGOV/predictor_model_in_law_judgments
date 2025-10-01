import pandas as pd

from util.parameters import log_file, append_to_data_log_list, update_data_log
from util.parameters import OUTLIERS_MIN_QUANTILE, OUTLIERS_MAX_QUANTILE, TARGET

def separate_zeros(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ip = df[(df['Dano-Moral'] == 0)]
    p = df[(df['Dano-Moral'] > 0)]
    log_file.write(f"Separando {ip.shape
                   } instâncias de Dano-Moral = 0 e {p.shape
                   } instâncias de Dano-Moral > 0.\n")
    append_to_data_log_list("Alteracoes nas Features", f"Removidas {ip.shape[0]} instancias com Dano-Moral = 0")
    return ip, p

def trim_confactors(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    conf1 = 'culpa_exclusiva_consumidor'
    conf2 = 'fechamento_aeroporto'
    if conf1 not in df.columns:
        log_file.write(f"Coluna confator {conf1} já removida.\n")
        conf1 = conf2
    if conf2 not in df.columns:
        log_file.write(f"Coluna confator {conf2} já removida.\n")
        if conf2 == conf1:
            # create an empty DataFrame with the same columns as df
            return df, pd.DataFrame(columns=df.columns)
        conf2 = conf1
    pro = df[(df[conf1] == 0) & (df[conf2] == 0)]
    con = df[(df[conf1] == 1) | (df[conf2] == 1)]
    append_to_data_log_list('Features Removidas', conf1)
    append_to_data_log_list('Features Removidas', conf2)
    append_to_data_log_list('Alteracoes nas Features', f"Removidas {con.shape[0]} instâncias que continham confactors: {conf1}, {conf2}")
    pro = pro.drop(columns=[conf1, conf2])
    log_file.write(f"Removendo colunas de co-fatores: {conf1}, {conf2}.\n   --> Resultando em {pro.shape} instâncias sem confactors e {con.shape} instâncias com confactors.\n")
    return pro, con

def remove_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # remove outliers based on out_col and the quantiles
    q_low = df[TARGET].quantile(OUTLIERS_MIN_QUANTILE)
    q_hi  = df[TARGET].quantile(OUTLIERS_MAX_QUANTILE)
    df_main = df[(df[TARGET] < q_hi) & (df[TARGET] > q_low)]
    df_out = df[(df[TARGET] >= q_hi) | (df[TARGET] <= q_low)]
    log_file.write(f"Removendo outliers baseados na coluna {TARGET}:\n")
    log_file.write(f"   --> Limite inferior ({OUTLIERS_MIN_QUANTILE * 100}% quantil): {q_low}\n")
    log_file.write(f"   --> Limite superior ({OUTLIERS_MAX_QUANTILE * 100}% quantil): {q_hi}\n")
    log_file.write(f"   --> Resultando em {df_main.shape
                        } instâncias principais e {df_out.shape
                        } instâncias outliers.\n")
    log_file.write(f"   --> Valores de {TARGET} entre {df_main[TARGET].min()} e {df_main[TARGET].max()}.\n")
    update_data_log("Numero de Outliers Removidos", f"{df_out.shape[0]}, com valores <= {q_low} ou >= {q_hi}")
    return df_main, df_out
