import pandas as pd

from util.parameters import append_to_data_log_list, update_data_log, log_file_preprocessing as log_file
from util.parameters import OUTLIERS_MIN_QUANTILE, OUTLIERS_MAX_QUANTILE, TARGET, LOG_DATA_PATH


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
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

    df_out.to_csv(f'{LOG_DATA_PATH}_Outliers.csv', index=False)

    update_data_log("Valor Minimo", int(df_main[TARGET].min()))
    update_data_log("Valor Maximo", int(df_main[TARGET].max()))
    update_data_log("Valor Medio", round(df_main[TARGET].mean(), 2))
    update_data_log("Instancias Usadas", df_main.shape[0])
    return df_main

def separate_zeros(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ip = df[(df['Dano-Moral'] == 0)]
    p = df[(df['Dano-Moral'] > 0)]
    log_file.write(f"Separando {ip.shape
                   } instâncias de Dano-Moral = 0 e {p.shape
                   } instâncias de Dano-Moral > 0.\n")
    append_to_data_log_list("Alteracoes nas Features", f"Removidas {ip.shape[0]
                            } instancias com Dano-Moral = 0")
    return ip, p
