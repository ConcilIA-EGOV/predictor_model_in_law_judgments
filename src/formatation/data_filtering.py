import pandas as pd
from util.parameters import log_file

def separate_zeros(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ip = df[(df['Dano-Moral'] == 0)]
    p = df[(df['Dano-Moral'] > 0)]
    log_file.write(f"Separando {ip.shape} instâncias de Dano-Moral = 0 e {p.shape} instâncias de Dano-Moral > 0.\n")
    return ip, p

def trim_confactors(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    conf1 = 'culpa_exclusiva_consumidor'
    conf2 = 'fechamento_aeroporto'
    if conf1 not in df.columns:
        log_file.write(f"Coluna co-fator {conf1} já removidas.\n")
        conf1 = conf2
    if conf2 not in df.columns:
        if conf2 == conf1:
            log_file.write("Colunas de co-fatores já removidas.\n")
            # create an empty DataFrame with the same columns as df
            return df, pd.DataFrame(columns=df.columns)
        log_file.write(f"Coluna co-fator {conf2} já removidas.\n")
        conf2 = conf1
    pro = df[(df[conf1] == 0) & (df[conf2] == 0)]
    con = df[(df[conf1] == 1) | (df[conf2] == 1)]
    pro = pro.drop(columns=[conf1, conf2])
    log_file.write(f"Removendo colunas de co-fatores: {conf1}, {conf2}.\n   --> Resultando em {pro.shape} instâncias Pro e {con.shape} instâncias Con.\n")
    return pro, con

def remove_outliers(df: pd.DataFrame, out_col:str) -> tuple[pd.DataFrame, pd.DataFrame]:    
    # remove outliers based on out_col and the quantile
    q_low = df[out_col].quantile(0.01)
    q_hi  = df[out_col].quantile(0.99)
    df_main = df[(df[out_col] < q_hi) & (df[out_col] > q_low)]
    df_out = df[(df[out_col] > q_hi) | (df[out_col] < q_low)]
    log_file.write(f"Removendo outliers baseados na coluna {out_col}:\n")
    log_file.write(f"   --> Limite inferior (1% quantil): {q_low}\n")
    log_file.write(f"   --> Limite superior (99% quantil): {q_hi}\n")
    log_file.write(f"   --> Resultando em {df_main.shape} instâncias principais e {df_out.shape} instâncias outliers.\n")
    return df_main, df_out
