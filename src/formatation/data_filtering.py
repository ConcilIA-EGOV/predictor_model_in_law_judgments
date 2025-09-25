import pandas as pd

def separate_zeros(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ip = df[(df['Dano-Moral'] == 0)]
    p = df[(df['Dano-Moral'] > 0)]
    return ip, p

def trim_confactors(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    conf1 = 'culpa_exclusiva_consumidor'
    conf2 = 'fechamento_aeroporto'
    if conf1 not in df.columns:
        conf1 = conf2
    if conf2 not in df.columns:
        if conf2 == conf1:
            print("Colunas de co-fatores já removidas.")
            # create an empty DataFrame with the same columns as df
            return df, pd.DataFrame(columns=df.columns)
        conf2 = conf1
    pro = df[(df[conf1] == 0) & (df[conf2] == 0)]
    con = df[(df[conf1] == 1) | (df[conf2] == 1)]
    pro = pro.drop(columns=[conf1, conf2])
    return pro, con

def remove_outliers(df: pd.DataFrame, out_col:str) -> tuple[pd.DataFrame, pd.DataFrame]:    
    # remove outliers based on out_col and the quantile
    q_low = df[out_col].quantile(0.01)
    q_hi  = df[out_col].quantile(0.99)
    df_main = df[(df[out_col] < q_hi) & (df[out_col] > q_low)]
    df_out = df[(df[out_col] > q_hi) | (df[out_col] < q_low)]
    return df_main, df_out
