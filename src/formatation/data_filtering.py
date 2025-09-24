import pandas as pd

def separate_zeros(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ip = df[(df['Dano-Moral'] == 0)]
    p = df[(df['Dano-Moral'] > 0)]
    return ip, p

def trim_confactors(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if 'culpa_exclusiva_consumidor' in df.columns or 'fechamento_aeroporto' not in df.columns:
        print("Colunas de co-fatores já removidas.")
        # create an empty DataFrame with the same columns as df
        return df, pd.DataFrame(columns=df.columns)
    pro = df[(df['culpa_exclusiva_consumidor'] == 0) & (df['fechamento_aeroporto'] == 0)]
    con = df[(df['culpa_exclusiva_consumidor'] == 1) | (df['fechamento_aeroporto'] == 1)]
    pro = pro.drop(columns=['culpa_exclusiva_consumidor', 'fechamento_aeroporto'])
    return pro, con

def remove_outliers(df: pd.DataFrame, out_col:str) -> tuple[pd.DataFrame, pd.DataFrame]:    
    # remove outliers based on out_col and the quantile
    q_low = df[out_col].quantile(0.01)
    q_hi  = df[out_col].quantile(0.99)
    df_main = df[(df[out_col] < q_hi) & (df[out_col] > q_low)]
    df_out = df[(df[out_col] > q_hi) | (df[out_col] < q_low)]
    return df_main, df_out
