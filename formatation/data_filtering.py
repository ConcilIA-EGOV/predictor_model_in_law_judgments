import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_graphic_from_csv(data: pd.DataFrame,
        data_col:str,res_col:str,title:str): 
    """
    reads a csv file with 2 columns,
    and plot a boxplot or pairplot graphic comparing them 
    """
    if 'intervalo' in data_col and not 'faixa' in data_col:
        data.plot.scatter(x=data_col, y=res_col)
    else:
        data.boxplot(column=res_col, by=data_col)
    plt.xlabel(data_col)
    plt.ylabel(res_col)
    plt.title(title)


def plot_all_columns(df: pd.DataFrame, res_col: str, drop_cols: list):
    # df = df.drop(columns=['sentenca'])
    result_col = df[res_col].to_numpy()
    df = df.drop(columns=([res_col] + drop_cols))
    for col in df.columns:
        np_array = df[col].to_numpy()
        title = "{} x {}".format(col, res_col)
        # creates a new matrix with the ortogonal projection and the result
        new_matrix = np.column_stack((np_array, result_col))
        new_df = pd.DataFrame(new_matrix, columns=[col, res_col])
        plot_graphic_from_csv(new_df, col, res_col, title)
    plt.show()


def separate_zeros(df: pd.DataFrame, name:str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ip = df[(df['Dano-Moral'] == 0)]
    p = df[(df['Dano-Moral'] > 0)]
    ip.to_csv(f'projecao/IP/{name}.csv', index=False)
    p.to_csv(f'projecao/P/{name}.csv', index=False)
    return ip, p

def trim_confactors(df: pd.DataFrame, name:str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pro = df[(df['culpa_exclusiva_consumidor'] == 0) & (df['fechamento_aeroporto'] == 0)]
    con = df[(df['culpa_exclusiva_consumidor'] == 1) | (df['fechamento_aeroporto'] == 1)]
    pro = pro.drop(columns=['culpa_exclusiva_consumidor', 'fechamento_aeroporto'])
    pro.to_csv(f'projecao/{name}Pro.csv', index=False)
    con.to_csv(f'projecao/{name}Con.csv', index=False)
    return pro, con

def format(df: pd.DataFrame, name:str, out_col):
    # Inverting the values of assistencia_cia_aerea to make it a profactor
    df.loc[(df['faixa_intervalo_atraso'] == 0), 'assistencia_cia_aerea'] = -1
    df['assistencia_cia_aerea'] = df['assistencia_cia_aerea'].replace(1, -1)
    df['assistencia_cia_aerea'] = df['assistencia_cia_aerea'].replace(0, 1)
    df['assistencia_cia_aerea'] = df['assistencia_cia_aerea'].replace(-1, 0)
    df = df.drop(columns=['extravio_temporario', 'atraso', 'intervalo_extravio_temporario'])
    
    # remove outliers based on out_col and the quantile
    q_low = df[out_col].quantile(0.01)
    q_hi  = df[out_col].quantile(0.99)
    df_out = df[(df[out_col] > q_hi) | (df[out_col] < q_low)]
    df_main = df[(df[out_col] < q_hi) & (df[out_col] > q_low)]
    df_out.to_csv(f'projecao/{name}-Outliers.csv', index=False)
    df_main.to_csv(f'projecao/{name}.csv', index=False)


def main():
    reformat = True
    if reformat:
        df = pd.read_csv(f'projecao/original.csv')
        ip, p = separate_zeros(df, 'all')
        trim_confactors(df, 'all/')
        trim_confactors(ip, 'IP/')
        main = trim_confactors(p, 'P/')[0]
        format(main, 'main', 'Dano-Moral')
    main = pd.read_csv(f'projecao/main.csv')
    # plot_all_columns(main, 'Dano-Moral', ['sentenca'])
    

if __name__ == '__main__':
    os.makedirs('projecao', exist_ok=True)
    os.makedirs('projecao/IP', exist_ok=True)
    os.makedirs('projecao/P', exist_ok=True)
    os.makedirs('projecao/all', exist_ok=True)
    main()
