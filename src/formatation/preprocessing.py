# Adiciona o diretório base do projeto ao caminho de busca do Python
import sys, os
this_path = os.path.dirname(os.path.abspath(__file__))
if not this_path in sys.path:
    sys.path.append(this_path)
this_path = os.path.dirname(this_path)
if not this_path in sys.path:
    sys.path.append(this_path)
import pandas as pd
# to avoid SettingWithCopyWarning
# pd.set_option('future.no_silent_downcasting', True)

from util.log_aux import append_to_data_log_list, update_data_log, log_file_preprocessing
from util.parameters import VAR_THRESHOLD, BALANCE_STRATEGY, RANDOM_STATE, SUPORTED_COLS
from util.parameters import CANCELAMENTO, TARGET, BIN_COL, ID_COL, N_FOLDS
import feature_formatation as ff
from feature_selection import feature_selection
from test_preparation import split_data, balance_data
from filtering import remove_outliers, separate_zeros

def feature_name_coherence(df: pd.DataFrame, suported_cols:list[str]) -> pd.DataFrame:
    """
    parameters:
        - df: dataframe que será alterado.
        - suported_cols: features/colunas nessa lista serão desconsideradas (isto é, as colunas em si serão removidas, perdendo-se a informação de quais entradas/linhas possuiam valores não 0 nessas features)
    Garantir a coerência dos nomes das colunas
    1. Renomeia dano_moral_individual para Target
    1.5. Verifica se a coluna Target existe
    2. Renomeia faixa_intervalo_atraso para intervalo_atraso
    2.5. Se intervalo_atraso já existir, remove-a antes de renomear faixa_intervalo_atraso
    3. Renomeia faixa_intervalo_extravio_temporario para intervalo_extravio_temporario
    3.5. Se intervalo_extravio_temporario já existir, remove-a antes de renomear faixa_intervalo_extravio_temporario
    4. Renomeia assistencia_cia_aerea para desamparo
    5. Renomeia cancelamento/alteracao_destino para cancelamento
    6. Renomeia condicoes_climaticas/fechamento_aeroporto para fechamento_aeroporto
    7. Remove colunas não reconhecidas
    7. Retorna o DataFrame com os nomes das colunas coerentes
    8. Loga todas as mudanças feitas no arquivo de log
    """
    # renames dano_moral_individual to Target
    if 'dano_moral_individual' in df.columns:
        log_file_preprocessing.write("Renomeando coluna dano_moral_individual para {}\n".format(TARGET))
        df = df.rename(columns={'dano_moral_individual': TARGET})
    # check if target column exists
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in DataFrame columns.")
    # renames faixa_intervalo_atraso to intervalo_atraso
    if 'faixa_intervalo_atraso' in df.columns:
        if 'intervalo_atraso' in df.columns:
            log_file_preprocessing.write("Coluna faixa_intervalo_atraso ja existe. Removendo intervalo_atraso.\n")
            df = df.drop(columns=['intervalo_atraso'])
        log_file_preprocessing.write("Renomeando coluna faixa_intervalo_atraso para intervalo_atraso\n")
        df.rename(columns={'faixa_intervalo_atraso': 'intervalo_atraso'}, inplace=True)
    # renames faixa_intervalo_extravio_temporario to intervalo_extravio_temporario
    if 'faixa_intervalo_extravio_temporario' in df.columns:
        if 'intervalo_extravio_temporario' in df.columns:
            log_file_preprocessing.write("Coluna faixa_intervalo_extravio_temporario ja existe. Removendo intervalo_extravio_temporario.\n")
            df = df.drop(columns=['intervalo_extravio_temporario'])
        log_file_preprocessing.write("Renomeando coluna faixa_intervalo_extravio_temporario para intervalo_extravio_temporario\n")
        df = df.rename(columns={'faixa_intervalo_extravio_temporario': 'intervalo_extravio_temporario'})
    # Inverting the values of assistencia_cia_aerea to make it a profactor
    if 'assistencia_cia_aerea' in df.columns:
        log_file_preprocessing.write("Renomeando coluna assistencia_cia_aerea para desamparo\n")
        df.rename(columns={'assistencia_cia_aerea': 'desamparo'}, inplace=True)
    # renaming some columns for coherence
    if 'cancelamento/alteracao_destino' in df.columns:
        log_file_preprocessing.write("Renomeando coluna cancelamento/alteracao_destino para cancelamento\n")
        df.rename(columns={'cancelamento/alteracao_destino': 'cancelamento'}, inplace=True)
    if 'condicoes_climaticas/fechamento_aeroporto' in df.columns:
        log_file_preprocessing.write("Renomeando coluna condicoes_climaticas/fechamento_aeroporto para fechamento_aeroporto\n")
        df.rename(columns={'condicoes_climaticas/fechamento_aeroporto': 'fechamento_aeroporto'}, inplace=True)

    remove_columns = [col for col in df.columns if col not in suported_cols]
    df = df.drop(columns=remove_columns)
    log_file_preprocessing.write(f"Removidas as colunas: {remove_columns}\n")
    append_to_data_log_list('Features Removidas', remove_columns)
    return df

def format_data(df: pd.DataFrame) -> pd.DataFrame:
    # Aplicar a função de reformatação aos valores float nas colunas
    for coluna in df.columns:
        ff.current_column = coluna  # setting the current column for logging
        ff.first_run = True  # indicating it's the first run for this column
        df[coluna] = df[coluna].apply(ff.FUNCTIONS[coluna])
    # combining intervalo_atraso, noshow and cancelamento into intervalo_atraso
    if 'cancelamento' in df.columns and ('intervalo_atraso' in df.columns):
        i = df.loc[df['intervalo_atraso'] == -1, 'cancelamento']
        if 'noshow' in df.columns:
            c = df.loc[(df['cancelamento'] == 1) | (df['noshow'] == 1), 'intervalo_atraso']
            drop_cols = ['cancelamento', 'noshow']
            # logging the change
            log_file_preprocessing.write("Combinando as colunas cancelamento, noshow e intervalo_atraso em intervalo_atraso\n")
            append_to_data_log_list("Alteracoes nas Features", f"cancelamento e noshow sao combinadas com intervalo_atraso, onde cancelamento=1 ou noshow=1 torna-se intervalo_atraso={CANCELAMENTO}")
        else:
            c = df.loc[df['cancelamento'] == 1, 'intervalo_atraso']
            drop_cols = ['cancelamento']
            # logging the change
            log_file_preprocessing.write("Combinando as colunas cancelamento e intervalo_atraso em intervalo_atraso\n")
            append_to_data_log_list("Alteracoes nas Features", f"cancelamento combinado com intervalo_atraso, onde cancelamento=1 torna-se intervalo_atraso={CANCELAMENTO}")

        ci = pd.concat([c, i]).index
        df.loc[ci, 'intervalo_atraso'] = CANCELAMENTO  # setting to the corresponding value
        df = df.drop(columns=drop_cols)
    # Inverting the values of desamparo to make it a profactor
    des_col = 'desamparo'
    if des_col in df.columns:
        log_file_preprocessing.write("Invertendo os valores de Assistência para torna-la um profator\n")
        df.loc[(df['intervalo_atraso'] == 0), 'desamparo'] = -1
        df[des_col] = df[des_col].replace(1, -1)
        df[des_col] = df[des_col].replace(0, 1)
        df[des_col] = df[des_col].replace(-1, 0)
        # logging the change
        append_to_data_log_list("Alteracoes nas Features", "assistencia_cia_aerea e invertida (0 torna-se 1, 1 e -1 tornam-se 0) para tornar-se um profactor (desamparo)")
    return df

def separate_features_labels_bins(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Separar features (X) dos labels (Y) e das Faixas (Y_bin)
    Baseado na constante TARGET definida em parameters.py
    E na constante BIN_COL, também lá definida
    Retorna features (X), labels (y) e bins (y_bin).
    """
    y = data[TARGET]
    if BIN_COL in data.columns:
        y_bin = data[BIN_COL]
        cols = [TARGET, BIN_COL, ID_COL]
    else:
        cols = [TARGET, ID_COL]
        y_bin = data[TARGET]
    X = data.drop(columns=cols)
    return X, y, y_bin

def load_data(csv_file: str, log_data_path:str,
              split:bool=True, balance:bool=True, label:bool=True) -> list[
    tuple[
        tuple[pd.DataFrame, pd.Series, pd.Series],
        tuple[pd.DataFrame, pd.Series, pd.Series]
        ]
    ]:
    """
    Carregar e preparar os dados do arquivo CSV;
    Lê o arquivo de caminho `csv_file` usando pandas e salva os datasets intermediários em `log_data_path`
    Altera os nomes das colunas para evitar variações
    Formata os dados conforme necessário
    Separa os dados em procedentes e não procedentes
    Remove outliers
    Seleciona os melhores features e elimina entradas que dependam dos features removidos
    Salva o arquivo principal formatado
    Se split=True
        Divide os dados em conjuntos de treino e teste
    Se balance=True
        Balanceia os dados
    Se label=True
        Separa features (X) e labels (y)
        Retorna list[(X_train, y_train, bin_train), (X_test, y_test, bin_test)]
        (se split=False, X_train=X_test)
    Se label=False
        Não divide em X, y e bin; retorna só o dataset com o target e faixas
    """
    steps = 0
    # Ler o arquivo CSV usando pandas
    data = pd.read_csv(csv_file)
    log_file_preprocessing.write(f"Dados carregados: {data.shape}\n")
    log_file_preprocessing.write(f"Colunas originais: {data.columns.tolist()}\n---\n")
    update_data_log("Numero de Instancias Originais", data.shape[0])
    data.to_csv(f'{log_data_path}{steps}-Original_data.csv', index=False)
    steps += 1

    # Garantir a coerência dos nomes das colunas
    data = feature_name_coherence(data, SUPORTED_COLS)
    log_file_preprocessing.write(f"\n---\nColunas após coerencia de nomes: {data.columns.tolist()}\n")
    data.to_csv(f'{log_data_path}{steps}-Coherent_names.csv', index=False)
    steps += 1

    # Formata os features conforme necessário
    log_file_preprocessing.write("\n-----\nFormatando dados...\n")
    data = format_data(data)
    data.to_csv(f'{log_data_path}{steps}-Formatted_data.csv', index=False)
    steps += 1

    # Separa os dados em procedentes e não procedentes
    log_file_preprocessing.write("\n-----\nSeparando dados procedentes e nao procedentes...\n")
    ip, data = separate_zeros(data)
    data.to_csv(f'{log_data_path}{steps}-Procedentes.csv', index=False)
    ip.to_csv(f'{log_data_path}{steps}.5-Improcedentes.csv', index=False)
    steps += 1

    # Feature Selection
    log_file_preprocessing.write("\n-----\nSelecionando os melhores features\n")
    data, con = feature_selection(data, log_data_path, RANDOM_STATE, VAR_THRESHOLD)
    log_file_preprocessing.write(f"\n-----\nFeatures Selecionados: {data.columns.tolist()}\n")
    data.to_csv(f'{log_data_path}{steps}-Selected-Features.csv', index=False)
    con.to_csv(f'{log_data_path}{steps}.5-Removed-Features.csv', index=False)
    append_to_data_log_list("Features Usadas", list(data.columns[1:-1]))  # all except target
    steps += 1

     # Remove outliers
    log_file_preprocessing.write("\n-----\nRemovendo outliers...\n")
    data, df_out = remove_outliers(data)
    data.to_csv(f'{log_data_path}{steps}-No-Outliers.csv', index=False)
    df_out.to_csv(f'{log_data_path}{steps}.5-Outliers.csv', index=False)
    steps += 1

    # storing the main data
    prep_data_path = f'{log_data_path}{steps}-Preprocessed.csv'
    data.to_csv(prep_data_path, index=False)
    log_file_preprocessing.write(f"\n-----\nDados formatados salvos em: {prep_data_path}\n")
    steps += 1

    if split:
        # Split the data into training and testing sets
        cv_folds = split_data(data, N_FOLDS, RANDOM_STATE)
        log_file_preprocessing.write("-> Dados divididos em treino e teste.\n")
        split_folder = f"{log_data_path}{steps}-Splits"
        os.makedirs(split_folder, exist_ok=True)
        steps += 1
    else:
        cv_folds = [(data, data)]
        split_folder = "ERRO"

    if balance:
        bal_folder = f"{log_data_path}{steps}-Balanced_Data"
        os.makedirs(bal_folder, exist_ok=True)
    else:
        bal_folder = "ERRO"

    output = []

    for i, (train, test) in enumerate(cv_folds):
        if split:
            train.to_csv(f"{split_folder}/{i}_Train.csv", index=False)
            test.to_csv(f"{split_folder}/{i}_Test.csv", index=False)

        if balance:
            train = balance_data(train, BALANCE_STRATEGY, RANDOM_STATE)
            train.to_csv(f"{bal_folder}/Balanced_Train_{i}.csv", index=False)
            log_file_preprocessing.write(f"-> Dados {i} balanceados.\n")

        if label:
            # Separa features (X) e labels (y)
            log_file_preprocessing.write(f"\n-----\nSeparando features e labels do split {i} \n")
            output.append(
                (
                    separate_features_labels_bins(train),
                    separate_features_labels_bins(test)
                )
            )
        else:
            output.append(((train, train[TARGET], train[TARGET]), (test, test[TARGET], train[TARGET])))

    return output

from util.parameters import FILE_PATH, LOG_PATH, LOG_DATA_PATH
from feature_selection import feature_exploration
if __name__ == "__main__":
    # log1 = "_log-Full/data/"
    # os.makedirs(log1, exist_ok=True)
    # datasets1 = load_data(FILE_PATH, log1)
    # log2 = "_log-No-Split/data/"
    # os.makedirs(log2, exist_ok=True)
    # datasets2 = load_data(FILE_PATH, log2, split=False)
    # log3 = "_log-No-Balancement/data/"
    # os.makedirs(log3, exist_ok=True)
    # datasets3 = load_data(FILE_PATH, log3, balance=False)
    # log4 = "_log-No-Label-Split/data/"
    # os.makedirs(log4, exist_ok=True)
    # datasets4 = load_data(FILE_PATH, log4, label=False)

    datasets0 = load_data(FILE_PATH, LOG_DATA_PATH, split=False, balance=False, label=False)

    X, _, _ = datasets0[0][0]
    _ = feature_exploration(X, LOG_DATA_PATH, VAR_THRESHOLD, RANDOM_STATE, verbose=True)
