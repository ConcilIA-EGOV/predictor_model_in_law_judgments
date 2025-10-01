# caminhos para os arquivos de dados e modelos
FILE_PATH = "input/original.csv"
DATA_PATH = "data/main.csv"
LOG_PATH = "model/logs/"
LOG_DATA_PATH = LOG_PATH + "data/"
PIPELINE_LOG_PATH = LOG_PATH + "log-pipeline.json"
log_file = open(LOG_PATH + "log_data_preparation.txt", 'a')

MODEL_NAME = "DecisionTree"  # 'DecisionTree' ou 'RandomForest'
# diretório para salvar os modelos treinados
MODEL_PATH = "model/"
MAIN_MODEL_FILE = MODEL_PATH + MODEL_NAME + ".pkl"

# Tamanho do conjunto de teste
TEST_SIZE = 0.2
# Random Seed
RANDOM_STATE = 42
# Outliers removal
OUTLIERS_MIN_QUANTILE = 0.05
OUTLIERS_MAX_QUANTILE = 0.95
# Estratégia de balanceamento
BALANCE_STRATEGY = 'not majority' # 'all', 'not majority', 'not minority', 'minority', 'auto'
# tamanho do intervalo de cada faixa para balanceamento
FOLD_SIZE = 1000

FAIXAS_EXTRAVIO = [1, 24, 72, 168]
FAIXAS_ATRASO = [1, 4, 8, 12, 16, 24, 28]
CANCELAMENTO = len(FAIXAS_ATRASO) + 1
TARGET = 'Dano-Moral'

import json, os
def get_data_log() -> dict:
    # first verify if the file exists
    if not os.path.exists(PIPELINE_LOG_PATH):
        write_log(start_data_log)  # create the file with the initial data
    # reads the dictionary from the json file
    output = {}
    with open(PIPELINE_LOG_PATH, 'r') as f:
        output = json.load(f)
    return output

def update_data_log(key: str, value) -> None:
    DATA_LOG = get_data_log()
    if key in DATA_LOG:
        DATA_LOG[key] = value
        write_log(DATA_LOG)
    else:
        raise KeyError(f"Chave '{key}' não encontrada em DATA_LOG.")

def append_to_data_log_list(key: str, value) -> None:
    DATA_LOG = get_data_log()
    if key in DATA_LOG and isinstance(DATA_LOG[key], list):
        if isinstance(value, list):
            DATA_LOG[key].extend(value)
        else:
            DATA_LOG[key].append(value)
        write_log(DATA_LOG)
    else:
        raise KeyError(f"Chave '{key}' não encontrada em DATA_LOG ou não é uma lista.")

def write_log(data_log: dict) -> None:
    with open(PIPELINE_LOG_PATH, 'w') as f:
        json.dump(data_log, f, indent=4)

start_data_log = {
    "Numero de Instancias Originais": 0,
    "Features Removidas": [],
    "Alteracoes nas Features": [],
    "Features Usadas": [],
    'Quantis de Outliers': f'min: {OUTLIERS_MIN_QUANTILE*100}% e max: {OUTLIERS_MAX_QUANTILE*100}%',
    "Numero de Outliers Removidos": "",
    "Instancias Usadas": 0,
    "Valor Minimo": 0,
    "Valor Maximo": 0,
    "Valor Medio": 0,
    'Random Seed': RANDOM_STATE,
    'Numero de Faixas de Valor': 0,
    "Tamanho de cada Faixa": 0,
    "Limite de tamanho para cada Faixa": FOLD_SIZE,
    "Bibliteca de Balanceamento": "",
    "Metodo de Balanceamento": "",
    'Estrategia de Balanceamento': BALANCE_STRATEGY,
    "Numero de Instancias Apos Balanceamento": 0,
    "Valor Medio Apos Balanceamento": 0,
    "Valor Minimo Apos Balanceamento": 0,
    "Valor Maximo Apos Balanceamento": 0,
    'Tamanho percentual do Conjunto de treino': f"{(1 - TEST_SIZE)*100}%",
    "Tamanho do Conjunto de Treino": 0,
    'Tamanho percentual do Conjunto de teste': f"{TEST_SIZE*100}%",
    "Tamanho do Conjunto de Teste": 0
}
