import json, os
from src.util.parameters import PIPELINE_LOG_PATH, LOG_PATH, LOG_DATA_PATH, start_data_log, MODELS_FOLDERS

os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(LOG_DATA_PATH, exist_ok=True)

for mf in MODELS_FOLDERS.values():
    os.makedirs(mf, exist_ok=True)


log_file_preprocessing = open(LOG_PATH + "preprocessing.txt", 'a')
log_file_preparation = open(LOG_PATH + "preparation.txt", 'a')
# log_file_filtering = open(LOG_PATH + "filtering.txt", 'a')


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