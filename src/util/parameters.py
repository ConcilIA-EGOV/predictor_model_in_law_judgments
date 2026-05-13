# caminhos para os arquivos de entrada e saída
FILE_PATH = "input/original.csv"
LOG_PATH = "_logs/"
LOG_DATA_PATH = LOG_PATH + "data/"
PIPELINE_LOG_PATH = LOG_PATH + "pipeline.json"

# Tamanho do conjunto de teste
N_FOLDS = 5
# Random Seed
RANDOM_STATE = 42
# Outliers removal
OUTLIERS_MIN_QUANTILE = 0.05
OUTLIERS_MAX_QUANTILE = 0.95
# Estratégia de balanceamento
BALANCE_STRATEGY = 'not majority' # 'all', 'not majority', 'not minority', 'minority', 'auto'
# tamanho do intervalo de cada faixa para balanceamento
FOLD_SIZE = 1000

# Definição das faixas de atraso e extravio em horas
FAIXAS_EXTRAVIO = [1, 24, 72, 168]
FAIXAS_ATRASO = [1, 4, 8, 12, 16, 24, 28]
# índice para cancelamento
CANCELAMENTO = -1
# Target variable
TARGET = 'Dano-Moral'
BIN_COL = "Faixa-de-Valores"
ID_COL = "sentenca"
# Features to Remove
REMOVED_FEATURES = [
    'culpa_exclusiva_consumidor',
    'fechamento_aeroporto',
    "direito_de_arrependimento",
    "descumprimento_de_oferta",
]

MODELS = [
    "DecisionTree",
    "RandomForest",
]
# diretório para salvar os modelos treinados
MODELS_FOLDERS = {mn: f"model_{mn}/" for mn in MODELS}
MODELS_FILES = {mn: f"{MODELS_FOLDERS[mn]}_Model.pkl" for mn in MODELS}

DT_PARAMS = {
    'random_state': RANDOM_STATE,
    'splitter': 'best',
    'criterion': 'poisson',
    'min_samples_split': 2,
    'max_features': 1.0,
    'max_depth': 10,
}


RF_PARAMS = {
    'random_state': RANDOM_STATE,
    'min_samples_split': 2,
    'criterion': 'poisson',
    'n_estimators': 330,
    'max_features': 1.0,
    'max_depth': 10,
    'n_jobs': -1,
}


MODELS_PARAMS = {
    'RandomForest': RF_PARAMS,
    'DecisionTree': DT_PARAMS
}

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
    "Numero de Instancias Pre-Balanceamento": 0,
    "Numero de Instancias Adicionadas pelo Balanceamento": 0,
    "Numero de Instancias Apos Balanceamento": 0,
    "Valor Medio Pre-Balanceamento": 0,
    "Valor Medio Pos-Balanceamento": 0,
    "Tamanho do Conjunto de Treino": 0,
    "Tamanho do Conjunto de Teste": 0
}

param_grid_RandForest = {
    'n_estimators': [200, 330, 400],
    'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
    'max_depth': [None, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
    'max_features': ['sqrt', 'log2', None, 1.0],
    'max_leaf_nodes': [None, 10, 20, 30],
    'min_impurity_decrease': [0.0, 0.1, 0.2],
    'bootstrap': [True, False],
    'oob_score': [True, False],
    'n_jobs': [-1],
    'random_state': [RANDOM_STATE],
    'verbose': [0],
    'warm_start': [False, True],
    'ccp_alpha': [0.0, 0.1, 0.2],
    'max_samples': [None, 0.5, 0.75, 1.0]
}


param_grid_DecisionTree = {
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter': 'best',
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
    'max_features': [None, 'sqrt', 'log2', 1.0],
    'random_state': [RANDOM_STATE],
    'max_leaf_nodes': [None, 10, 20, 30],
    'min_impurity_decrease': [0.0, 0.1, 0.2],
    'ccp_alpha': [0.0, 0.1, 0.2]
}


param_grids = {
    'RandomForest': param_grid_RandForest,
    'DecisionTree': param_grid_DecisionTree
}