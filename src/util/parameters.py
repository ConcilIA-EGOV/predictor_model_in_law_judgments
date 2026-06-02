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
# Minimum tolerated variance
VAR_THRESHOLD = 0.5
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

SUPORTED_COLS = [
    TARGET,
    BIN_COL,
    ID_COL,

    "intervalo_atraso",
    "cancelamento",
    "desamparo",
    'noshow',

    "intervalo_extravio_temporario",
    "violacao_furto_avaria",
    "extravio_definitivo",
    "hipervulneravel",
    "overbooking",

    "direito_de_arrependimento",
    "descumprimento_de_oferta",

    # 'culpa_exclusiva_consumidor',
    # 'fechamento_aeroporto',

    # 'extravio_temporario',
    # 'atraso',
]

DT_PARAMS = {
    'random_state': RANDOM_STATE,
    'splitter': 'best',
    'criterion': 'squared_error',
    'max_depth': None,
    'max_features': None,
    "max_leaf_nodes":  None,
    "min_samples_leaf":  1,
    'min_samples_split': 2,
    "min_weight_fraction_leaf":  0.0,
    "min_impurity_decrease":  0.0,
    "monotonic_cst":  None,
    "ccp_alpha":  0.0,
}

RF_PARAMS = {
    'random_state': RANDOM_STATE,
    'min_samples_split': 2,
    'criterion': 'squared_error',
    'n_estimators': 25,
    'max_features': None,
    'max_depth': None,
    'n_jobs': -1,
}

LR_PARAMS = {
    "fit_intercept": True,
    "copy_X": True,
    "tol": 1e-06,
    "n_jobs": -1,
    "positive": False
}

GB_PARAMS = {
    "loss": 'squared_error',
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 1.0,
    "criterion": 'friedman_mse',
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_depth": None,
    "min_impurity_decrease": 0.0,
    "init": None,
    "random_state": RANDOM_STATE,
    "max_features": None,
    "verbose": 0,
    "max_leaf_nodes": None,
    "warm_start": False,
    "validation_fraction": 0.1,
    "n_iter_no_change": None,
    "tol": 0.0001,
    "ccp_alpha": 0.0,
}

NN_PARAMS = {
    "loss": 'squared_error',
    "hidden_layer_sizes": (100, ),
    "activation": 'relu', # {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
    "solver": 'lbfgs', # {‘lbfgs’, ‘sgd’, ‘adam’}
    "alpha": 0.0001,
    "batch_size": 32, #'auto',
    "learning_rate": 'adaptive', # {‘constant’, ‘invscaling’, ‘adaptive’}
    "learning_rate_init": 0.001,
    "power_t": 0.5,
    "max_iter": 1000,
    "shuffle": True,
    "random_state": RANDOM_STATE,
    "tol": 0.0001,
    "verbose": False,
    "warm_start": False,
    "momentum": 0.9,
    "nesterovs_momentum": True,
    "early_stopping": False,
    "validation_fraction": 0.1,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epsilon": 1e-08,
    "n_iter_no_change": 10,
    "max_fun": 15000
}

SVM_PARAMS = {
    "kernel":  'poly', # ‘poly’, ‘rbf’, ‘sigmoid’, ‘linear’, ‘precomputed’
    "degree":  3, # only for 'poly' kernel
    "gamma":  'scale', # ‘scale’ (1/(N*X.var())) or ‘auto’ (1/N)
    "coef0":  0.0, # for ‘poly’ and ‘sigmoid’
    "tol":  0.001,
    "C":  1.0, # Regularization parameter for rbf
    "epsilon":  0.1, # epsilon-tube within which no penalty is associated in the training loss function
    "shrinking":  True,
    "cache_size":  200,
    "verbose":  False,
    "max_iter":  -1,
}

NB_PARAMS = {
    "priors": None, # Prior probabilities of the classes. If specified, the priors are not adjusted according to the data.
    "var_smoothing": 1e-09, # Portion of the largest variance of all features that is added to variances for calculation stability.
}

MODELS_PARAMS = {
    'DecisionTree': DT_PARAMS,
    'RandomForest': RF_PARAMS,
    "GradientBoost": GB_PARAMS,
    "LinearRegression": LR_PARAMS,
    "NeuralNetork": NN_PARAMS,
    "SVM": SVM_PARAMS,
    "NaiveBayes": NB_PARAMS,
}

param_grid_DecisionTree = {
    'criterion': ['squared_error'],
    'splitter': ['best'],
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
    'max_features': [None, 'sqrt', 'log2', 1.0],
    'random_state': [RANDOM_STATE],
    'max_leaf_nodes': [None, 10, 30, 50],
    'min_impurity_decrease': [0.0, 0.01, 0.1],
    'ccp_alpha': [0.0, 0.01, 0.1, 0.5],
}
param_grid_RandForest = {
    'n_estimators': [200, 330, 400],
    'criterion': ['squared_error'],
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
    'max_features': [None, 'sqrt', 'log2', 1.0],
    'max_leaf_nodes': [None, 10, 30, 50],
    'min_impurity_decrease': [0.0, 0.01, 0.1],
    'bootstrap': [True, False],
    'oob_score': [True, False],
    'n_jobs': [-1],
    'random_state': [RANDOM_STATE],
    'verbose': [0],
    'warm_start': [False, True],
    'ccp_alpha': [0.0, 0.01, 0.1, 0.5],
    'max_samples': [None, 0.5, 0.75, 1.0]
}
param_grid_GBoost = {
    "loss": ['squared_error'],
    "learning_rate": [0.01, 0.1],
    "n_estimators": [100, 200, 300],
    "subsample": [0.5, 1.0],
    "criterion": ['friedman_mse'],
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
    'min_impurity_decrease': [0.0, 0.01, 0.1],
    "init": [None, 'zero'],
    'max_features': [None, 'sqrt', 'log2', 1.0],
    'random_state': [RANDOM_STATE],
    'max_leaf_nodes': [None, 10, 30, 50],
    'verbose': [0],
    'warm_start': [False, True],
    "tol": [0.01, 0.0001],
    'ccp_alpha': [0.0, 0.01, 0.1, 0.5],

}
param_grid_LinearReg = {
    "fit_intercept": [True],
    "copy_X": [True],
    "tol": [1e-03, 1e-05, 1e-6],
    "n_jobs": [-1],
    "positive": [False, True],

}
param_grid_NN = {
    "loss": ['squared_error'],
    "hidden_layer_sizes": [(50, ), (100, ), (200, ) ],
    "activation": ['relu', 'identity', 'tanh', 'logistic'],
    "solver": ['lbfgs', 'sgd', 'adam'],
    "alpha": [0.1, 0.01, 0.0001],
    "batch_size": ['auto', 32],
    "learning_rate": ['invscaling', 'adaptive'],
    "learning_rate_init": [0.1, 0.001],
    "power_t": [0.1, 0.5, 1.0],
    "max_iter": [200, 1000, 10000],
    "shuffle": [True],
    "random_state": [RANDOM_STATE],
    "tol": [0.01, 0.0001],
    "verbose": [False],
    "momentum": [0.1, 0.9],
    "early_stopping": [True, False],

}
param_grid_SVM = {
    "kernel":  ['poly', 'rbf', 'sigmoid', 'linear'],
    "degree":  [2, 3, 5],
    "gamma":  ['scale', 'auto'],
    "coef0":  [0.0],
    "tol":  [0.0001, 0.01],
    "C":  [0.2, 0.5, 1.0],
    "epsilon":  [0.0, 0.01, 0.1, 0.5],
    "shrinking":  [True, False],
    "cache_size":  [200],
    "verbose":  [False],
    "max_iter":  [1000, -1],

}
param_grid_NB = {
    "priors": [None],
    "var_smoothing": [1e-5, 1e-09, 1e-10],
}

PARAM_GRIDS = {
    'RandomForest': param_grid_RandForest,
    'DecisionTree': param_grid_DecisionTree,
    "GradientBoost": param_grid_GBoost,
    "LinearRegression": param_grid_LinearReg,
    "NeuralNetork": param_grid_NN,
    "SVM": param_grid_SVM,
    "NaiveBayes": param_grid_NB,
}


# diretório para salvar os modelos treinados
OUT_PATH = f"_Models/"
BEST_MODEL_PATH = f"{OUT_PATH}_Best_Model.pkl"
MODELS_FOLDERS = {mn: f"{OUT_PATH}model_{mn}/" for mn in PARAM_GRIDS.keys()}
MODELS_FILES = {mn: f"{MODELS_FOLDERS[mn]}_Model.pkl" for mn in PARAM_GRIDS.keys()}

start_data_log = {
    "Numero de Instancias Originais": 0,
    "Features Removidas": [],
    "Features Eliminadas": [],
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