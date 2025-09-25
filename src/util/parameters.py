# caminhos para os arquivos de dados e modelos
FILE_PATH = "input/original.csv"
DATA_PATH = "data/main.csv"
LOG_PATH = "logs/"
log_file = open(LOG_PATH + "log_data_formatation.txt", 'w')
# RESULT_FILE_PATH = "data/result.csv"
MODEL_NAME = "DecisionTree"  # 'DecisionTree' ou 'RandomForest'
# diretório para salvar os modelos treinados
MODEL_PATH = "models_storage/"
MAIN_MODEL_FILE = MODEL_PATH + MODEL_NAME + ".pkl"

TARGET = 'Dano-Moral'

FAIXAS_EXTRAVIO = [1, 24, 72, 168]
FAIXAS_ATRASO = [1, 4, 8, 12, 16, 24, 28]
FAIXAS_DANO = [1, 2000, 4000, 6000, 8000, 10000]


##############
# Parâmetros #
##############

# Número de folds para a validação cruzada
DM_FOLDS = 14
# Tamanho do conjunto de teste
TEST_SIZE = 0.2
# Random Seed
RANDOM_STATE = 42
# Estratégia de balanceamento
BALANCE_STRATEGY = 'not majority' # 'all', 'not majority', 'not minority', 'minority', 'auto'

PIPELINE_PARAMS = {
    'Número de folds': DM_FOLDS,
    'Tamanho percentual do conjunto de teste': TEST_SIZE,
    'Random Seed': RANDOM_STATE,
    'Estratégia de balanceamento': BALANCE_STRATEGY,
}

DATA_VARS = [
    'sentenca',
    'direito_de_arrependimento',
    'descumprimento_de_oferta',
    'extravio_definitivo',
    'intervalo_extravio_temporario',
    'faixa_intervalo_extravio_temporario',
    'violacao_furto_avaria',
    'cancelamento',
    'intervalo_atraso',
    'faixa_intervalo_atraso',
    "culpa_exclusiva_consumidor",
    "fechamento_aeroporto",
    'noshow',
    'overbooking',
    'assistencia_cia_aerea',
    'hipervulneravel',
    'Dano-Moral'
]

