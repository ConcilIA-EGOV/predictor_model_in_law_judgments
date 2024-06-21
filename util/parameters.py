# caminhos para os arquivos de dados e modelos
FILE_PATH = "data/Teste.csv"
RESULT_FILE_PATH = "data/result.csv"
MODEL_PATH = "models_storage/"
MAIN_MODEL_FILE = MODEL_PATH + "main_model.joblib"
BEST_SCORE_STORAGE = MODEL_PATH + "best_scores.json"


# Variáveis categorizadas
DATA_VARS_RANGE = [
    'direito_de_arrependimento',
    'descumprimento_de_oferta',
    'extravio_definitivo',
    'faixa_intervalo_extravio_temporário',
    'violação_furto_avaria',
    'cancelamento/alteração_destino',
    'faixa_intervalo_atraso',
    'culpa_exclusiva_consumidor',
    'condições_climáticas/fechamento_aeroporto',
    'noshow',
    'overbooking',
    'assistência_cia_aérea',
    'hipervulnerável',
    'faixa_dano_moral_individual'
]
FAIXAS_EXTRAVIO = [24, 72, 168]
FAIXAS_ATRASO = [1, 4, 8, 12, 16, 24, 28]
FAIXAS_DANO = [1, 2000, 4000, 6000, 8000, 10000]

# Variáveis contínuas
DATA_VARS_CONTINUOUS = [
    'direito_de_arrependimento',
    'descumprimento_de_oferta',
    'extravio_definitivo',
    'intervalo_extravio_temporário',
    'violação_furto_avaria',
    'cancelamento/alteração_destino',
    'intervalo_atraso',
    'culpa_exclusiva_consumidor',
    'condições_climáticas/fechamento_aeroporto',
    'noshow',
    'overbooking',
    'assistência_cia_aérea',
    'hipervulnerável',
    'dano_moral_individual'
]

USE_RANGES = True
PREP = False
REFIT = False
# DATA_VARS = DATA_VARS_CONTINUOUS
DATA_VARS = DATA_VARS_RANGE

##############
# Parâmetros #
##############

# Número de épocas
NUM_EPOCHS = 100
# Número de folds para a validação cruzada
CV = 5
# Tamanho do conjunto de teste
TEST_SIZE = 0.3

"""
# pytorch parameters
# Número de características
INPUT_SIZE = 13
# Número de classes
OUTPUT_SIZE = 5
# Taxa de aprendizado
LR = 0.001
# Tamanho do lote
BATCH_SIZE = 32
PYTORCH_MODEL_FILE = MODEL_PATH + "model.pth"
"""
