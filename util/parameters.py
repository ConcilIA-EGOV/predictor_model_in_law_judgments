# caminhos para os arquivos de dados e modelos
FILE_PATH = "data/main.csv"
RESULT_FILE_PATH = "data/result.csv"
MODEL_PATH = "models_storage/"
MAIN_MODEL_FILE = MODEL_PATH + "main_model.joblib"
BEST_SCORE_STORAGE = MODEL_PATH + "best_scores.json"


FAIXAS_EXTRAVIO = [1, 24, 72, 168]
FAIXAS_ATRASO = [1, 4, 8, 12, 16, 24, 28]
FAIXAS_DANO = [1, 2000, 4000, 6000, 8000, 10000]

# Variáveis contínuas
DATA_VARS = [
    'direito_de_arrependimento',
    'descumprimento_de_oferta',
    'extravio_definitivo',
#    'intervalo_extravio_temporario',
    'faixa_intervalo_extravio_temporario',
    'violacao_furto_avaria',
    'cancelamento',
#    'intervalo_atraso',
    'faixa_intervalo_atraso',
    'noshow',
    'overbooking',
    'assistencia_cia_aerea',
    'hipervulneravel',
    'Dano-Moral'
]

# Pesos das variáveis
"""
direito_de_arrependimento = 2000.0
descumprimento_de_oferta = 2000.0
"""

BASE_WEIGHTS = []

TARGET = 'Dano-Moral'

USE_RANGES = False
PREP = False
REFIT = False

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
