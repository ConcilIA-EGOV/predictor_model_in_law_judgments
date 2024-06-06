# caminhos para os arquivos de dados e modelos
FILE_PATH = "data/Teste.csv"
RESULT_FILE_PATH = "data/result.csv"
MODEL_PATH = "models_storage/"
MAIN_MODEL_FILE = MODEL_PATH + "main_model.joblib"
BEST_SCORE_STORAGE = MODEL_PATH + "best_scores.json"


DATA_VARS = [
    "Direito de arrependimento/Cancelamento pelo consumidor",
    "Descumprimento de oferta (assento)",
    "Extravio Definitivo",
    "Extravio Temporário",
    "Intervalo do Extravio",
    "Violação (furto, avaria)",
    "Cancelamento (sem realocação)/Alteração de destino",
    "Atraso (com realocação)",
    "Intervalo do Atraso",
    "Culpa Exclusiva do Consumidor",
    "Condições Climáticas Desfavoráveis/Fechamento Aeroporto",
    "No Show",
    "Overbooking",
    "Assistência da Cia Aérea",
    "Hipervulnerável (idoso/criança/pcd)",
    "Dano Moral"
]
USE_RANGES = True
##############
# Parâmetros #
##############

# Número de épocas
NUM_EPOCHS = 100
# Número de folds para a validação cruzada
CV = 5
# Tamanho do conjunto de teste
TEST_SIZE = 0.3
# Semente aleatória
RANDOM_STATE = 42
# parâmetros do modelo

# pytorch parameters
"""
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
