##############
# Parâmetros #
##############

# Número de características
INPUT_SIZE = 13
# Número de classes
OUTPUT_SIZE = 5
# Taxa de aprendizado
LR = 0.001
# Tamanho do lote
BATCH_SIZE = 32
# Número de épocas
NUM_EPOCHS = 100
# Tamanho do conjunto de teste
TEST_SIZE = 0.3
# Número de folds para a validação cruzada
CV = 5
# Semente aleatória
RANDOM_STATE = 42
# Loss function
LOSS = 'hinge'
# Número máximo de iterações
MAX_ITER = 1000
# Tolerância
TOL = 1e-3

# Definir o caminho para os arquivos CSV
FILE_PATH = "data/Teste.csv"
RESULT_FILE_PATH = "data/result.csv"
MODEL_PATH = "models_storage/"
SCIKIT_MODEL_FILE = MODEL_PATH + "model.joblib"
BEST_SCORE_STORAGE = MODEL_PATH + "best_scores.json"
PYTORCH_MODEL_FILE = MODEL_PATH + "model.pth"
# Índice da coluna de resultados (última)
RESULTS_COLUMN = -1

# Definir os hiperparâmetros a serem ajustados
PARAM_GRID = {
            'loss': ['hinge', 'log', 'modified_huber',
                    'squared_hinge', 'perceptron',
                    'squared_loss', 'huber',
                    'epsilon_insensitive',
                    'squared_epsilon_insensitive'],
            'max_iter': [0, 1000, 10000],
            'tol': [1e-3, 1e-4, 1e-5]
}
