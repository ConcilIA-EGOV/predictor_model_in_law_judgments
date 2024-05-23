###
import sys
import os

# Obtém o diretório atual do script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Adiciona o diretório base do projeto ao caminho de busca do Python
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
###
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from util.parameters import TEST_SIZE, RANDOM_STATE

def split_train_test(X, y):
    """
    Dividir em conjuntos de treino e teste e normalizar os dados
    """
    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y,
                                         test_size=TEST_SIZE,
                                         random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test

def preprocessing(X_, y_):
    """
    Preprocessar os dados
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(X_)
    y = y_

    return X, y
