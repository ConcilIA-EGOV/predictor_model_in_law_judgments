from sklearn.model_selection import cross_val_score
from joblib import dump, load
import json
import numpy as np
###
import sys
import os

# Obtém o diretório atual do script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Adiciona o diretório base do projeto ao caminho de busca do Python
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
###
from util.parameters import FILE_PATH, RESULTS_COLUMN, CV, BEST_SCORE_STORAGE
from util.parameters import NUM_EPOCHS, MAIN_MODEL_FILE
from formatation.input_formatation import load_data, separate_features_labels
from src.preprocessing import preprocessing

def train_model(model, X_train, y_train):
    """
    Treinar o modelo usando o conjunto de treino.
    Se o modelo já estiver treinado, o treinamento será incrementado.
    """    
    if hasattr(model, "partial_fit"):
        # Se o modelo suporta treinamento incremental, use partial_fit
        model.partial_fit(X_train, y_train, classes=np.unique(y_train))
    else:
        # Caso contrário, use fit normal
        model.fit(X_train, y_train)
        
    return model


def test_model(model, X, y, cv):
    '''
    Testar o modelo usando o conjunto de teste
    '''
    cv_score = cross_val_score(model, X, y, cv=cv)
    return cv_score


def save_model(model, cv_score, model_file, best_score_storage):
    """
    Salvar o modelo treinado em um arquivo e o score em outro
    """
    score = {"Cross Validation Scores": cv_score.tolist(),
             "Cross Validation Mean": cv_score.mean()}
    json.dump(score, open(best_score_storage, "w"))
    dump(model, model_file)
    return


def print_results(cv_score, epoch, num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    prt_cv = " - ".join([f"{(score)*100:.2f}%" for score in cv_score])
    print(f"Cross Validation Scores: {prt_cv}")
    print(f"Cross Validation Mean: {(cv_score.mean())*100:.2f}%\n")


# Função principal para executar o pipeline
def main():
    # Passo 1: Carregar os dados do CSV
    data = load_data(FILE_PATH)
    
    # Passo 2: Separar features (X) dos labels (Y)
    X, y = separate_features_labels(data, RESULTS_COLUMN)
    
    # Passo 3: Pré-processar os dados
    X, y = preprocessing(X, y)

    # Passo 4: Dividir em conjuntos de treino e teste
    # X_train, X_test, y_train, y_test = split_train_test(X, y, TEST_SIZE, RANDOM_STATE)
    
    # Passo 5: Inicializar o modelo de Classificação
    model = load(MAIN_MODEL_FILE)

    best_acc = json.load(open(BEST_SCORE_STORAGE, "r"))["Cross Validation Mean"]
    for epoch in range(NUM_EPOCHS):
        # Passo 6: Treinar o modelo
        train_model(model, X, y)
        
        # Passo 7: Testar o modelo usando o conjunto de teste
        cv_score = test_model(model, X, y, CV)

        # Passo 8: Salvar o melhor modelo
        if cv_score.mean() > best_acc:
            best_acc = cv_score.mean()
            save_model(model, cv_score, MAIN_MODEL_FILE,
                       BEST_SCORE_STORAGE)
            print_results(cv_score, epoch, NUM_EPOCHS)



# Chamando a função principal para treinar o modelo
if __name__ == "__main__":
    print("Training the scikit model\n---------------------------")
    main()
    print("\n---------------------------\nModel trained successfully!")
