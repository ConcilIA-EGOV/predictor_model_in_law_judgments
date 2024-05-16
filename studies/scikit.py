from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import json
from sklearn.linear_model import SGDClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.kernel_approximation import RBFSampler
# from sklearn.kernel_approximation import Nystroem
# from sklearn.kernel_approximation import AdditiveChi2Sampler
# from sklearn.kernel_approximation import PolynomialCountSketch
# from sklearn.kernel_approximation import SkewedChi2Sampler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
# -----------
from sklearn.model_selection import GridSearchCV
###
import sys
import os

# Obtém o diretório atual do script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Adiciona o diretório base do projeto ao caminho de busca do Python
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
###
from util.parameters import FILE_PATH, RESULTS_COLUMN, PARAM_GRID
from util.parameters import NUM_EPOCHS, SCIKIT_MODEL_FILE, TEST_SIZE
from util.parameters import RANDOM_STATE, LOSS, MAX_ITER, TOL, CV, BEST_SCORE_STORAGE
from formatation.input_formatation import load_data, separate_features_labels


def grid_search(X_train, y_train):

    # Inicializar o classificador
    classifier = SGDClassifier()

    # Realizar a busca em grade
    grid_search = GridSearchCV(classifier, PARAM_GRID, cv=5)
    grid_search.fit(X_train, y_train)

    # Melhor combinação de hiperparâmetros
    best_params = grid_search.best_params_
    return best_params


def split_train_test(X, y, test_size=0.3, random_state=42):
    """
    Dividir em conjuntos de treino e teste e normalizar os dados
    """
    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y,
                                         test_size=test_size,
                                         random_state=random_state)    
    # Normalizar os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_model(model, X_train_, y_train):
    """
    Treinar o modelo usando o conjunto de treino
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_)
    model.fit(X_train, y_train)
    return model


def test_model(model, X, y, cv):
    '''
    Testar o modelo usando o conjunto de teste
    '''
    cv_score = cross_val_score(model, X, y, cv=cv)
    return cv_score


def save_model(model, mse, acc_score, cv_score,
               model_file=SCIKIT_MODEL_FILE,
               best_score_storage=BEST_SCORE_STORAGE):
    """
    Salvar o modelo treinado em um arquivo
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
    
    # Passo 3: Dividir em conjuntos de treino e teste
    # X_train, X_test, y_train, y_test = split_train_test(X, y, TEST_SIZE, RANDOM_STATE)
    
    # Passo 4: Inicializar o modelo de Classificação
    global model    

    best_acc = json.load(open(BEST_SCORE_STORAGE, "r"))["Cross Validation Mean"]
    for epoch in range(NUM_EPOCHS):
        # Passo 5: Treinar o modelo
        train_model(model, X, y)
        
        # Passo 6: Testar o modelo usando o conjunto de teste
        cv_score = test_model(model, X, y, CV)

        # Passo 7: Salvar o melhor modelo
        if cv_score.mean() > best_acc:
            best_acc = cv_score.mean()
            save_model(model, cv_score, SCIKIT_MODEL_FILE,
                       BEST_SCORE_STORAGE)
            print_results(cv_score, epoch, NUM_EPOCHS)



model = SGDClassifier(loss=LOSS, max_iter=MAX_ITER, tol=TOL)
# Chamando a função principal para treinar o modelo
if __name__ == "__main__":
    print("\n--------------------\nTraining the scikit model...")
    main()
    print("\n--------------------\nModel trained successfully!")
else:
    model = load(SCIKIT_MODEL_FILE)
