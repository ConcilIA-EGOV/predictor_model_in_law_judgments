###
import sys
import os

# Obtém o diretório atual do script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Adiciona o diretório base do projeto ao caminho de busca do Python
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
###
from formatation.input_formatation import load_data, separate_features_labels
from src.preprocessing import preprocessing
from src.training import train_model, test_model, save_model, get_models
from src.training import print_results, split_train_test, is_best_model
from src.loss_function import normalEqn, gradientDescent, computeCost

# Função principal para executar o pipeline
def main():
    # Passo 1: Carregar os dados do CSV
    data = load_data()
    if data is None:
        print("Erro ao carregar os dados!")
        return
    
    # Passo 2: Separar features (X) dos labels (Y)
    X, y = separate_features_labels(data)
    
    # Passo 3: Pré-processar os dados
    X = preprocessing(X)

    # Passo 4: Dividir em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    # Passo 5: Inicializar o modelo de Classificação
    models = get_models()

    for key, model in models.items():
        # Passo 6: Treinar o modelo
        train_model(model, X_train, y_train)
        
        # Passo 7: Testar o modelo usando o conjunto de teste
        score = test_model(model, X_test, y_test)

        # Passo 8: Salvar o melhor modelo
        if is_best_model(score):
            save_model(model, score)
        print_results(key, score)

def manual():
    # Passo 1: Carregar os dados do CSV
    data = load_data()
    if data is None:
        print("Erro ao carregar os dados!")
        return
    
    # Passo 2: Separar features (X) dos labels (Y)
    X, y = separate_features_labels(data)
    
    # Passo 3: Usar o método da Equação Normal ou Gradiente Descendente
    # theta = normalEqn(X, y)
    # print("Theta usando Equação Normal:\n", theta.tolist())
    theta0 = [3345.6007871535476, 4535.495933611547, 6535.178100594566, 1810.5839771490014, 3651.706337893997, 4788.096143484616, 1522.4848712172102, 651.9448718609747, 843.7176041490065, 580.7944899252104, 694.6626243113485]
    (theta, _, _, J_history) = gradientDescent(X, y, theta0, 0.0001, 10000)
    print("Theta usando Gradiente Descendente:\n", theta)

    # Passo 4: Calcular o custo usando o theta inicial
    J = computeCost(X, y, theta)*2
    print("Custo MSE usando theta inicial:", J)
    print("Custo RMSE usando theta inicial:", J**0.5)


if __name__ == "__main__":
    # main()
    manual()
