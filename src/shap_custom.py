###
import sys
import os
# Obtém o diretório atual do script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Adiciona o diretório base do projeto ao caminho de busca do Python
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
###
import shap
import matplotlib.pyplot as plt

from src.training import get_models
from src.file_op import load_data

def map_sentences_idx(X_test, sentencas_test, numero_sentenca:list[int]) -> list[int]:
    # Busca os índices em que essa sentença aparece no X_test
    idxs = []
    for num in numero_sentenca:
        idx = sentencas_test[sentencas_test == num].index.tolist()[0]
        idxs.append(X_test.index.get_loc(idx))
    return idxs

def explain_prediction(shap_values, N_features, sent_num:list[int], sent_pos:list[int]):
    """
    mostrando o valor base, cada contribuição das variáveis, e o valor final previsto.
    """
    N = len(sent_pos)
    for i in range(N):
        # Gráfico de barras
        shap.plots.bar(shap_values[sent_pos[i]], show=False, max_display=N_features)
        plt.title(f"SHAP - Sentença {sent_num[i]} - Posição {sent_pos[i]}")
        plt.savefig(f"shap_sentenca_{sent_num[i]}.png", bbox_inches="tight", dpi=300)
        plt.clf()


def explain_global(shap_values, N_features):
    ## Global graphs

    # Importância média global
    shap.plots.bar(shap_values, show=False, max_display=N_features)
    plt.savefig("shap_global.png", bbox_inches="tight", dpi=300)
    plt.clf()
    # Dispersão do impacto por feature
    shap.plots.beeswarm(shap_values, show=False, max_display=N_features)
    plt.savefig("shap_beeswarm.png", bbox_inches="tight", dpi=300)
    plt.clf()
    return 0

def custom_shap(model, X_train, X_test, y_train) -> shap.Explanation:
    """
    Calculate SHAP values for a given model and dataset.
    Parameters:
    - model: The machine learning model to explain.
    - X_train: The training data.
    - X_test: The test data.
    - y_train: The target variable for the training data.
    Returns:
    - shap_values: The SHAP values for the test data.
    """
    # Fit the base model
    model.fit(X_train, y_train)

    # Create a SHAP explainer
    explainer = shap.Explainer(model, X_train)
    # Calculate SHAP values
    shap_values = explainer(X_test)

    return shap_values


def main():
    # receive the arguments from the command line
    args = sys.argv[1:]

    # Importar o modelo base
    model = get_models()
    base_model = model['DecisionTree']
    # Carregar os dados
    X_train, X_test, y_train, _, _, _, sent_test = load_data(split=True)
    N_features = X_test.shape[1]
    # Calculate SHAP values
    shap_values = custom_shap(base_model, X_train, X_test, y_train)

    # check if the arguments are empty
    if len(args) > 0 and args[0].isdigit():
        # check if the first argument is a number
        # convert the first argument to an integer
        sent_num = int(args[0])
        sent_pos = map_sentences_idx(X_test, [sent_test], [sent_num])
        explain_prediction(shap_values, N_features, [sent_num], [sent_pos])
    else:
        print("No valid arguments provided.\nWill explain globally")
        explain_global(shap_values, N_features)

if __name__ == '__main__':
    main()
