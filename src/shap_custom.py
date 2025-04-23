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
from joblib import load
from src.file_op import load_data


def get_values(model, X_train, X_test) -> shap.Explanation:
    """
    Calculate SHAP values for a given model and dataset.
    Parameters:
    - model: The machine learning model to explain.
    - X_train: The training data.
    - X_test: The test data.
    Returns:
    - shap_values: The SHAP values for the test data.
    """

    # Create a SHAP explainer
    explainer = shap.Explainer(model, X_train)
    # Calculate SHAP values
    shap_values = explainer(X_test)

    return shap_values


def map_sentences_idx(X_test, sentencas_test, numero_sentenca:int) -> int:
    # Busca os índices em que essa sentença aparece no X_test
    idx = sentencas_test[sentencas_test == numero_sentenca].index.tolist()[0]
    return X_test.index.get_loc(idx)


def explain_prediction(shap_values, N_features, sent_num:int, sent_pos:int):
    """
    mostrando o valor base, cada contribuição das variáveis, e o valor final previsto.
    """
    shap.plots.waterfall(shap_values[sent_pos], show=False, max_display=N_features)
    plt.title(f"SHAP - Sentença {sent_num}")
    plt.savefig(f"shap_sentenca_{sent_num}.png", bbox_inches="tight", dpi=300)
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


def custom_shap(model_path='models_storage/DecisionTree.pkl', sent_num=0):
    # Importar o modelo base
    base_model = load(model_path)
    # Carregar os dados
    X_train, X_test, _, _, _, _, sent_test = load_data(split=True)
    N_features = X_test.shape[1]
    # Calculate SHAP values
    shap_values = get_values(base_model, X_train, X_test)
    if sent_num > 0:
        sent_pos = map_sentences_idx(X_test, sent_test, sent_num)
        explain_prediction(shap_values, N_features, sent_num, sent_pos)
    else:
        explain_global(shap_values, N_features)

if __name__ == '__main__':
    # receive the arguments from the command line
    args = sys.argv[1:]
    # check if the arguments are empty
    if len(args) == 0:
        print("No arguments provided.\nusage: python shap_custom.py model_path [sentence_number]")
        custom_shap()
    elif len(args) > 1 and args[1].isdigit():
        model_path = args[0]
        sent_num = int(args[1])
        print(f"Explaining sentence: {sent_num}")
        custom_shap(model_path, sent_num)
    else:
        print("No sentence provided.\nWill explain globally")
        model_path = args[0]
        custom_shap(model_path)
