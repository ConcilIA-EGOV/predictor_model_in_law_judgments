# Adiciona o diretório base do projeto ao caminho de busca do Python
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
###
import shap
import matplotlib.pyplot as plt
import joblib as jl
import pandas as pd
from util.parameters import TEST_SIZE, MODEL_PATH
from util.parameters import MAIN_MODEL_FILE, FILE_PATH, LOG_DATA_PATH
from util.parameters import BALANCE_STRATEGY, RANDOM_STATE
from src.formatation.preprocessing import load_data
from formatation.data_preparation import balance_data, split_data


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
    # Save the SHAP values to a file

    return shap_values


def map_sentences_idx(X_test: pd.DataFrame, numero_sentenca:int) -> int:
    # Busca os índices em que essa sentença aparece no X_test
    for idx, sent in enumerate(X_test['sentenca'].to_list()):
        if not isinstance(sent, int):
            print(f"Sentença-type = {type(sent)}")
            continue
        if sent == numero_sentenca:
            return idx
        # else:
        #     print(f"Sentença {sent} não é igual a {numero_sentenca}")
    raise KeyError(f"Sentença {numero_sentenca} não encontrada no conjunto de teste.")


def explain_prediction(shap_values, N_features, sent_num:int, sent_pos:int):
    """
    mostrando o valor base, cada contribuição das variáveis, e o valor final previsto.
    """
    shap.plots.waterfall(shap_values[sent_pos], show=False, max_display=N_features)
    plt.title(f"SHAP - Sentença {sent_num}")
    plt.savefig(f"{MODEL_PATH}shap_sentenca_{sent_num}.png", bbox_inches="tight", dpi=300)
    plt.clf()


def explain_global(shap_values, N_features):
    ## Global graphs

    # Importância média global
    shap.plots.bar(shap_values, show=False, max_display=N_features)
    plt.savefig(f"{MODEL_PATH}shap_global.png", bbox_inches="tight", dpi=300)
    plt.clf()
    # Dispersão do impacto por feature
    shap.plots.beeswarm(shap_values, show=False, max_display=N_features)
    plt.savefig(f"{MODEL_PATH}shap_beeswarm.png", bbox_inches="tight", dpi=300)
    plt.clf()
    return 0


if __name__ == '__main__':
    # Importar o modelo base
    base_model = jl.load(MAIN_MODEL_FILE)
    # Carregar os dados
    X, y = load_data(FILE_PATH)
    # Balancear os dados
    X_bal, y_bal, y_bin = balance_data(X, y, BALANCE_STRATEGY, RANDOM_STATE)
    # Split into train and test sets
    X_train, X_test, y_train, y_test, y_test_bin = split_data(X_bal, y_bal, TEST_SIZE, y_bin)
    # Calculate SHAP values
    shap_values = get_values(base_model, X_train, X_test)
    #
    N_features = X_test.shape[1]
    # Is needed to map the sentence number to its index in X_test
    X_test = pd.read_csv(f'{LOG_DATA_PATH}Test.csv')
    # receive the arguments from the command line
    args = sys.argv[1:]
    # check if the arguments are empty
    if len(args) > 0 and args[0].isdigit() and int(args[0]) > 0:
        sent_num = int(args[0])
        try:
            sent_pos = map_sentences_idx(X_test, sent_num)
            print(f"Explaining sentence: {sent_num}")
            explain_prediction(shap_values, N_features, sent_num, sent_pos)
        except Exception as e:
            print("Error explaining the sentence:", sent_num)
            print(e)
    else:
        print("No sentence provided.\nWill explain globally")
        explain_global(shap_values, N_features)
        print(args)
