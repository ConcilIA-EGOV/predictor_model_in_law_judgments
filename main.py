###
import sys
import os

# Obtém o diretório atual do script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Adiciona o diretório base do projeto ao caminho de busca do Python
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
###
import custom_models.DecisionTree as DT
import custom_models.RandomForest as RF

# Função principal para executar o pipeline
def main():
    DT.main()
    #RF.main()
if __name__ == "__main__":
    main()
