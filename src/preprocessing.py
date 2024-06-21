###
import sys
import os
# Obtém o diretório atual do script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Adiciona o diretório base do projeto ao caminho de busca do Python
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
###

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
from util.parameters import FILE_PATH, PREP

def preprocessing(X: pd.DataFrame):
    """
    Preprocessar os dados
    """
    if not PREP:
        return X
    categorical_features = X.select_dtypes(include=['int', 'int64']).columns.tolist()
    categorical_features = [col for col in categorical_features if X[col].nunique() > 2]
    continuous_features = X.select_dtypes(include=['float', 'float64']).columns.tolist()
    
    # Aplicar RobustScaler seguido por StandardScaler às colunas contínuas e float
    if continuous_features:
        scaler_pipeline = Pipeline([
            ('robust', RobustScaler()),
            ('standard', StandardScaler())
        ])
        X[continuous_features] = scaler_pipeline.fit_transform(X[continuous_features])
    
    # Aplicar OneHotEncoder às colunas categóricas
    if categorical_features:
        ohe = OneHotEncoder(drop='if_binary', sparse_output=False)
        X_ohe = ohe.fit_transform(X[categorical_features])
        ohe_feature_names = ohe.get_feature_names_out(categorical_features)
        X_ohe_df = pd.DataFrame(X_ohe, columns=ohe_feature_names, index=X.index)
        X = X.drop(columns=categorical_features)
        X = pd.concat([X, X_ohe_df], axis=1)
    
    # Salvar o DataFrame modificado de volta ao arquivo CSV
    new_file = FILE_PATH.replace(".csv", "__PREP.csv")
    X.to_csv(new_file, index=False)
    print(X.columns)

    return X


if __name__ == "__main__":
    pass
