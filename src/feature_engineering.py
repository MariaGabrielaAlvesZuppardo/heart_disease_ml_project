import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class FeatureEngineering:
    def __init__(self):
        pass

    def handle_missing_data(self, df):
        """Trata dados ausentes, se houver."""
        return df.fillna(df.mean())

    def encode_categorical_data(self, df):
        """Realiza a codificação de variáveis categóricas, se necessário."""
        return pd.get_dummies(df)

    def feature_selection(self, df, target, k=10):
        """Seleciona as top-k features mais relevantes para o modelo (para simplificação)"""
        return df
