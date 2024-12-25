import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

class DataLoader:
    def __init__(self, data_id=45):

        self.data = None
        self.target = None
        self.load_data(data_id)

    def load_data(self, data_id):

        heart_disease = fetch_ucirepo(id=data_id)
        self.data = heart_disease.data.features
        self.target = heart_disease.data.targets
        print(f"Data loaded: {heart_disease.metadata['name']}, shape: {self.data.shape}")
        
        if self.target is None:
            print("Warning: No target variable found.")

    def preprocess_data(self):

        # Caso os dados estejam sem a target, um valor padrão será atribuído
        if self.target is None:
            print("Warning: No target variable found. Assigning the last column as target.")
            self.target = self.data.iloc[:, -1]  # Assume a última coluna como alvo
        
        # Achata o vetor da variável target (fazendo de uma matriz (n, 1) para vetor (n,))
        y = self.target.values.ravel()

        # Divisão em treino e teste (80% treino, 20% teste)
        X_train, X_test, y_train, y_test = train_test_split(self.data, y, test_size=0.2, random_state=42)

        # Escalando as características (normalizando os valores de X)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
