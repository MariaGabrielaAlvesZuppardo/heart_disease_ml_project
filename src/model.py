from sklearn.ensemble import RandomForestClassifier

class Model:
    def __init__(self):
        """
        Inicializa a classe Model com um modelo de classificação (Random Forest).
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        """
        Treina o modelo com os dados de treino.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Faz previsões com os dados de teste.
        """
        return self.model.predict(X_test)
