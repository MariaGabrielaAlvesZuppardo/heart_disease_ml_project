from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

class Model:
    def __init__(self, model_type='random_forest'):
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Modelo {model_type} não suportado.")
    
    def train(self, X_train, y_train):
        """Treina o modelo com dados de treino."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Faz previsões com o modelo treinado."""
        return self.model.predict(X_test)

    def save_model(self, filename):
        """Salva o modelo treinado em um arquivo."""
        joblib.dump(self.model, filename)

    def load_model(self, filename):
        """Carrega um modelo treinado salvo a partir de um arquivo."""
        self.model = joblib.load(filename)

    def evaluate(self, y_test, y_pred):
        """Avalia o modelo utilizando métricas como acurácia."""
        return accuracy_score(y_test, y_pred)
