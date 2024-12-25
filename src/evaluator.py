from sklearn.metrics import accuracy_score

class Evaluation:
    def __init__(self):
        """
        Inicializa a classe Evaluation para avaliação de desempenho do modelo.
        """
        pass

    def evaluate(self, model, X_test, y_test):
        """
        Avalia o desempenho do modelo usando precisão.
        """
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
