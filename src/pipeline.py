from src.data_loader import DataLoader
from src.model import Model
from src.evaluator import Evaluation

class Pipeline:
    def __init__(self):
        """
        Inicializa a pipeline com os componentes do DataLoader, Model e Evaluation.
        """
        self.data_loader = DataLoader()
        self.model = Model()
        self.evaluation = Evaluation()

    def run(self):
        """
        Roda toda a pipeline de treinamento e avaliação.
        """
        # Carrega e pré-processa os dados
        X_train, X_test, y_train, y_test = self.data_loader.preprocess_data()

        # Treina o modelo com os dados de treino
        self.model.train(X_train, y_train)

        # Avalia o modelo com os dados de teste
        accuracy = self.evaluation.evaluate(self.model.model, X_test, y_test)
        print(f'Model accuracy: {accuracy:.4f}')


# Executar a pipeline
if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()
