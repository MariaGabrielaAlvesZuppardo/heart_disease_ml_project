from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineering
from src.model import Model
from src.evaluator import Evaluator

class Pipeline:
    def __init__(self, dataset_id=45):
        self.data_loader = DataLoader(dataset_id=dataset_id)
        self.feature_engineering = FeatureEngineering()
        self.model = Model(model_type='random_forest')  
        self.evaluator = Evaluator()

    def run(self):
        """Executa toda a pipeline de ML."""
        print("Carregando dados...")
        data = self.data_loader.load_data()

        print("Pré-processando dados...")
        X, y = self.data_loader.preprocess_data()

        print("Dividindo os dados em treino e teste...")
        X_train, X_test, y_train, y_test = self.data_loader.split_data(X, y)

        print("Treinando o modelo...")
        self.model.train(X_train, y_train)

        print("Fazendo predições...")
        y_pred = self.model.predict(X_test)

        print("Avaliando o modelo...")
        metrics = self.evaluator.evaluate(y_test, y_pred)
        print("Métricas do modelo:", metrics)

if __name__ == "__main__":
    # Rodar a pipeline com o ID do dataset desejado
    pipeline = Pipeline(dataset_id=45)
    pipeline.run()
