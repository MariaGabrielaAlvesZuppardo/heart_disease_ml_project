from sklearn.metrics import classification_report, confusion_matrix

class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, y_test, y_pred):
        """Avalia o modelo com base nas previsões."""
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm
        }

    def print_metrics(self, metrics):
        """Exibe as métricas do modelo."""
        print("Classification Report:")
        print(metrics['classification_report'])
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])
