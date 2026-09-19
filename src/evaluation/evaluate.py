from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_predictions(y_true, y_pred):
    """
    Calcule l'accuracy et génère un rapport d'évaluation détaillé.
    """
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": conf_matrix
    }
    return metrics
