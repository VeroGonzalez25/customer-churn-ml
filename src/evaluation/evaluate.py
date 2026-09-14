from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd

def calculate_metrics(y_true, y_pred, y_proba=None) -> dict:
    """
    Calcula métricas clave para clasificación binaria.
    """
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        
    cm = confusion_matrix(y_true, y_pred)
    # cm: [[TN, FP], [FN, TP]]
    metrics['false_negatives'] = int(cm[1, 0])
    metrics['false_positives'] = int(cm[0, 1])
    
    return metrics