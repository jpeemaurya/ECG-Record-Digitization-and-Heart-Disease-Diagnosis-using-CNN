import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from statsmodels.stats.contingency_tables import mcnemar

def evaluate_performance(y_true, y_pred_classes):
    """Computes comprehensive evaluation metrics (Accuracy, Precision, Recall, F1)."""
    acc = accuracy_score(y_true, y_pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_classes, average='weighted')
    cm = confusion_matrix(y_true, y_pred_classes)
    
    return {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Confusion Matrix": cm
    }

def perform_mcnemar_test(baseline_preds, proposed_preds, y_true):
    """Performs McNemar's statistical test comparing baseline vs. proposed model."""
    # Contingency table: b = baseline wrong & proposed right, c = baseline right & proposed wrong
    baseline_correct = (baseline_preds == y_true)
    proposed_correct = (proposed_preds == y_true)
    
    b = np.sum(~baseline_correct & proposed_correct)
    c = np.sum(baseline_correct & ~proposed_correct)
    
    table = [[0, c], [b, 0]]
    result = mcnemar(table, exact=False, correction=True)
    
    return {
        "chi2_corrected": result.statistic,
        "p-value": result.pvalue,
        "b": b,
        "c": c
    }