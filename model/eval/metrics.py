from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

def get_f1(y_true, y_pred):
    """Compute F1 score (binary classification)."""
    return f1_score(y_true, y_pred)

def get_precision(y_true, y_pred):
    """Compute precision (binary classification)."""
    return precision_score(y_true, y_pred)

def get_recall(y_true, y_pred):
    """Compute recall (binary classification)."""
    return recall_score(y_true, y_pred)

def get_roc_auc(y_true, y_score):
    """Compute ROC-AUC (binary classification)."""
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        # If only one class present in y_true, roc_auc is not defined
        return None

def get_pr_auc(y_true, y_score):
    """Compute PR-AUC (average precision) (binary classification)."""
    return average_precision_score(y_true, y_score)
