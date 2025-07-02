import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(y_true, y_pred, classes=None, normalize=False, save_path=None):
    """
    Plot a confusion matrix given true labels and predicted labels.
    If normalize=True, the confusion matrix is normalized by true label counts.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    # If class names not provided, use numeric labels
    if classes is None:
        classes = [str(i) for i in range(len(np.unique(y_true)))]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
