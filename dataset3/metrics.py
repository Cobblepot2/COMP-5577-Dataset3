"""Ten-class classification metrics computed explicitly from a confusion matrix."""
import numpy as np


def classification_metrics(labels, predictions, num_classes=10):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    tp = matrix.diagonal()
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(num_classes), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(num_classes), where=support != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(num_classes), where=(support + predicted) != 0)
    return {"records": int(matrix.sum()), "accuracy": float(tp.sum() / matrix.sum()),
            "macro_f1": float(f1.mean()), "confusion_matrix_rows_true_columns_predicted": matrix.tolist(),
            "per_class": [{"label": k, "support": int(support[k]),
                           "precision": float(precision[k]), "recall": float(recall[k]), "f1": float(f1[k])}
                          for k in range(num_classes)]}
