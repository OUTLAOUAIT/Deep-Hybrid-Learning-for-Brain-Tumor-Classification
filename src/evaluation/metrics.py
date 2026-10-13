from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)


def classification_metrics(y_true, y_pred, probabilities=None):
    result = {"accuracy": accuracy_score(y_true, y_pred),
              "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
              "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
              "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
              "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
              "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
              "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
              "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
              "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
              "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)}
    if probabilities is not None:
        try:
            result["roc_auc_ovr_macro"] = roc_auc_score(y_true, probabilities,
                                                         multi_class="ovr", average="macro")
        except ValueError:
            result["roc_auc_ovr_macro"] = None
    return result
