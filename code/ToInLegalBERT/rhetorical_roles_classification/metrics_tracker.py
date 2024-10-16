import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


class MetricsTracker:
    def __init__(self, device="cpu"):
        self._device = torch.device(device)

    def reset(self):
        self._y_pred = torch.Tensor().to(self._device)
        self._y_true = torch.Tensor().to(self._device)

    def accumulate(self, y_pred, y_true):
        y_pred, y_true = y_pred.flatten(), y_true.flatten()

        ignore_mask = y_true != -100
        y_pred, y_true = y_pred[ignore_mask], y_true[ignore_mask]

        self._y_pred = torch.cat((self._y_pred, y_pred))
        self._y_true = torch.cat((self._y_true, y_true))

    def get(self):
        y_true = self._y_true.cpu()
        y_pred = self._y_pred.cpu()
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "MCC": matthews_corrcoef(y_true, y_pred),
            "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "Micro F1": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "Micro Precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
            "Micro Recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        }
