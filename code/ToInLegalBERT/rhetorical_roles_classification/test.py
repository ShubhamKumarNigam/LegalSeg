import pandas as pd
import csv
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from rhetorical_roles_classification import MetricsTracker


def test_BERT(
    model,
    test_dataset,
    label2rhetRole,
    device="cuda",
):
    model.to(device)
    metrics_tracker = MetricsTracker(device=device)

    model.eval()
    metrics_tracker.reset()

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128)

    with torch.no_grad():
        for data, labels in tqdm(test_dataloader):
            labels = labels.to(device)
            output = model(data.to(device), labels=labels)
            loss, logits = output.loss, output.logits
            predictions = logits.argmax(dim=-1)
            metrics_tracker.accumulate(predictions, labels)

    y_true = metrics_tracker._y_true.cpu()
    y_pred = metrics_tracker._y_pred.cpu()

    evaluate(
        y_true=y_true,
        y_pred=y_pred,
        label2rhetRole=label2rhetRole,
    )


def test_ToBERT(
    model,
    test_dataset,
    label2rhetRole,
    device="cpu",
):
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
    )
    documents = test_dataset._documents

    model.to(device)

    model.eval()
    predictions = []
    with torch.no_grad():
        for data, label in tqdm(dataloader):
            output = model(data.to(device))
            logits = output.logits
            predictions.append(logits.argmax(dim=-1))

    true_labels = []
    predicted_labels = []
    for pred, doc in zip(predictions, documents):
        n_segments = len(doc["segments"])
        true_labels += doc["labels"]
        predicted_labels += pred.flatten().tolist()[:n_segments]
    df = pd.DataFrame({
    'true': true_labels,
    'predicted': predicted_labels
    })
    df.to_csv('predictions.csv', index=False)
    evaluate(
        y_true=true_labels,
        y_pred=predicted_labels,
        label2rhetRole=label2rhetRole,
    )


def evaluate(y_true, y_pred, label2rhetRole):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    print("Overall")
    print(f"\tAccuracy: {accuracy_score(y_true, y_pred)}")
    print(f"\tMCC: {matthews_corrcoef(y_true, y_pred)}")
    print(f"\tMacro F1: {f1_score(y_true, y_pred, average='macro', zero_division=0)}")
    print(
        f"\tMacro Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0)}"
    )
    print(f"\tMacro Recall: {recall_score(y_true, y_pred, average='macro', zero_division=0)}")
    print(f"\tMicro F1: {f1_score(y_true, y_pred, average='micro', zero_division=0)}")
    print(
        f"\tMicro Precision: {precision_score(y_true, y_pred, average='micro', zero_division=0)}"
    )
    print(f"\tMicro Recall: {recall_score(y_true, y_pred, average='micro', zero_division=0)}")

    for label, rhetRole in label2rhetRole.items():
        print(f"Rhetorical role: {rhetRole}")
        true_labels = y_true == label
        pred_labels = y_pred == label
        print(
            f"\tMacro F1: {f1_score(true_labels, pred_labels, average='macro', zero_division=0)}"
        )
        print(
            f"\tMacro Precision: {precision_score(true_labels, pred_labels, average='macro', zero_division=0)}"
        )
        print(
            f"\tMacro Recall: {recall_score(true_labels, pred_labels, average='macro', zero_division=0)}"
        )
