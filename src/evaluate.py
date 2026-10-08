"""
evaluate.py
-----------
Evaluation utilities for the dual-head audio classifier.

Produces a per-class classification report (precision, recall, F1)
plus micro-F1, macro-F1, and macro-mAP, saved to a CSV file.
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import label_binarize


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    task: str,
    output_csv: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Evaluate the model on one dataset split and save results to CSV.

    Parameters
    ----------
    model      : trained DualHeadClassifier
    loader     : DataLoader for the evaluation split
    device     : torch.device
    task       : "task_b" (BSD10K, single-label) or
                 "task_f" (FSD50K, multi-label)
    output_csv : path where the CSV report is written
    threshold  : sigmoid threshold for multi-label prediction (task_f only)

    Returns
    -------
    pd.DataFrame with per-class metrics and summary rows.
    """
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    for x, y, _ in loader:
        x = x.to(device)
        logits, _ = model(x, task)

        if task == "task_b":
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

        elif task == "task_f":
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(y)

    # ── Consolidate ──────────────────────────────────────────────────────────
    all_probs = torch.cat(all_probs).numpy()

    if task == "task_f":
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
    else:
        all_labels = np.array(all_labels)

    # ── Per-class report ─────────────────────────────────────────────────────
    report_dict = classification_report(
        all_labels, all_preds, output_dict=True, zero_division=0
    )
    df = pd.DataFrame(report_dict).transpose()

    # ── Summary metrics ──────────────────────────────────────────────────────
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    df.loc["micro_f1"] = {"precision": None, "recall": None, "f1-score": round(micro_f1, 4), "support": None}
    df.loc["macro_f1"] = {"precision": None, "recall": None, "f1-score": round(macro_f1, 4), "support": None}

    # ── mAP ──────────────────────────────────────────────────────────────────
    if task == "task_b":
        num_classes = all_probs.shape[1]
        y_bin = label_binarize(all_labels, classes=np.arange(num_classes))
        macro_map = average_precision_score(y_bin, all_probs, average="macro")
    else:  # task_f — already multi-hot
        macro_map = average_precision_score(all_labels, all_probs, average="macro")

    df.loc["macro_mAP"] = {"precision": None, "recall": None, "f1-score": round(macro_map, 4), "support": None}

    # ── Round numeric columns to 3 decimal places ────────────────────────────
    numeric_cols = ["precision", "recall", "f1-score", "support"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(3)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv)
    print(
        f"[{task}] micro-F1: {micro_f1:.4f} | macro-F1: {macro_f1:.4f} | "
        f"macro-mAP: {macro_map:.4f}  →  saved to {output_csv}"
    )

    return df
