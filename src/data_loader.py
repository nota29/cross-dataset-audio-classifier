"""
data_loader.py
--------------
Loads pre-computed ONE-PEACE embeddings and labels for BSD10K and FSD50K,
returning ready-to-use PyTorch DataLoaders.
"""

import csv
import os
from typing import Tuple

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset


def _load_embeddings(embedding_dir: str, sound_ids: list) -> np.ndarray:
    """Stack .npy embedding files for a list of sound IDs."""
    arrays = [
        np.load(os.path.join(embedding_dir, f"{sid}.npy"))
        for sid in sound_ids
    ]
    return np.stack([a.reshape(-1) for a in arrays]) # (N, D)


def _read_csv(path: str, id_col: int = 0, label_col: int = 1):
    """Return (sound_ids, raw_label_strings) from a metadata CSV."""
    sound_ids, labels = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            sound_ids.append(row[id_col])
            labels.append(row[label_col])
    return sound_ids, labels


# ── BSD10K ────────────────────────────────────────────────────────────────────

def load_bsd10k(
    embeddings_dir: str,
    metadata_dev: str,
    metadata_eval: str,
) -> Tuple[DataLoader, DataLoader, int, LabelEncoder]:
    """
    Load BSD10K train/eval splits.

    BSD10K is a *single-label* dataset → CrossEntropyLoss.

    Returns
    -------
    train_loader, eval_loader, num_classes, label_encoder
    """
    le = LabelEncoder()

    # ── Dev (train) ──────────────────────────────────────────────────────────
    ids_dev, raw_labels_dev = _read_csv(metadata_dev)
    X_dev = torch.tensor(_load_embeddings(embeddings_dir, ids_dev), dtype=torch.float32)
    ids_dev_t = torch.tensor([int(s) for s in ids_dev], dtype=torch.long)

    # Fit encoder on dev labels only
    y_dev = torch.tensor(le.fit_transform(raw_labels_dev), dtype=torch.long)
    num_classes = len(le.classes_)

    # ── Eval ─────────────────────────────────────────────────────────────────
    ids_eval, raw_labels_eval = _read_csv(metadata_eval)
    X_eval = torch.tensor(_load_embeddings(embeddings_dir, ids_eval), dtype=torch.float32)
    ids_eval_t = torch.tensor([int(s) for s in ids_eval], dtype=torch.long)

    # Transform with the same encoder fitted on dev
    y_eval = torch.tensor(le.transform(raw_labels_eval), dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_dev, y_dev, ids_dev_t),
        batch_size=32, shuffle=True, drop_last=True,
    )
    eval_loader = DataLoader(
        TensorDataset(X_eval, y_eval, ids_eval_t),
        batch_size=32, shuffle=False,
    )

    return train_loader, eval_loader, num_classes, le


# ── FSD50K ────────────────────────────────────────────────────────────────────

def load_fsd50k(
    embeddings_dir: str,
    metadata_dev: str,
    metadata_eval: str,
) -> Tuple[DataLoader, DataLoader, int, MultiLabelBinarizer]:
    """
    Load FSD50K train/eval splits.

    FSD50K is a *multi-label* dataset → BCEWithLogitsLoss.
    The dataset provides flat AudioSet labels as comma-separated strings.

    Returns
    -------
    train_loader, eval_loader, num_classes, mlb
    """
    mlb = MultiLabelBinarizer()

    # ── Dev (train) ──────────────────────────────────────────────────────────
    ids_dev, raw_labels_dev = _read_csv(metadata_dev)
    X_dev = torch.tensor(_load_embeddings(embeddings_dir, ids_dev), dtype=torch.float32)
    ids_dev_t = torch.tensor([int(s) for s in ids_dev], dtype=torch.long)

    split_labels_dev = [lbl.split(",") for lbl in raw_labels_dev]
    y_dev = torch.tensor(mlb.fit_transform(split_labels_dev), dtype=torch.float32)

    # ── Eval ─────────────────────────────────────────────────────────────────
    ids_eval, raw_labels_eval = _read_csv(metadata_eval)
    X_eval = torch.tensor(_load_embeddings(embeddings_dir, ids_eval), dtype=torch.float32)
    ids_eval_t = torch.tensor([int(s) for s in ids_eval], dtype=torch.long)

    split_labels_eval = [lbl.split(",") for lbl in raw_labels_eval]
    y_eval = torch.tensor(mlb.transform(split_labels_eval), dtype=torch.float32)
    num_classes = y_eval.shape[1]

    train_loader = DataLoader(
        TensorDataset(X_dev, y_dev, ids_dev_t),
        batch_size=32, shuffle=True, drop_last=True,
    )
    eval_loader = DataLoader(
        TensorDataset(X_eval, y_eval, ids_eval_t),
        batch_size=32, shuffle=False,
    )

    return train_loader, eval_loader, num_classes, mlb
