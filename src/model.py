"""
model.py
--------
Dual-head audio classifier with a shared residual body and two
task-specific classification heads:

  - task_b  →  BSD10K   (single-label, CrossEntropy)
  - task_f  →  FSD50K   (multi-label, BCEWithLogits)
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Feed-forward residual block:  LayerNorm(x + FC2(act(FC1(x))))

    Parameters
    ----------
    dim     : feature dimension (kept constant)
    dropout : dropout probability
    """

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.norm(x + residual)


class DualHeadClassifier(nn.Module):
    """
    Shared-body, dual-head classifier for simultaneous multi-task learning.

    Architecture
    ────────────
    input → Linear(input_dim, hidden_dim)
          → N × ResidualBlock(hidden_dim)        ← shared body
          → task-specific head                   ← two-layer MLP per task

    Parameters
    ----------
    input_dim   : dimensionality of input embeddings
    num_classes_b : number of BSD10K classes (single-label)
    num_classes_f : number of FSD50K classes (multi-label)
    hidden_dim  : width of all hidden layers (default 512)
    depth       : number of residual blocks (default 6)
    dropout     : dropout probability (default 0.3)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes_b: int,
        num_classes_f: int,
        hidden_dim: int = 512,
        depth: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.body = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(depth)]
        )

        def _head(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )

        self.heads = nn.ModuleDict({
            "task_b": _head(num_classes_b),
            "task_f": _head(num_classes_f),
        })

    def forward(
        self, x: torch.Tensor, task: str
    ):
        """
        Parameters
        ----------
        x    : (B, input_dim) embedding tensor
        task : "task_b" or "task_f"

        Returns
        -------
        logits   : (B, num_classes)   raw (un-activated) scores
        features : (B, hidden_dim)    shared body representations
        """
        x = self.input_layer(x)
        features = self.body(x)
        logits = self.heads[task](features)
        return logits, features
