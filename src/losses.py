"""
losses.py
---------
Task-specific classification losses and the NTXent alignment loss
used to encourage shared representations across the two datasets.
"""

import itertools

import torch
import torch.nn.functional as F


# ── Per-task classification loss ──────────────────────────────────────────────

def compute_task_loss(task: str, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Return the appropriate loss for the given task.

    task_b (BSD10K)  → CrossEntropy  (single-label)
    task_f (FSD50K)  → BCEWithLogits (multi-label)
    """
    if task == "task_b":
        return F.cross_entropy(logits, targets)
    elif task == "task_f":
        return F.binary_cross_entropy_with_logits(logits, targets)
    else:
        raise ValueError(f"Unknown task '{task}'. Expected 'task_b' or 'task_f'.")


# ── NTXent alignment loss ─────────────────────────────────────────────────────

def ntxent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Normalised Temperature-scaled Cross-Entropy (NT-Xent) loss.

    Treats every (z1_i, z2_i) pair as a positive and all other
    2B-2 samples as negatives, encouraging the shared body to
    produce aligned representations across datasets.

    Parameters
    ----------
    z1, z2      : (B, D) feature tensors from each task branch
    temperature : softmax temperature (lower → sharper distribution)

    Returns
    -------
    Scalar loss tensor.
    """
    B = z1.size(0)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)         # (2B, D)
    sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)

    # Numerically stable: subtract row-wise max before softmax
    sim = sim - torch.max(sim, dim=1, keepdim=True)[0]

    # Positive pair indices: z1_i ↔ z2_i
    positives = torch.cat([
        torch.arange(B, 2 * B),
        torch.arange(0, B),
    ]).to(z.device)

    return F.cross_entropy(sim, positives)


# ── Combined multi-task loss ───────────────────────────────────────────────────

def compute_combined_loss(
    task_outputs: dict,          # {task: (logits, features, targets)}
    task_weights: dict,          # {task: float}
    alpha_align: float = 0.0001,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Weighted sum of per-task classification losses + NTXent alignment loss.

    Total loss = Σ w_t · L_t   +   α · (1/|pairs|) Σ NTXent(f_t1, f_t2)

    Parameters
    ----------
    task_outputs : dict mapping task name → (logits, features, targets)
    task_weights : per-task loss weight
    alpha_align  : alignment loss coefficient
    temperature  : NTXent temperature

    Returns
    -------
    Scalar total loss tensor.
    """
    total = torch.tensor(0.0, device=next(iter(task_outputs.values()))[0].device)

    features_by_task = {}
    for task, (logits, features, targets) in task_outputs.items():
        loss = compute_task_loss(task, logits, targets)
        total = total + task_weights.get(task, 1.0) * loss
        features_by_task[task] = features

    # Alignment loss over all task pairs
    pairs = list(itertools.combinations(features_by_task.keys(), 2))
    if pairs and alpha_align > 0:
        align = sum(
            ntxent_loss(features_by_task[t1], features_by_task[t2], temperature)
            for t1, t2 in pairs
        ) / len(pairs)
        total = total + alpha_align * align

    return total
