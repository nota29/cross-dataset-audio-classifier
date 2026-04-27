"""
train.py
--------
Training loop for the dual-head audio classifier.
All hyperparameters are passed explicitly — no defaults here.
Values come from configs/config.yaml via main.py.
"""

import torch

from src.losses import compute_combined_loss


def train_one_epoch(
    model: torch.nn.Module,
    loaders: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_weights: dict,
    alpha_align: float,
    temperature: float,
) -> float:
    """
    Run one full training epoch over all task loaders simultaneously.

    The shorter loader is cycled so both tasks are seen at every step.
    The number of gradient steps equals max(len(loader) for each task).

    Parameters
    ----------
    model        : DualHeadClassifier
    loaders      : {"task_b": DataLoader, "task_f": DataLoader}
    optimizer    : e.g. AdamW
    device       : torch.device
    task_weights : per-task loss multipliers (passed from config.yaml)
    alpha_align  : NTXent alignment loss coefficient (passed from config.yaml)
    temperature  : NTXent temperature (passed from config.yaml)

    Returns
    -------
    Average total loss over all steps.
    """
    model.train()
    total_loss = 0.0

    iters = {task: iter(loader) for task, loader in loaders.items()}
    max_steps = max(len(loader) for loader in loaders.values())

    for _ in range(max_steps):
        optimizer.zero_grad()

        task_outputs = {}
        for task in loaders:
            try:
                x, y, _ = next(iters[task])
            except StopIteration:
                iters[task] = iter(loaders[task])
                x, y, _ = next(iters[task])

            x, y = x.to(device), y.to(device)
            logits, features = model(x, task)
            task_outputs[task] = (logits, features, y)

        loss = compute_combined_loss(
            task_outputs,
            task_weights=task_weights,
            alpha_align=alpha_align,
            temperature=temperature,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max_steps


def run_training(
    model: torch.nn.Module,
    loaders: dict,
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    task_weights: dict,
    alpha_align: float,
    temperature: float,
) -> None:
    """
    Full training routine with epoch-level logging.

    Parameters
    ----------
    model         : DualHeadClassifier
    loaders       : {"task_b": train_loader_b, "task_f": train_loader_f}
    device        : torch.device
    num_epochs    : total training epochs
    learning_rate : AdamW learning rate
    task_weights  : per-task loss multipliers
    alpha_align   : NTXent alignment loss coefficient
    temperature   : NTXent temperature
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model,
            loaders,
            optimizer,
            device,
            task_weights=task_weights,
            alpha_align=alpha_align,
            temperature=temperature,
        )
        print(f"Epoch {epoch + 1:02d}/{num_epochs} | Loss: {train_loss:.4f}")
