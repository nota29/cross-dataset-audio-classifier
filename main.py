"""
main.py
-------
Entry point for training and evaluating the Cross-Dataset Audio Classifier.

Usage
-----
    python main.py                        # uses configs/config.yaml
    python main.py --config configs/config.yaml
    python main.py --epochs 50 --lr 1e-4  # CLI overrides
"""

import argparse
import os

import torch
import yaml

from src.data_loader import load_bsd10k, load_fsd50k
from src.evaluate import evaluate
from src.model import DualHeadClassifier
from src.train import run_training


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the dual-head audio classifier."
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--epochs",    type=int,   default=None)
    parser.add_argument("--lr",        type=float, default=None)
    parser.add_argument("--batch_size",type=int,   default=None)
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # CLI overrides
    if args.epochs     is not None: cfg["training"]["num_epochs"]   = args.epochs
    if args.lr         is not None: cfg["training"]["learning_rate"] = args.lr
    if args.batch_size is not None: cfg["training"]["batch_size"]    = args.batch_size

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n── Loading BSD10K ──────────────────────────────────────────────")
    train_loader_b, eval_loader_b, num_classes_b, _ = load_bsd10k(
        embeddings_dir=cfg["bsd"]["embeddings_dir"],
        metadata_dev=cfg["bsd"]["metadata_dev"],
        metadata_eval=cfg["bsd"]["metadata_eval"],
        batch_size=cfg["training"]["batch_size"],
    )
    print(f"   BSD10K classes: {num_classes_b}")

    print("\n── Loading FSD50K ──────────────────────────────────────────────")
    train_loader_f, eval_loader_f, num_classes_f, _ = load_fsd50k(
        embeddings_dir=cfg["fsd"]["embeddings_dir"],
        metadata_dev=cfg["fsd"]["metadata_dev"],
        metadata_eval=cfg["fsd"]["metadata_eval"],
        batch_size=cfg["training"]["batch_size"],
    )
    print(f"   FSD50K classes: {num_classes_f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    # input_dim is set in config (ONE-PEACE audio embeddings = 1536-dim)
    input_dim = cfg.get("input_dim", next(iter(train_loader_b))[0].shape[1])
    model_cfg = cfg["model"]

    model = DualHeadClassifier(
        input_dim=input_dim,
        num_classes_b=num_classes_b,
        num_classes_f=num_classes_f,
        hidden_dim=model_cfg.get("hidden_dim", 512),
        depth=model_cfg.get("depth", 6),
        dropout=model_cfg.get("dropout", 0.3),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")

    # ── Training ──────────────────────────────────────────────────────────────
    train_cfg = cfg["training"]
    print("\n── Training ────────────────────────────────────────────────────")
    run_training(
        model=model,
        loaders={"task_b": train_loader_b, "task_f": train_loader_f},
        device=device,
        num_epochs=train_cfg["num_epochs"],
        learning_rate=train_cfg["learning_rate"],
        task_weights=train_cfg["task_weights"],
        alpha_align=train_cfg["alpha_align"],
        temperature=train_cfg["ntxent_temperature"],
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    results_dir = cfg["output"]["results_dir"]
    eval_cfg    = cfg["evaluation"]

    print("\n── Evaluation ──────────────────────────────────────────────────")
    evaluate(
        model=model,
        loader=eval_loader_b,
        device=device,
        task="task_b",
        output_csv=os.path.join(results_dir, cfg["output"]["report_bsd"]),
        threshold=eval_cfg["threshold"],
    )
    evaluate(
        model=model,
        loader=eval_loader_f,
        device=device,
        task="task_f",
        output_csv=os.path.join(results_dir, cfg["output"]["report_fsd"]),
        threshold=eval_cfg["threshold"],
    )


if __name__ == "__main__":
    main()
