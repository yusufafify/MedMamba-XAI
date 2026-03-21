"""Test-set evaluation script with confusion matrix generation.

Loads a trained checkpoint, runs inference on the test split, and produces:
- Per-class precision, recall, F1
- Overall accuracy and macro-F1
- Confusion matrix saved as PNG

Usage::

    python scripts/evaluate.py \\
        --config configs/default.yaml \\
        --data configs/data/pathmnist.yaml \\
        --checkpoint outputs/best_model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from medical_mamba.data import (
    DATASET_META,
    MedMNISTFolder,
    get_val_transforms,
)
from medical_mamba.models import MedicalVMamba
from medical_mamba.utils import load_checkpoint, set_seed


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """Run inference on the test set.

    Parameters
    ----------
    model : nn.Module
        Loaded model in eval mode.
    loader : DataLoader
        Test data loader.
    device : torch.device
        Inference device.

    Returns
    -------
    tuple
        ``(all_preds, all_labels)`` as numpy arrays.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            logits, _ = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="MedMamba-XAI Test Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/eval")
    args = parser.parse_args()

    # Load config
    cfg: Dict[str, Any] = {}
    for path in [args.config, args.data]:
        if path and Path(path).exists():
            with open(path) as f:
                partial = yaml.safe_load(f) or {}
                for key, val in partial.items():
                    if isinstance(val, dict) and key in cfg and isinstance(cfg[key], dict):
                        cfg[key].update(val)
                    else:
                        cfg[key] = val

    set_seed(cfg.get("project", {}).get("seed", 42))
    device = torch.device(cfg.get("project", {}).get("device", "cuda:0"))

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    ds_name = data_cfg.get("dataset", "pathmnist")
    img_size = data_cfg.get("img_size", 224)

    # Dataset
    test_ds = MedMNISTFolder(
        ds_name,
        data_cfg.get("root", "./data"),
        "test",
        get_val_transforms(ds_name, img_size),
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    # Model
    task_classes = {"default": DATASET_META[ds_name]["num_classes"]}
    model = MedicalVMamba(
        in_channels=model_cfg.get("in_channels", 3),
        depths=model_cfg.get("depths", [2, 2, 9, 2]),
        dims=model_cfg.get("dims", [96, 192, 384, 768]),
        task_classes=task_classes,
        drop_path_rate=model_cfg.get("drop_path_rate", 0.2),
        ssm_ratio=model_cfg.get("ssm_ratio", 2.0),
        patch_size=model_cfg.get("patch_size", 4),
    )
    model, _, epoch, ckpt_metrics = load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device)

    print(f"Loaded checkpoint from epoch {epoch} | ckpt metrics: {ckpt_metrics}")

    # Evaluate
    preds, labels = evaluate(model, test_loader, device)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)

    print(f"\n{'='*50}")
    print(f"Dataset : {ds_name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-macro: {f1:.4f}")
    print(f"{'='*50}\n")
    print(classification_report(labels, preds, zero_division=0))

    # Confusion matrix
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {ds_name} (Acc={acc:.2%})")
    fig.savefig(str(output_dir / f"{ds_name}_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix → {output_dir / f'{ds_name}_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
