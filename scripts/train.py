"""Main training entry point.

Merge decisions
---------------
Config loading : Agent's YAML-driven approach (``--config``, ``--data``,
                 ``--model``, ``--training``) over website's flat argparse.
                 The project uses ``configs/`` YAML files as the source of
                 truth; CLI flags only override them.

TrainConfig    : After YAML loading, the dict is converted to a typed
                 ``TrainConfig`` dataclass so the rest of the codebase gets
                 IDE autocomplete and type checking.

DataLoaders    : Constructed here (not inside ``Trainer.__init__``) so they
                 can be inspected, logged, or replaced in experiments without
                 touching the trainer class.  The ``Trainer`` receives finished
                 loaders, not config dicts.

set_seed       : From agent — essential for reproducible experiments.

Usage::

    # Single dataset, fast iteration
    python scripts/train.py \\
        --config configs/default.yaml \\
        --data   configs/data/pathmnist.yaml \\
        --model  configs/model/vmamba_tiny.yaml \\
        --training configs/training/single_task.yaml

    # Multi-task
    python scripts/train.py \\
        --config configs/default.yaml \\
        --data   configs/data/multitask.yaml \\
        --model  configs/model/vmamba_tiny.yaml \\
        --training configs/training/multi_task.yaml

    # Resume + test only
    python scripts/train.py \\
        --config configs/default.yaml \\
        --resume runs/medical_mamba/checkpoint_best.pt \\
        --test
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from medical_mamba.data.constants import DATASET_META
from medical_mamba.data.dataset import build_dataloaders
from medical_mamba.data.transforms import build_transforms_map
from medical_mamba.training.trainer import Trainer, TrainConfig
from medical_mamba.utils.seed import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if path is None or not Path(path).exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge_configs(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge YAML dicts left-to-right (later dicts win)."""
    merged: Dict[str, Any] = {}
    for d in dicts:
        for key, value in d.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MedicalVMamba training script")
    parser.add_argument("--config",   type=str, default="configs/default.yaml")
    parser.add_argument("--data",     type=str, default=None)
    parser.add_argument("--model",    type=str, default=None)
    parser.add_argument("--training", type=str, default=None)
    parser.add_argument("--resume",   type=str, default=None,
                        help="Path to checkpoint.pt to resume from")
    parser.add_argument("--test",     action="store_true",
                        help="Skip training and run test evaluation only")
    args = parser.parse_args()

    # ── Merge YAML configs ────────────────────────────────────────────────
    cfg_dict = merge_configs(
        _load_yaml(args.config),
        _load_yaml(args.data),
        _load_yaml(args.model),
        _load_yaml(args.training),
    )

    # CLI overrides
    if args.resume:
        cfg_dict.setdefault("resume", args.resume)

    # ── Build typed config ────────────────────────────────────────────────
    project_cfg  = cfg_dict.get("project",  {})
    data_cfg     = cfg_dict.get("data",     {})
    model_cfg    = cfg_dict.get("model",    {})
    training_cfg = cfg_dict.get("training", {})

    dataset_roots = data_cfg.get("roots", {})

    cfg = TrainConfig(
        dataset_roots   = dataset_roots,
        single_dataset  = data_cfg.get("single_dataset", None),
        model_size      = model_cfg.get("size",           "tiny"),
        patch_size      = model_cfg.get("patch_size",     8),
        head_dropout    = model_cfg.get("head_dropout",   0.1),
        epochs          = training_cfg.get("epochs",      100),
        lr              = training_cfg.get("lr",          1e-4),
        weight_decay    = training_cfg.get("weight_decay", 0.05),
        warmup_epochs   = training_cfg.get("warmup_epochs", 10),
        grad_clip       = training_cfg.get("grad_clip",   1.0),
        label_smoothing = training_cfg.get("label_smoothing", 0.1),
        use_amp         = training_cfg.get("mixed_precision", True),
        patience        = training_cfg.get("early_stopping_patience", 15),
        output_dir      = project_cfg.get("output_dir",  "runs/medical_mamba"),
        device          = project_cfg.get("device",      "cuda" if torch.cuda.is_available() else "cpu"),
        resume          = args.resume,
    )

    # ── Reproducibility ───────────────────────────────────────────────────
    set_seed(project_cfg.get("seed", 42))

    # ── Determine active datasets ─────────────────────────────────────────
    if cfg.single_dataset is not None:
        active_roots = {cfg.single_dataset: cfg.dataset_roots[cfg.single_dataset]}
    else:
        active_roots = cfg.dataset_roots

    # ── Transforms ───────────────────────────────────────────────────────
    transforms_map = build_transforms_map(list(active_roots.keys()))

    # ── DataLoaders ───────────────────────────────────────────────────────
    loaders = build_dataloaders(
        dataset_roots=active_roots,
        transforms_map=transforms_map,
        batch_size=project_cfg.get("batch_size", 32),
        num_workers=project_cfg.get("num_workers", 0),
        pin_memory="cuda" in cfg.device,
        single_dataset=None,   # already resolved above
    )

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(
        cfg=cfg,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders["test"],
    )

    if args.test:
        trainer.evaluate_test(args.resume)
    else:
        result = trainer.fit()
        print(
            f"\n✓ Training complete. "
            f"Best avg_f1_macro: {result['best_avg_f1']:.4f}\n"
            f"  Checkpoint: {result['checkpoint']}"
        )
        trainer.evaluate_test()


if __name__ == "__main__":
    main()
