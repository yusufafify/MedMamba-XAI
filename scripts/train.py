"""Main training entry point.

Config system
-------------
All settings live in YAML files under configs/. CLI flags select which
files to load. Files are deep-merged left-to-right — later keys win.

Merge order:
    default.yaml → data/*.yaml → model/*.yaml → training/*.yaml

Each config section owns its keys exclusively:
    project.*  → infrastructure  (batch_size, num_workers, device, seed)
    data.*     → data loading    (roots, single_dataset, img_size)
    model.*    → architecture    (size, patch_size, embed_dim, depths)
    training.* → hyperparameters (lr, epochs, scheduler, patience)

No section duplicates keys from another section.

Usage::

    # Single dataset — VMamba
    python scripts/train.py \\
        --config   configs/default.yaml \\
        --data     configs/data/pathmnist.yaml \\
        --model    configs/model/vmamba_tiny.yaml \\
        --training configs/training/single_task.yaml

    # Single dataset — ResNet50 baseline
    python scripts/train.py \\
        --config   configs/default.yaml \\
        --data     configs/data/pathmnist.yaml \\
        --model    configs/model/resnet50.yaml \\
        --training configs/training/single_task.yaml

    # Multi-task
    python scripts/train.py \\
        --config   configs/default.yaml \\
        --data     configs/data/multitask.yaml \\
        --model    configs/model/vmamba_tiny.yaml \\
        --training configs/training/multi_task.yaml

    # Resume from checkpoint
    python scripts/train.py \\
        --config   configs/default.yaml \\
        --data     configs/data/pathmnist.yaml \\
        --resume   runs/medical_mamba/checkpoint_best.pt

    # Test-set evaluation only
    python scripts/train.py \\
        --config   configs/default.yaml \\
        --data     configs/data/pathmnist.yaml \\
        --resume   runs/medical_mamba/checkpoint_best.pt \\
        --test
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from medical_mamba.data.dataset import build_dataloaders
from medical_mamba.data.transforms import build_transforms_map
from medical_mamba.models.medical_vmamba import build_model
from medical_mamba.models.resnet_baseline import ResNetBaseline
from medical_mamba.models.vit_baseline import ViTBaseline
from medical_mamba.data.constants import DATASET_META
from medical_mamba.training.trainer import Trainer, TrainConfig
from medical_mamba.utils.seed import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    """Load a YAML file. Returns {} silently if path is None or missing."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base. Override wins on conflicts."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    config:   Optional[str],
    data:     Optional[str],
    model:    Optional[str],
    training: Optional[str],
) -> Dict[str, Any]:
    """Load and merge all config files in correct precedence order."""
    cfg = _load_yaml(config)
    for path in (data, model, training):
        cfg = _deep_merge(cfg, _load_yaml(path))
    return cfg


def _require(cfg: Dict[str, Any], section: str, key: str, hint: str = "") -> Any:
    """Get a required config value — raises clearly if missing."""
    val = cfg.get(section, {}).get(key)
    if val is None:
        msg = f"Missing required config: {section}.{key}"
        if hint:
            msg += f" ({hint})"
        raise KeyError(msg)
    return val


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MedicalVMamba training script")
    parser.add_argument("--config",   type=str, default="configs/default.yaml",
                        help="Base config (default: configs/default.yaml)")
    parser.add_argument("--data",     type=str, default=None,
                        help="Data config    e.g. configs/data/pathmnist.yaml")
    parser.add_argument("--model",    type=str, default=None,
                        help="Model config   e.g. configs/model/vmamba_tiny.yaml")
    parser.add_argument("--training", type=str, default=None,
                        help="Training config e.g. configs/training/single_task.yaml")
    parser.add_argument("--resume",   type=str, default=None,
                        help="Checkpoint path to resume training from")
    parser.add_argument("--test",     action="store_true",
                        help="Skip training — run test evaluation only")
    args = parser.parse_args()

    # ── Load and merge all configs ────────────────────────────────────────
    cfg = load_config(args.config, args.data, args.model, args.training)

    project  = cfg.get("project",  {})
    data     = cfg.get("data",     {})
    model    = cfg.get("model",    {})
    training = cfg.get("training", {})

    # ── Reproducibility ───────────────────────────────────────────────────
    set_seed(project.get("seed", 42))

    # ── Resolve device ────────────────────────────────────────────────────
    device = project.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # ── Build TrainConfig ─────────────────────────────────────────────────
    # Every field is read from its canonical section.
    # No hardcoded fallbacks that could silently hide missing config keys.
    cfg_obj = TrainConfig(
        # data
        dataset_roots  = data.get("roots", {}),
        single_dataset = data.get("single_dataset", None),
        # model
        model_size     = model.get("size",         "tiny"),
        patch_size     = model.get("patch_size",   8),
        head_dropout   = model.get("head_dropout", 0.1),
        # training — optimiser
        epochs         = training.get("epochs",        100),
        lr             = training.get("lr",             1e-4),
        min_lr         = training.get("min_lr",         1e-6),
        weight_decay   = training.get("weight_decay",   0.05),
        warmup_epochs  = training.get("warmup_epochs",  10),
        grad_clip      = training.get("grad_clip",      1.0),
        label_smoothing= training.get("label_smoothing", 0.1),
        use_amp        = training.get("mixed_precision", True),
        # training — scheduler
        scheduler      = training.get("scheduler",       "cosine_warmup"),
        plateau_factor = training.get("plateau_factor",  0.5),
        plateau_patience=training.get("plateau_patience", 5),
        plateau_min_lr = training.get("plateau_min_lr",  1e-6),
        # training — stopping & saving
        patience       = training.get("early_stopping_patience", 15),
        save_best_metric=training.get("save_best_metric", "avg_f1_macro"),
        # training — contrastive domain discovery (SupCon)
        use_contrastive                   = training.get("use_contrastive", False),
        contrastive_lambda                = training.get("contrastive_lambda", 0.1),
        contrastive_temp                  = training.get("contrastive_temp", 0.07),
        contrastive_warmup                = training.get("contrastive_warmup", 10),
        compute_prototypes_after_training = training.get("compute_prototypes_after_training", True),
        # infrastructure
        output_dir     = project.get("output_dir", "runs/medical_mamba"),
        device         = device,
        resume         = args.resume,
    )

    # ── Validate dataset roots ────────────────────────────────────────────
    if not cfg_obj.dataset_roots:
        raise ValueError(
            "No dataset roots configured. "
            "Check data.roots in configs/default.yaml or your data config."
        )

    # ── Resolve active datasets ───────────────────────────────────────────
    if cfg_obj.single_dataset is not None:
        if cfg_obj.single_dataset not in cfg_obj.dataset_roots:
            raise KeyError(
                f"single_dataset='{cfg_obj.single_dataset}' not in data.roots. "
                f"Available: {list(cfg_obj.dataset_roots.keys())}"
            )
        active_roots = {cfg_obj.single_dataset: cfg_obj.dataset_roots[cfg_obj.single_dataset]}
    else:
        active_roots = cfg_obj.dataset_roots

    # ── Transforms ───────────────────────────────────────────────────────
    transforms_map = build_transforms_map(list(active_roots.keys()))

    # ── DataLoaders ───────────────────────────────────────────────────────
    loaders = build_dataloaders(
        dataset_roots  = active_roots,
        transforms_map = transforms_map,
        batch_size     = project.get("batch_size",  32),
        num_workers    = project.get("num_workers", 0),
        pin_memory     = "cuda" in device,
        single_dataset = None,  # already resolved into active_roots
    )

    # ── Resolve active task configs (needed for model head construction) ──────
    if cfg_obj.single_dataset is not None:
        active_task_keys = [cfg_obj.single_dataset]
    else:
        active_task_keys = list(active_roots.keys())
    task_configs = [
        (name, DATASET_META[name]["num_classes"])
        for name in active_task_keys
    ]

    # ── Build model ────────────────────────────────────────────────────
    model_type = model.get("type", "vmamba").lower()
    if model_type == "resnet50":
        print(f"[model] Building ResNet50 baseline (pretrained={model.get('pretrained', True)})")
        built_model = ResNetBaseline(
            task_configs=task_configs,
            pretrained=model.get("pretrained", True),
            head_dropout=model.get("head_dropout", 0.1),
        )
    elif model_type == "vit":
        print(f"[model] Building ViT baseline (size={cfg_obj.model_size}, pretrained={model.get('pretrained', True)})")
        built_model = ViTBaseline(
            task_configs=task_configs,
            model_size=cfg_obj.model_size,
            pretrained=model.get("pretrained", True),
            head_dropout=model.get("head_dropout", 0.1),
        )
    else:
        print(f"[model] Building VMamba-{cfg_obj.model_size} (patch={cfg_obj.patch_size})")
        built_model = build_model(
            task_configs=task_configs,
            model_size=cfg_obj.model_size,
            patch_size=cfg_obj.patch_size,
            head_dropout=cfg_obj.head_dropout,
        )

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(
        cfg          = cfg_obj,
        train_loader = loaders["train"],
        val_loader   = loaders["val"],
        test_loader  = loaders["test"],
        model        = built_model,
    )

    if args.test:
        trainer.evaluate_test(args.resume)
    else:
        result = trainer.fit()
        print(
            f"\nTraining complete."
            f"\n  Best {cfg_obj.save_best_metric}: {result['best_avg_f1']:.4f}"
            f"\n  Checkpoint: {result['checkpoint']}"
        )
        trainer.evaluate_test()


if __name__ == "__main__":
    main()