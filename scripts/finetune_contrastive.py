"""Contrastive domain fine-tuning for an existing checkpoint.

Problem being solved
--------------------
A model trained with ``contrastive_lambda=0.1`` and ``contrastive_temp=0.07``
produces prototype routing accuracy of only 65% (pathmnist: 48.9%) because
the SupCon gradient reaching the backbone was too weak to meaningfully
separate modality features in the raw feature space.

This script loads such a checkpoint and runs a focused fine-tuning pass:
  - Classification heads and Kendall log_sigma are FROZEN.
  - Only backbone + domain_projector are updated.
  - Loss is pure SupCon (ContrastiveDomainLoss), no CE.
  - Higher lambda and temperature correct the gradient magnitude.
  - Prototype recomputation happens automatically at the end.
  - The updated weights are saved back into the same checkpoint file
    (or a new path via --output).

Typical runtime: ~20 epochs × 7 min = ~2.5 h (same GPU setup as training).
Expected routing improvement: pathmnist 48.9% → 70%+.

Usage
-----
    python scripts/finetune_contrastive.py \\
        --checkpoint runs/medical_mamba/checkpoint_best.pt \\
        --data_root  dataset/ \\
        --epochs     20 \\
        --lambda     0.5 \\
        --temp       0.15 \\
        --lr         5e-5

    # Save to a separate file instead of overwriting:
    python scripts/finetune_contrastive.py \\
        --checkpoint runs/medical_mamba/checkpoint_best.pt \\
        --data_root  dataset/ \\
        --output     runs/medical_mamba/checkpoint_contrastive.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from medical_mamba.data.constants import DATASET_META
from medical_mamba.data.dataset import build_dataloaders
from medical_mamba.data.transforms import build_transforms_map
from medical_mamba.models.medical_vmamba import build_model
from medical_mamba.training.losses import ContrastiveDomainLoss
from medical_mamba.utils.seed import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Contrastive domain fine-tuning to improve prototype routing."
    )
    p.add_argument("--checkpoint",   required=True,
                   help="Path to trained checkpoint (.pt)")
    p.add_argument("--data_root",    required=True,
                   help="Dataset root containing <name>_dataset/ folders")
    p.add_argument("--output",       default=None,
                   help="Where to save the updated checkpoint. "
                        "Defaults to overwriting --checkpoint.")
    p.add_argument("--epochs",       type=int,   default=20,
                   help="Fine-tuning epochs (default 20)")
    p.add_argument("--lr",           type=float, default=5e-5,
                   help="Backbone learning rate (default 5e-5)")
    p.add_argument("--lam",          type=float, default=0.5, dest="contrastive_lambda",
                   help="SupCon loss weight (default 0.5)")
    p.add_argument("--temp",         type=float, default=0.15,
                   help="SupCon temperature (default 0.15)")
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[finetune] {msg}", flush=True)


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def _unfreeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    ckpt_path = Path(args.checkpoint)
    out_path  = Path(args.output) if args.output else ckpt_path

    # ── Load checkpoint ──────────────────────────────────────────────────
    _log(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    task_names: List[str] = state["task_names"]
    cfg = state.get("config", state.get("args", {}))

    task_configs = [(n, DATASET_META[n]["num_classes"]) for n in task_names]
    model = build_model(
        task_configs=task_configs,
        model_size=cfg.get("model_size", "tiny"),
        patch_size=cfg.get("patch_size", 8),
        head_dropout=cfg.get("head_dropout", 0.1),
    )
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.to(device)

    _log(f"Tasks: {task_names}")
    _log(f"Original routing accuracy (from confusion matrix notebook): "
         "see notebook section 11 for baseline")

    # ── Freeze everything except backbone + domain_projector ─────────────
    # Classification heads (self.heads) and Kendall log_sigma must stay
    # frozen so we don't damage the classification F1 scores.
    for name, module in model.named_children():
        if name in ("backbone", "domain_projector"):
            _unfreeze(module)
            _log(f"  TRAINABLE : {name}")
        else:
            _freeze(module)
            _log(f"  frozen    : {name}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params) / 1e6
    _log(f"Trainable parameters: {n_trainable:.1f}M")

    # ── Data ─────────────────────────────────────────────────────────────
    data_root = Path(args.data_root)
    dataset_roots = {
        name: str(data_root / f"{name}_dataset") for name in task_names
    }
    for name, path in dataset_roots.items():
        assert Path(path).exists(), f"Dataset folder missing: {path}"

    transforms_map = build_transforms_map(task_names)
    loaders = build_dataloaders(
        dataset_roots=dataset_roots,
        transforms_map=transforms_map,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    _log(f"Train samples: {len(loaders['train'].dataset):,}")

    # ── Loss + Optimizer ──────────────────────────────────────────────────
    criterion = ContrastiveDomainLoss(temperature=args.temp).to(device)
    _log(f"SupCon | lambda={args.contrastive_lambda} | temp={args.temp}")

    # Separate LRs: projector learns 2× faster than backbone so the
    # SupCon gradient can reach the backbone more aggressively.
    optimizer = optim.AdamW([
        {"params": list(model.backbone.parameters()),
         "lr": args.lr, "weight_decay": 0.05},
        {"params": list(model.domain_projector.parameters()),
         "lr": args.lr * 2.0, "weight_decay": 0.0},
    ])

    # Cosine decay over the fine-tuning window (no warmup — model is pre-trained)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    # ── Fine-tuning loop ──────────────────────────────────────────────────
    best_loss = float("inf")
    best_state: dict = {}

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches  = 0
        skipped    = 0

        pbar = tqdm(loaders["train"], desc=f"Ep {epoch+1:02d}/{args.epochs}", leave=False)
        for batch in pbar:
            images   = batch["image"].to(device, non_blocking=True)
            task_ids = batch["task_id"].to(device, non_blocking=True)

            # Need ≥2 distinct tasks in batch for SupCon to be defined.
            if task_ids.unique().numel() < 2:
                skipped += 1
                continue

            # backbone → project → L2-normalised embeddings
            features, _ = model.backbone(images)
            projections = model.project(features.float())

            loss = args.contrastive_lambda * criterion(projections, task_ids)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        elapsed  = time.time() - t0
        lr_now   = optimizer.param_groups[0]["lr"]
        _log(
            f"Ep {epoch+1:02d}/{args.epochs} | "
            f"loss={avg_loss:.4f} | lr={lr_now:.2e} | "
            f"{elapsed:.0f}s | skipped_batches={skipped}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Snapshot only the fine-tuned params — we'll merge back below
            best_state = {
                k: v.clone()
                for k, v in model.state_dict().items()
                if "backbone" in k or "domain_projector" in k
            }
            _log(f"  ↑ new best loss")

    _log(f"Fine-tuning done. Best SupCon loss: {best_loss:.4f}")

    # ── Recompute prototypes from training set ────────────────────────────
    # Load best backbone/projector weights back into model before computing
    # prototypes so we get the best-separated feature space.
    _log("Loading best checkpoint weights before computing prototypes...")
    current_sd = model.state_dict()
    current_sd.update(best_state)
    model.load_state_dict(current_sd)

    _log("Computing domain prototypes from training set...")
    model.eval()
    model.compute_prototypes(loaders["train"], device)
    _log("Prototypes computed.")

    # ── Save updated checkpoint ───────────────────────────────────────────
    # Merge updated backbone + projector + new prototypes back into the
    # original checkpoint dict so all other fields (epoch, best_avg_f1,
    # task_names, config) are preserved.
    state["model_state_dict"] = model.state_dict()
    state["prototypes_computed"] = True
    state["finetune_contrastive"] = {
        "epochs":   args.epochs,
        "lambda":   args.contrastive_lambda,
        "temp":     args.temp,
        "best_loss": best_loss,
    }

    torch.save(state, out_path)
    _log(f"Saved updated checkpoint → {out_path}")
    _log("Re-run notebook section 11 (prototype routing confusion matrix) to verify improvement.")


if __name__ == "__main__":
    main()
