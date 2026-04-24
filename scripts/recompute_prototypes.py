"""Recompute domain prototypes using the same normalisation as predict.py.

The prototypes baked into the checkpoint by Trainer.fit() are computed with
per-dataset normalisation (each sample normalised with its own dataset's
mean/std).  predict.py uses averaged normalisation at autonomous inference
because the modality is unknown.  This mismatch causes the nearest-prototype
routing to fail.

This script rebuilds the prototypes using the same averaged normalisation that
predict.py applies, then overwrites the checkpoint so routing works correctly.

Usage
-----
    python scripts/recompute_prototypes.py \\
        --checkpoint runs/medical_mamba/checkpoint_best.pt \\
        --data_root  dataset/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from medical_mamba.data.constants import DATASET_META
from medical_mamba.models.medical_vmamba import build_model


# ─────────────────────────────────────────────────────────────────────────────
# Averaged transform — must match predict.py exactly
# ─────────────────────────────────────────────────────────────────────────────

def averaged_val_transform(img_size: int = 224) -> T.Compose:
    means = [DATASET_META[n]["mean"] for n in DATASET_META]
    stds  = [DATASET_META[n]["std"]  for n in DATASET_META]
    avg_mean = [sum(m[i] for m in means) / len(means) for i in range(3)]
    avg_std  = [sum(s[i] for s in stds)  / len(stds)  for i in range(3)]
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=avg_mean, std=avg_std),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Minimal flat-folder dataset — handles both layout A and B
# ─────────────────────────────────────────────────────────────────────────────

from PIL import Image
from torch.utils.data import Dataset


class FlatImageDataset(Dataset):
    """All images in a directory, labelled by filename suffix (*_{label}.ext)."""

    def __init__(self, folder: Path, task_id: int, transform: T.Compose) -> None:
        exts = {".jpg", ".jpeg", ".png"}
        self.paths = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )
        self.task_id = task_id
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return {
            "image":   self.transform(img),
            "label":   torch.tensor(0, dtype=torch.long),   # not needed here
            "task_id": torch.tensor(self.task_id, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute domain prototypes with averaged normalisation"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="checkpoint_best.pt to patch in-place")
    parser.add_argument("--data_root",  required=True,
                        help="Root containing <dataset>_dataset/ folders")
    parser.add_argument("--split",      default="train",
                        help="Which split to average over (default: train)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers",type=int, default=4)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    data_root = Path(args.data_root)
    transform = averaged_val_transform()

    # ── Load checkpoint + rebuild model ──────────────────────────────────
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    task_names = state["task_names"]
    cfg        = state["config"]

    task_configs = [(n, DATASET_META[n]["num_classes"]) for n in task_names]
    model = build_model(
        task_configs = task_configs,
        model_size   = cfg.get("model_size",   "tiny"),
        patch_size   = cfg.get("patch_size",   8),
        head_dropout = cfg.get("head_dropout", 0.1),
    )
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.to(device).eval()

    print(f"Loaded model: {len(task_names)} tasks — {task_names}")
    print(f"Using averaged normalisation (same as predict.py)")

    # ── Build dataloaders for each task using averaged normalisation ──────
    loaders = []
    for tid, name in enumerate(task_names):
        folder = data_root / f"{name}_dataset" / args.split
        if not folder.exists():
            print(f"  WARNING: {folder} not found — skipping {name}")
            loaders.append(None)
            continue
        ds = FlatImageDataset(folder, task_id=tid, transform=transform)
        loader = DataLoader(ds, batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False, pin_memory="cuda" in args.device)
        loaders.append(loader)
        print(f"  {name}: {len(ds)} images from {folder}")

    # ── Combine all loaders into one iterable ─────────────────────────────
    def combined():
        for loader in loaders:
            if loader is not None:
                yield from loader

    # ── Recompute ─────────────────────────────────────────────────────────
    print("\nRecomputing prototypes...")
    model.compute_prototypes(combined(), device=device)

    print("\nNew prototypes (L2 norms — should all be > 0):")
    for i, name in enumerate(task_names):
        norm = model.domain_prototypes[i].norm().item()
        print(f"  {name}: norm={norm:.4f}")

    # ── Patch and save checkpoint ─────────────────────────────────────────
    state["model_state_dict"] = model.state_dict()
    state["prototypes_computed"] = True
    torch.save(state, args.checkpoint)
    print(f"\nCheckpoint updated: {args.checkpoint}")
    print("Prototype routing should now work correctly with predict.py.")


if __name__ == "__main__":
    main()
