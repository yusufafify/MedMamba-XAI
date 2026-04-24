"""Autonomous inference script — no task_id required at inference time.

Uses the prototype-routing feature (domain discovery via Supervised
Contrastive Learning) to identify which medical imaging modality an
image belongs to, then runs the matching classification head.

Usage
-----
Single image, auto-detect modality::

    python scripts/predict.py \\
        --checkpoint runs/medical_mamba/checkpoint_best.pt \\
        --image      test_images/unknown_slide.jpg

Batch of images from a folder::

    python scripts/predict.py \\
        --checkpoint runs/medical_mamba/checkpoint_best.pt \\
        --image_dir  test_images/ \\
        --output     predictions.csv

Single image with known task (bypass auto-detection)::

    python scripts/predict.py \\
        --checkpoint runs/medical_mamba/checkpoint_best.pt \\
        --image      test_images/skin_lesion.jpg \\
        --task       dermamnist
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

from medical_mamba.data.constants import DATASET_META
from medical_mamba.data.transforms import get_val_transforms
from medical_mamba.models.medical_vmamba import build_model


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def _averaged_val_transform(img_size: int = 224) -> T.Compose:
    """Val transform with normalisation averaged across all 4 datasets.

    Per-dataset normalisation is impossible at autonomous inference because
    we don't yet know the modality. The contrastive-trained backbone has
    learned modality-invariant features large enough to dwarf this
    distribution shift, so prototype routing still separates cleanly.
    """
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


def _load_image(path: Path, transform: T.Compose) -> torch.Tensor:
    """Read an image from disk and apply the inference transform."""
    img = Image.open(path).convert("RGB")
    return transform(img)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def _build_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
):
    """Reconstruct a MedicalVMamba from a saved checkpoint.

    Reads ``task_names`` and ``config`` from the checkpoint, rebuilds the
    model with matching architecture, loads state_dict, and restores the
    ``prototypes_computed`` flag.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    task_names = state["task_names"]
    cfg        = state["config"]

    task_configs = [
        (name, DATASET_META[name]["num_classes"])
        for name in task_names
    ]

    model = build_model(
        task_configs = task_configs,
        model_size   = cfg.get("model_size",   "tiny"),
        patch_size   = cfg.get("patch_size",   8),
        head_dropout = cfg.get("head_dropout", 0.1),
    )
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.prototypes_computed = state.get("prototypes_computed", False)
    model.to(device).eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _predict_with_task(
    model,
    image: torch.Tensor,
    task_name: str,
) -> Tuple[str, int, float]:
    """Run inference with the caller-specified task (bypass prototype routing)."""
    if task_name not in model.heads:
        raise KeyError(
            f"Unknown task '{task_name}'. "
            f"Registered tasks: {model.task_names}"
        )
    features, _ = model.backbone(image)
    logits = model.heads[task_name](features)
    probs  = logits.softmax(dim=-1)
    class_idx  = int(logits.argmax(dim=-1).item())
    confidence = float(probs.max().item())
    return task_name, class_idx, confidence


def _predict_one(
    model,
    img_tensor: torch.Tensor,
    task_override: Optional[str],
    device: torch.device,
) -> Tuple[str, int, float]:
    """Run a single prediction. Routes via prototypes unless --task is given."""
    img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
    if task_override is not None:
        return _predict_with_task(model, img_tensor, task_override)

    if not model.prototypes_computed:
        raise RuntimeError(
            "This checkpoint has no domain prototypes — autonomous "
            "routing is unavailable. Either pass --task <name> to bypass "
            "routing, or retrain with "
            "training.compute_prototypes_after_training=true."
        )
    return model.predict(img_tensor)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _collect_images(image_dir: Path) -> List[Path]:
    """Return all supported image files in ``image_dir`` (non-recursive)."""
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous MedicalVMamba inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint_best.pt")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",     type=str, help="Single image path")
    src.add_argument("--image_dir", type=str, help="Folder of images (batch mode)")
    parser.add_argument("--output",  type=str, default="predictions.csv",
                        help="Output CSV path (batch mode). Default: predictions.csv")
    parser.add_argument("--task",    type=str, default=None,
                        help="Optional task override — bypasses prototype routing")
    parser.add_argument("--device",  type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model  = _build_model_from_checkpoint(args.checkpoint, device)

    # --task → use that dataset's exact val transforms (matches training
    # distribution). Autonomous → averaged normalisation.
    if args.task is not None:
        transform = get_val_transforms(args.task, img_size=224)
    else:
        transform = _averaged_val_transform(img_size=224)

    if args.image is not None:
        img_path = Path(args.image)
        tensor   = _load_image(img_path, transform)
        task_name, class_idx, confidence = _predict_one(
            model, tensor, args.task, device
        )
        print(f"File       : {img_path}")
        print(f"Task       : {task_name}")
        print(f"Class      : {class_idx}")
        print(f"Confidence : {confidence:.4f}")
        return

    # Batch folder mode
    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {image_dir}")
    paths = _collect_images(image_dir)
    if not paths:
        raise FileNotFoundError(f"No .jpg/.jpeg/.png files in {image_dir}")

    out_path = Path(args.output)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_task", "predicted_class", "confidence"])
        for p in paths:
            tensor = _load_image(p, transform)
            task_name, class_idx, confidence = _predict_one(
                model, tensor, args.task, device
            )
            writer.writerow([p.name, task_name, class_idx, f"{confidence:.6f}"])
            print(f"{p.name:40s} → {task_name:12s} class={class_idx:<2d} conf={confidence:.4f}")

    print(f"\nWrote {len(paths)} predictions to {out_path}")


if __name__ == "__main__":
    main()
