"""XAI visualisation entry point — SSM-GradCAM for MedicalVMamba.

Merge decisions
---------------
Args           : Combined both versions.  Website had ``--task``, ``--image``,
                 ``--class_idx``, ``--n_samples``, ``--dataset_root``.
                 Agent had ``--dataset``, ``--stage``, ``--output``,
                 ``--class-idx``.  Merged and disambiguated.

Modes          : Three modes controlled by args:
                 1. ``--image PATH``       — single image analysis
                 2. ``--batch``            — random samples from a dataset split
                 3. default (folder mode)  — all images in --image if it is a dir

Image display  : Website re-opened files from disk for display (double I/O).
                 Now uses ``denormalize()`` from ``visualize.py`` which works
                 directly on the normalised tensor.

Checkpoint     : ``load_model_from_checkpoint`` (website) reconstructs model
                 from the checkpoint's ``task_names`` + ``config`` fields.
                 No need for a separate config file at inference time.

Usage::

    # Single image — auto-detects predicted class
    python scripts/explainability.py \\
        --checkpoint outputs/checkpoint_best.pt \\
        --task pathmnist \\
        --image D:/medmnist/pathmnist/test/3/img_0042.jpg \\
        --output outputs/xai/

    # Target a specific class
    python scripts/explainability.py \\
        --checkpoint outputs/checkpoint_best.pt \\
        --task pathmnist \\
        --image D:/medmnist/pathmnist/test/3/img_0042.jpg \\
        --class-idx 3

    # Batch mode — random samples from test split
    python scripts/explainability.py \\
        --checkpoint outputs/checkpoint_best.pt \\
        --task pathmnist \\
        --dataset-root D:/medmnist/pathmnist \\
        --batch --n-samples 16 \\
        --output outputs/xai/

    # Inspect an earlier stage (higher spatial resolution)
    python scripts/explainability.py \\
        --checkpoint outputs/checkpoint_best.pt \\
        --task pathmnist \\
        --image path/to/img.jpg \\
        --stage 2
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from medical_mamba.data.constants import DATASET_META
from medical_mamba.data.dataset import MedMNISTFolder
from medical_mamba.utils.seed import set_seed
from medical_mamba.xai.gradcam import SSMGradCAM, load_model_from_checkpoint
from medical_mamba.xai.visualize import (
    denormalize,
    overlay_heatmap,
    save_grid,
    visualize_single,
)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def build_val_transform(task_name: str) -> transforms.Compose:
    """Build the standard val/test transform for a given task."""
    meta = DATASET_META[task_name]
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=meta["mean"], std=meta["std"]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Single-image mode
# ─────────────────────────────────────────────────────────────────────────────

def run_single(
    cam: SSMGradCAM,
    img_path: Path,
    task_name: str,
    class_idx: Optional[int],
    device: str,
    output_dir: Path,
) -> None:
    """Compute and save saliency for one image."""
    transform = build_val_transform(task_name)
    meta      = DATASET_META[task_name]

    pil_img   = Image.open(img_path).convert("RGB")
    tensor    = transform(pil_img).unsqueeze(0).to(device)

    heatmap = cam(tensor, class_idx=class_idx, task_name=task_name)

    # Predicted class + confidence
    with torch.no_grad():
        logits, _ = cam.model.forward_single(tensor, task_name)
        pred_class = int(logits.argmax(dim=1).item())
        pred_conf  = float(torch.softmax(logits, dim=1).max().item())

    # Display-ready image from normalised tensor (no re-open)
    orig_np = denormalize(tensor.squeeze(0).cpu(), meta["mean"], meta["std"])

    save_path = output_dir / f"{img_path.stem}_xai.png"
    visualize_single(
        original_img=orig_np,
        saliency_map=heatmap.numpy(),
        task_name=task_name,
        pred_class=pred_class,
        pred_conf=pred_conf,
        save_path=save_path,
    )
    print(f"  Saved → {save_path}  (pred={pred_class}, conf={pred_conf:.1%})")


# ─────────────────────────────────────────────────────────────────────────────
# Batch mode
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    cam: SSMGradCAM,
    dataset_root: str,
    task_name: str,
    n_samples: int,
    device: str,
    output_dir: Path,
) -> None:
    """Compute saliency for random samples from the test split and save a grid."""
    meta = DATASET_META[task_name]
    transform = build_val_transform(task_name)

    ds = MedMNISTFolder(
        dataset_name=task_name,
        root=dataset_root,
        split="test",
        transform=transform,
        task_id=0,
    )

    indices = torch.randperm(len(ds))[:n_samples].tolist()

    images_np:    List[np.ndarray] = []
    heatmaps_np:  List[np.ndarray] = []
    labels:       List[str]        = []
    correct_flags: List[bool]      = []

    for idx in indices:
        item       = ds[idx]
        true_label = item["label"].item()
        tensor     = item["image"].unsqueeze(0).to(device)

        try:
            heatmap = cam(tensor, task_name=task_name)
        except Exception as exc:
            print(f"  [WARN] Saliency failed for idx={idx}: {exc}")
            continue

        with torch.no_grad():
            logits, _ = cam.model.forward_single(tensor, task_name)
            pred = int(logits.argmax(dim=1).item())

        orig_np = denormalize(tensor.squeeze(0).cpu(), meta["mean"], meta["std"])
        images_np.append(orig_np)
        heatmaps_np.append(heatmap.numpy())
        labels.append(f"GT:{true_label} Pred:{pred}")
        correct_flags.append(pred == true_label)

    grid_path = save_grid(
        images=images_np,
        heatmaps=heatmaps_np,
        output_path=output_dir / f"{task_name}_xai_grid.png",
        labels=labels,
        correct_flags=correct_flags,
    )
    n_correct = sum(correct_flags)
    print(
        f"  Grid saved → {grid_path}  "
        f"({n_correct}/{len(correct_flags)} correct = "
        f"{n_correct/max(len(correct_flags),1):.1%} acc on sample)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SSM-GradCAM Explainability for MedicalVMamba")

    parser.add_argument("--checkpoint",    required=True,
                        help="Path to checkpoint_best.pt")
    parser.add_argument("--task",          required=True,
                        help="Task name (e.g. pathmnist)")
    parser.add_argument("--image",         default=None,
                        help="Path to a single image file")
    parser.add_argument("--batch",         action="store_true",
                        help="Batch mode: visualize N random test samples")
    parser.add_argument("--dataset-root",  default=None,
                        help="Dataset root dir (required for --batch)")
    parser.add_argument("--n-samples",     type=int, default=16,
                        help="Number of samples for batch mode (default 16)")
    parser.add_argument("--class-idx",     type=int, default=None,
                        help="Target class index (default: predicted class)")
    parser.add_argument("--stage",         type=int, default=3,
                        help="Backbone stage to hook (0-3, default 3=last)")
    parser.add_argument("--output",        default="outputs/xai/",
                        help="Output directory")
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",          type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint} ...")
    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    cam   = SSMGradCAM(model, target_stage=args.stage)

    if args.batch:
        if not args.dataset_root:
            parser.error("--dataset-root is required for --batch mode")
        print(f"Batch mode: {args.n_samples} samples from {args.task}/test")
        run_batch(cam, args.dataset_root, args.task,
                  args.n_samples, args.device, output_dir)

    elif args.image:
        img_path = Path(args.image)
        if img_path.is_dir():
            # Folder of images
            img_files = sorted(img_path.glob("*.jpg")) + sorted(img_path.glob("*.png"))
            print(f"Folder mode: {len(img_files)} images")
            for f in img_files:
                run_single(cam, f, args.task, args.class_idx, args.device, output_dir)
        else:
            print(f"Single image: {img_path}")
            run_single(cam, img_path, args.task, args.class_idx, args.device, output_dir)

    else:
        parser.error("Provide either --image PATH or --batch (with --dataset-root)")

    print(f"\n✓ Done. Results saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
