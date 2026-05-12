"""Generate dual-pathway saliency visualisations.

For a set of sample images, produces a comparison grid showing:
  - the original image
  - the class-saliency overlay (what made it look like the predicted class)
  - the domain-saliency overlay (what made it look like the predicted modality)
  - the signed difference map (class - domain) — visualises where the two
    pathways DISAGREE, which is the empirical evidence of disentanglement

Two run modes:

    # One representative image per modality, auto-selected from the test set:
    python scripts/visualize_dual_saliency.py \\
        --checkpoint runs/medical_mamba/checkpoint_best.pt \\
        --data_root  dataset/ \\
        --output_dir runs/medical_mamba/dual_saliency/

    # User-specified images:
    python scripts/visualize_dual_saliency.py \\
        --checkpoint runs/medical_mamba/checkpoint_best.pt \\
        --images dataset/dermamnist_dataset/test/0_5.jpg \\
                 dataset/pathmnist_dataset/test/100_2.jpg \\
                 dataset/octmnist_dataset/test/50_3.jpg

Outputs:
    grid.png          — N-row × 4-column comparison
    metadata.json     — per-image predictions + similarity values
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from medical_mamba.data.constants import DATASET_META
from medical_mamba.data.transforms import get_val_transforms
from medical_mamba.models.medical_vmamba import build_model
from medical_mamba.xai.dual_saliency import DualPathwaySaliency


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root",  default=None,
                   help="If --images is not given, sample one test image per modality from here")
    p.add_argument("--images",     nargs="*", default=None,
                   help="Explicit list of image paths to visualise")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--target_stage", type=int, default=3,
                   help="Backbone stage to hook (default 3 = last)")
    p.add_argument("--alpha",        type=float, default=0.45,
                   help="Saliency overlay opacity (0..1)")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = {
    "pathmnist":  ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"],
    "dermamnist": ["AK", "BCC", "BKL", "DF", "MEL", "NV", "VASC"],
    "bloodmnist": ["baso", "eosi", "eryt", "ig", "lymp", "mono", "neut", "plat"],
    "octmnist":   ["CNV", "DME", "DRUSEN", "NORMAL"],
}


def parse_dataset_from_path(path: Path) -> Optional[str]:
    """Detect which modality an image belongs to from its parent dirs."""
    parts = {p.lower() for p in path.parts}
    for name in DATASET_META:
        if f"{name}_dataset" in parts or name in parts:
            return name
    return None


def sample_one_per_modality(data_root: Path, seed: int) -> List[Path]:
    """One random test image per known modality."""
    rng = random.Random(seed)
    picks: List[Path] = []
    for name in DATASET_META:
        test_dir = data_root / f"{name}_dataset" / "test"
        if not test_dir.exists():
            continue
        candidates = [
            p for p in test_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if candidates:
            picks.append(rng.choice(candidates))
    return picks


def denormalize(tensor: torch.Tensor, mean, std) -> np.ndarray:
    """Reverse a Normalize() transform → HWC image in [0, 1]."""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t  = torch.tensor(std).view(3, 1, 1)
    img = (tensor.cpu() * std_t + mean_t).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def overlay(img_hwc: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] HWC image with a [0,1] saliency map using jet colormap."""
    cmap = plt.get_cmap("jet")
    heat = cmap(cam)[..., :3]                # (H, W, 3) RGB
    return (1 - alpha) * img_hwc + alpha * heat


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # ── Load model ───────────────────────────────────────────────────────
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    task_names = state["task_names"]
    cfg = state.get("config", state.get("args", {}))

    task_configs = [(n, DATASET_META[n]["num_classes"]) for n in task_names]
    model = build_model(
        task_configs=task_configs,
        model_size=cfg.get("model_size", "tiny"),
        patch_size=cfg.get("patch_size", 8),
        head_dropout=cfg.get("head_dropout", 0.1),
    )
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.prototypes_computed = state.get("prototypes_computed", False)
    model.to(device).eval()

    if not model.prototypes_computed:
        print("ERROR: prototypes_computed=False. Run scripts/recompute_prototypes_per_dataset.py first.")
        return

    saliency = DualPathwaySaliency(model, target_stage=args.target_stage)

    # ── Resolve image set ────────────────────────────────────────────────
    if args.images:
        image_paths = [Path(p) for p in args.images]
    else:
        if args.data_root is None:
            print("ERROR: provide either --images or --data_root.")
            return
        image_paths = sample_one_per_modality(Path(args.data_root), args.seed)
        print(f"Auto-sampled {len(image_paths)} test images, one per modality.")

    if not image_paths:
        print("ERROR: no images selected.")
        return

    out_dir = Path(args.output_dir) if args.output_dir else (Path(args.checkpoint).parent / "dual_saliency")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # ── Per-image dual saliency ──────────────────────────────────────────
    n = len(image_paths)
    fig, axes = plt.subplots(n, 4, figsize=(4 * 3.5, n * 3.5))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    metadata: list = []

    for i, img_path in enumerate(image_paths):
        # Pick a per-dataset transform if we can identify the modality from the
        # path; otherwise default to pathmnist's transform (averaged would be
        # better but the per-dataset one matches training distribution).
        ds_name = parse_dataset_from_path(img_path)
        if ds_name is None:
            ds_name = task_names[0]  # fall back to first known modality
            print(f"  [warn] could not detect modality for {img_path.name}, using {ds_name}'s transform")
        transform = get_val_transforms(ds_name, 224)
        mean = DATASET_META[ds_name]["mean"]
        std  = DATASET_META[ds_name]["std"]

        pil = Image.open(img_path).convert("RGB")
        x = transform(pil).unsqueeze(0).to(device)        # (1, 3, 224, 224)

        # Compute both maps
        result = saliency.both(x)
        cls_map = result.class_map.numpy()
        dom_map = result.domain_map.numpy()
        diff    = cls_map - dom_map

        # Denormalised image for display
        img_hwc = denormalize(x.squeeze(0), mean, std)

        # ── Plot row ─────────────────────────────────────────────────────
        # 1: original
        axes[i, 0].imshow(img_hwc)
        axes[i, 0].set_title(f"Input ({ds_name})", fontsize=10)
        axes[i, 0].axis("off")

        # 2: class overlay
        cls_label = CLASS_NAMES[result.predicted_task][result.predicted_class] \
                    if result.predicted_task in CLASS_NAMES else str(result.predicted_class)
        axes[i, 1].imshow(overlay(img_hwc, cls_map, args.alpha))
        axes[i, 1].set_title(
            f"Class saliency\n→ {result.predicted_task}/{cls_label}\n"
            f"conf={result.class_confidence:.2f}",
            fontsize=9,
        )
        axes[i, 1].axis("off")

        # 3: domain overlay
        axes[i, 2].imshow(overlay(img_hwc, dom_map, args.alpha))
        axes[i, 2].set_title(
            f"Domain saliency\n→ {result.predicted_task}\n"
            f"sim={result.domain_similarity:.2f}",
            fontsize=9,
        )
        axes[i, 2].axis("off")

        # 4: signed difference (red = class-only, blue = domain-only)
        diff_norm = diff / (np.abs(diff).max() + 1e-8)
        im = axes[i, 3].imshow(diff_norm, cmap="seismic", vmin=-1, vmax=1)
        axes[i, 3].set_title("Class - Domain\n(red=class, blue=domain)", fontsize=9)
        axes[i, 3].axis("off")

        metadata.append({
            "path":              str(img_path),
            "true_modality":     ds_name,
            "predicted_task":    result.predicted_task,
            "predicted_class":   result.predicted_class,
            "class_label":       cls_label,
            "class_confidence":  result.class_confidence,
            "domain_similarity": result.domain_similarity,
            "domain_routing_correct": result.predicted_task == ds_name,
            "saliency_disagreement": float(np.abs(diff).mean()),
        })

        print(f"  [{i+1}/{n}] {img_path.name}: true={ds_name}, "
              f"pred={result.predicted_task}/{cls_label}, "
              f"sim={result.domain_similarity:.3f}, conf={result.class_confidence:.3f}, "
              f"|class-domain|={np.abs(diff).mean():.3f}")

    plt.suptitle("Dual-pathway saliency: class vs domain explanations", fontsize=12, y=1.005)
    plt.tight_layout()
    grid_path = out_dir / "grid.png"
    plt.savefig(grid_path, dpi=140, bbox_inches="tight")
    plt.close()

    # ── Save metadata ────────────────────────────────────────────────────
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Aggregate stat: average saliency disagreement ────────────────────
    mean_disag = np.mean([m["saliency_disagreement"] for m in metadata])
    print(f"\nGrid: {grid_path}")
    print(f"Metadata: {out_dir / 'metadata.json'}")
    print(f"Mean |class - domain| saliency disagreement: {mean_disag:.4f}")
    print("(higher = the two pathways highlight different pixels — empirical")
    print(" evidence of class/domain disentanglement in the backbone)")


if __name__ == "__main__":
    main()
