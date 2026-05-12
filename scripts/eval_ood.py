"""Out-of-distribution rejection via prototype-distance thresholding.

The architecture's domain prototypes give us a free OOD detector. For an
input image:

    score(x) = max over d of  cos(features(x), prototype[d])

Higher score → image looks like a known modality. Lower score → no
prototype is close → reject as out-of-distribution.

This script evaluates that detector against three baselines on a binary
(in-distribution vs OOD) detection problem:

  1. **prototype_max_sim**  — our score (highest cos-sim to any prototype)
  2. **softmax_max_prob**   — max softmax probability over the most-confident
                              task head (standard OOD baseline, MSP)
  3. **feature_norm**       — ||backbone(x)||₂ (some methods use this)
  4. **energy**             — -logsumexp(logits) over the most-confident head
                              (Liu et al. 2020, NeurIPS — Energy-based OOD)

Outputs:
  - AUROC + FPR@95TPR for each method
  - ROC curves overlaid on one figure
  - Score histogram (ID vs OOD) for the prototype method
  - Per-method JSON report

Usage
-----
    python scripts/eval_ood.py \\
        --checkpoint runs/medical_mamba/checkpoint_best.pt \\
        --in_data    dataset/                              \\
        --ood_dir    /path/to/chest_xray/test/             \\
        --output_dir runs/medical_mamba/eval_ood/

Note: works with the contrastive-trained checkpoint as-is. Use try-all-norms
preprocessing (per-dataset normalisation, like training) for ID images;
average normalisation for OOD images so they don't get unfairly penalised
or rewarded by being normalised with one specific dataset's stats.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from medical_mamba.data.constants import DATASET_META
from medical_mamba.data.dataset import MedMNISTFolder
from medical_mamba.data.transforms import get_val_transforms
from medical_mamba.models.medical_vmamba import build_model


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class FlatOODFolder(Dataset):
    """Loads any folder of images (recursive) with a fixed transform."""

    def __init__(self, folder: Path, transform: T.Compose, max_samples: Optional[int] = None):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        paths = sorted(p for p in Path(folder).rglob("*")
                       if p.is_file() and p.suffix.lower() in exts)
        if max_samples is not None and len(paths) > max_samples:
            rng = np.random.default_rng(42)
            keep = rng.choice(len(paths), max_samples, replace=False)
            paths = [paths[i] for i in sorted(keep)]
        self.paths = paths
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return {"image": self.transform(img)}


def averaged_val_transform(img_size: int = 224) -> T.Compose:
    """Used for OOD images, where modality is unknown by definition."""
    means = [DATASET_META[n]["mean"] for n in DATASET_META]
    stds  = [DATASET_META[n]["std"]  for n in DATASET_META]
    avg_mean = [sum(m[i] for m in means) / len(means) for i in range(3)]
    avg_std  = [sum(s[i] for s in stds)  / len(stds)  for i in range(3)]
    return T.Compose([
        T.Resize(256), T.CenterCrop(img_size), T.ToTensor(),
        T.Normalize(mean=avg_mean, std=avg_std),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_loader(model, loader, device, task_names, desc) -> dict:
    """Compute all 4 OOD-detection scores on a loader.

    Returns a dict with arrays for each method.
    """
    proto_sim, msp, feat_norm, energy = [], [], [], []

    for batch in tqdm(loader, desc=desc):
        images = batch["image"].to(device, non_blocking=True)
        feats, _ = model.backbone(images)                      # (B, feat_dim)

        # 1. max prototype similarity
        sim = F.cosine_similarity(
            feats.unsqueeze(1),
            model.domain_prototypes.unsqueeze(0),
            dim=-1,
        )                                                      # (B, n_tasks)
        proto_sim.append(sim.max(dim=-1).values.cpu().numpy())

        # 2 & 4. softmax max + energy across all task heads
        head_max_prob = []
        head_logsumexp = []
        for name in task_names:
            logits = model.heads[name](feats)
            head_max_prob.append(logits.softmax(dim=-1).max(dim=-1).values)
            head_logsumexp.append(logits.logsumexp(dim=-1))
        head_max_prob = torch.stack(head_max_prob, dim=1)      # (B, n_tasks)
        head_logsumexp = torch.stack(head_logsumexp, dim=1)
        # Take the most-confident task for each sample
        msp.append(head_max_prob.max(dim=-1).values.cpu().numpy())
        # Energy: lower energy = more in-distribution. Convention: negate so
        # higher score = more in-distribution (matches the other methods).
        energy.append(head_logsumexp.max(dim=-1).values.cpu().numpy())

        # 3. feature norm
        feat_norm.append(feats.norm(dim=1).cpu().numpy())

    return {
        "prototype_max_sim":    np.concatenate(proto_sim),
        "softmax_max_prob":     np.concatenate(msp),
        "feature_norm":         np.concatenate(feat_norm),
        "energy_lse":           np.concatenate(energy),
    }


def auroc_and_fpr95(in_scores: np.ndarray, ood_scores: np.ndarray) -> dict:
    y_true = np.concatenate([np.ones_like(in_scores), np.zeros_like(ood_scores)])
    scores = np.concatenate([in_scores, ood_scores])
    auroc = roc_auc_score(y_true, scores)
    fpr, tpr, thr = roc_curve(y_true, scores)
    # FPR @ 95% TPR
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr_95 = float(fpr[idx])
    threshold_95 = float(thr[idx])
    return {
        "auroc":         float(auroc),
        "fpr_at_95_tpr": fpr_95,
        "threshold_at_95_tpr": threshold_95,
        "in_mean":   float(in_scores.mean()),  "in_std":  float(in_scores.std()),
        "ood_mean":  float(ood_scores.mean()), "ood_std": float(ood_scores.std()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--in_data",    required=True,
                   help="Root containing the 4 known dataset folders (in-distribution)")
    p.add_argument("--ood_dir",    required=True,
                   help="Folder of OOD images (any modality the model wasn't trained on)")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--max_ood",    type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir) if args.output_dir else (Path(args.checkpoint).parent / "eval_ood")
    out_dir.mkdir(parents=True, exist_ok=True)

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
        print("WARNING: prototypes_computed=False — prototype score will be meaningless.")
        print("Run scripts/recompute_prototypes_per_dataset.py first.")

    # ── ID loader: per-dataset normalisation (matches training distribution) ──
    in_data = Path(args.in_data)
    in_sets = [
        MedMNISTFolder(
            dataset_name=name,
            root=str(in_data / f"{name}_dataset"),
            split="test",
            transform=get_val_transforms(name, 224),
            task_id=tid,
        )
        for tid, name in enumerate(task_names)
    ]
    in_loader = DataLoader(
        ConcatDataset(in_sets), batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    print(f"ID samples : {sum(len(d) for d in in_sets):,}")

    # ── OOD loader: averaged normalisation (modality unknown by definition) ──
    ood_loader = DataLoader(
        FlatOODFolder(Path(args.ood_dir), averaged_val_transform(), args.max_ood),
        batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    print(f"OOD samples: {len(ood_loader.dataset):,}  (from {args.ood_dir})")

    if len(ood_loader.dataset) == 0:
        print("ERROR: no OOD images found.")
        return

    # ── Score both sets ──────────────────────────────────────────────────
    in_scores  = score_loader(model, in_loader,  device, task_names, "ID")
    ood_scores = score_loader(model, ood_loader, device, task_names, "OOD")

    # ── Compute metrics ──────────────────────────────────────────────────
    methods = ["prototype_max_sim", "softmax_max_prob", "energy_lse", "feature_norm"]
    pretty = {
        "prototype_max_sim": "Prototype distance (ours)",
        "softmax_max_prob":  "Max softmax probability",
        "energy_lse":        "Energy (logsumexp)",
        "feature_norm":      "Feature L2 norm",
    }
    results = {m: auroc_and_fpr95(in_scores[m], ood_scores[m]) for m in methods}

    # ── Print report ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'Method':<32s} {'AUROC':>8s} {'FPR@95':>8s} "
          f"{'ID mean':>10s} {'OOD mean':>10s}")
    print("=" * 78)
    for m in methods:
        r = results[m]
        print(f"{pretty[m]:<32s} {r['auroc']:>8.4f} {r['fpr_at_95_tpr']:>8.4f} "
              f"{r['in_mean']:>10.4f} {r['ood_mean']:>10.4f}")
    print("=" * 78)

    # ── Plots ────────────────────────────────────────────────────────────
    # ROC overlay
    fig, ax = plt.subplots(figsize=(7, 6))
    for m in methods:
        y_true = np.concatenate([np.ones_like(in_scores[m]), np.zeros_like(ood_scores[m])])
        scores = np.concatenate([in_scores[m], ood_scores[m]])
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, lw=2, label=f"{pretty[m]} (AUROC={results[m]['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("OOD Detection — In-Distribution vs OOD (ROC)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roc.png", dpi=130, bbox_inches="tight")
    plt.close()

    # Score histogram (prototype method only — the headline)
    fig, ax = plt.subplots(figsize=(8, 5))
    in_s  = in_scores["prototype_max_sim"]
    ood_s = ood_scores["prototype_max_sim"]
    bins = np.linspace(min(in_s.min(), ood_s.min()),
                       max(in_s.max(), ood_s.max()), 60)
    ax.hist(in_s,  bins=bins, alpha=0.6, label="ID  (in-distribution)",  color="#1f77b4", density=True)
    ax.hist(ood_s, bins=bins, alpha=0.6, label="OOD (out-of-distribution)", color="#d62728", density=True)
    # Mark threshold at 95% TPR
    thr = results["prototype_max_sim"]["threshold_at_95_tpr"]
    ax.axvline(thr, color="black", ls="--", lw=1.5,
               label=f"τ@95%TPR = {thr:.3f}")
    ax.set_xlabel("max cos-sim to domain prototypes")
    ax.set_ylabel("density")
    ax.set_title(f"Prototype-distance OOD score  |  AUROC = {results['prototype_max_sim']['auroc']:.3f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "score_histogram.png", dpi=130, bbox_inches="tight")
    plt.close()

    # ── Save results ─────────────────────────────────────────────────────
    full_results = {
        "ood_source": str(args.ood_dir),
        "n_id":  int(len(in_loader.dataset)),
        "n_ood": int(len(ood_loader.dataset)),
        "task_names": task_names,
        "methods": results,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\nROC      : {out_dir / 'roc.png'}")
    print(f"Histogram: {out_dir / 'score_histogram.png'}")
    print(f"Metrics  : {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
