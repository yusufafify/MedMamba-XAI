# Interpretable Mamba Models for High-Fidelity Medical Image Classification

> Leveraging Selective State-Space Models (VMamba) with explainable AI for robust, multi-task medical image classification on MedMNIST+ benchmarks.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

1. [Motivation](#motivation)
2. [Architecture Overview](#architecture-overview)
3. [Datasets](#datasets)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Configuration System](#configuration-system)
7. [Project Structure](#project-structure)
8. [Citation](#citation)
9. [License](#license)

---

## Motivation

Vision Transformers (ViTs) have set the state-of-the-art in many medical imaging tasks, but their **O(n²)** self-attention cost becomes prohibitive at high resolutions. Selective State-Space Models (SSMs), exemplified by **Mamba**, offer **linear-time** sequence modelling with strong long-range dependency capture. This project adapts the **VMamba** architecture—originally designed for natural images—to the medical imaging domain and pairs it with a novel **SSM-GradCAM** explainability pipeline so clinicians can inspect _why_ a model makes a particular prediction.

Key advantages over ViT baselines:

| Property | ViT | VMamba (ours) |
|---|---|---|
| Sequence complexity | O(n²) | **O(n)** |
| Global receptive field | ✓ (via attention) | ✓ (via state-space) |
| Multi-task loss | Fixed weights | **Kendall uncertainty weighting** |
| Domain separation | None | **SupCon domain projector** |
| Task routing at inference | Requires task label | **Autonomous (prototype 1-NN)** |
| Interpretable gradients | Attention rollout | **Dual-pathway SSM-GradCAM** |

---

## Architecture Overview

### Forward Pass + Contrastive Training Loop

````
  ┌──────────────────────────── TRAINING  (one batch) ──────────────────────────────────┐
  │                                                                                       │
  │  Mixed batch  {images (B,3,224,224), labels (B,), task_ids (B,)}                    │
  │  ── samples from all 4 datasets are interleaved in each batch ──                     │
  │          │                                                                            │
  │          ▼                                                                            │
  │  ┌──────────────────────── SHARED VMamba BACKBONE ───────────────────────────────┐   │
  │  │                                                                                │   │
  │  │  PatchEmbed + LayerNorm                                                        │   │
  │  │    Conv2d(stride=8)  →  (B, 784, 96)   [28×28 tokens]                         │   │
  │  │         │                                                                      │   │
  │  │         ▼  2 × VSSBlock                                                        │   │
  │  │    Stage 0  ──────────────────────────  (B, 784,  96)  [28×28]                │   │
  │  │         │  PatchMerging (÷2 spatial, ×2 channels)                              │   │
  │  │         ▼  2 × VSSBlock                                                        │   │
  │  │    Stage 1  ──────────────────────────  (B, 196, 192)  [14×14]                │   │
  │  │         │  PatchMerging                                                         │   │
  │  │         ▼  6 × VSSBlock                                                        │   │
  │  │    Stage 2  ──────────────────────────  (B,  49, 384)  [ 7× 7]                │   │
  │  │         │  PatchMerging  (odd H/W padded before strided slice)                  │   │
  │  │         ▼  2 × VSSBlock                                                        │   │
  │  │    Stage 3  ──────────────────────────  (B,  16, 768)  [ 4× 4]                │   │
  │  │         │                                                                      │   │
  │  │         ▼  LayerNorm  →  mean(dim=1)  [GAP]                                   │   │
  │  │    features  ─────────────────────────  (B, 768)                               │   │
  │  └──────────────────────────────────┬─────────────────────────────────────────────┘  │
  │                                     │                                                 │
  │           ┌─────────────────────────┴──────────────────────────┐                     │
  │           │                                                      │                    │
  │           ▼  task_id mask routes each sample                     ▼  (all samples)    │
  │  ┌─────────────────────────────────┐      ┌───────────────────────────────────────┐  │
  │  │    TASK HEADS (classification)   │      │  DOMAIN PROJECTOR  (SupCon branch)    │  │
  │  │                                  │      │                                        │  │
  │  │  PathMNIST : LN→Drop→Linear→(B,9)│      │  Linear(768→512) → BN1d → ReLU       │  │
  │  │  DermaMNIST: LN→Drop→Linear→(B,7)│      │  → Linear(512→128) → L2-norm         │  │
  │  │  BloodMNIST: LN→Drop→Linear→(B,8)│      │  → z  (B, 128)                       │  │
  │  │  OCTMNIST  : LN→Drop→Linear→(B,4)│      │                                        │  │
  │  │                                  │      │  ⚠  active only after warmup (ep ≥ 10) │  │
  │  └────────────────┬─────────────────┘      └──────────────────────┬─────────────────┘  │
  │                   │                                                 │                   │
  │                   ▼                                                 ▼                   │
  │  ┌────────────────────────────────┐      ┌──────────────────────────────────────────┐  │
  │  │  KENDALL MULTI-TASK LOSS        │      │  SUPCON DOMAIN LOSS  (Khosla 2020)       │  │
  │  │                                 │      │                                           │  │
  │  │  L_i = CE(logits_i, labels_i)  │      │  Positives = same task_id in the batch   │  │
  │  │                                 │      │  Negatives = different task_id           │  │
  │  │  L_K = Σ_i  (1/σ²_i)·L_i       │      │                                           │  │
  │  │           + log(σ_i)            │      │  L_SC(i) = -1/|P(i)| ·                  │  │
  │  │                                 │      │    Σ_{p∈P(i)} log[                       │  │
  │  │  σ_i is learnable per task —    │      │      exp(z_i·z_p / τ) /                  │  │
  │  │  high uncertainty → low weight  │      │      Σ_{a≠i} exp(z_i·z_a / τ) ]         │  │
  │  │  (Kendall et al., CVPR 2018)    │      │                                           │  │
  │  │                                 │      │  τ = 0.07,  cast to float32 (AMP-safe)   │  │
  │  └──────────────┬──────────────────┘      └─────────────────────┬─────────────────────┘  │
  │                 │                                                 │                   │
  │                 │       L_total = L_K  +  λ · L_SC               │                   │
  │                 └─────────────────────┬───────────────────────────┘                   │
  │                                       │  λ = 0.1 (contrastive_lambda)                │
  │                                       │  L_SC gated off for epoch < 10 (warmup)      │
  │                                       ▼                                               │
  │                                  .backward()                                          │
  │                                       │                                               │
  │                    ┌──────────────────┼──────────────────┐                           │
  │                    ▼                  ▼                   ▼                           │
  │            backbone (lr=1×)   projector (lr=2×)   heads + σ (lr=5× / 1×)            │
  └───────────────────────────────────────────────────────────────────────────────────────┘

  ── AFTER TRAINING ──────────────────────────────────────────────────────────────────────

  compute_prototypes(train_loader):
    for each task_id  →  mean backbone features  →  domain_prototypes (4, 768) [saved in ckpt]

  ── INFERENCE  (no task_id required) ────────────────────────────────────────────────────

    image  →  backbone  →  features (1, 768)
                                │
                                ▼  cosine_similarity(features, domain_prototypes)
                           argmax  →  task_name
                                │
                                ▼
                           task_head  →  class_idx,  confidence
````

### Loss Functions

**Kendall Multi-Task Loss** — each task's cross-entropy is weighted by a learnable $\sigma_i$. Tasks the model finds harder get larger $\sigma_i$ and lower effective weight automatically:

$$\mathcal{L}_\text{Kendall} = \sum_{i} \frac{1}{\sigma_i^2} \mathcal{L}_i^{CE} + \log \sigma_i$$

**Supervised Contrastive Domain Loss** — applied to L2-normalised projections $\mathbf{z} \in \mathbb{R}^{128}$ with temperature $\tau = 0.07$. All images from the same dataset are positives; images from other datasets are negatives. This forces the backbone to produce modality-discriminative representations alongside class-discriminative ones:

$$\mathcal{L}_\text{SupCon}(i) = -\frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \neq i} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$$

**Combined objective:**

$$\mathcal{L} = \mathcal{L}_\text{Kendall} + \lambda \cdot \mathcal{L}_\text{SupCon}, \quad \lambda = 0.1$$

The SupCon term is gated off for the first 10 epochs (configurable via `contrastive_warmup`) so the classification heads can stabilise before the contrastive gradient reaches the backbone.

### Dual-Pathway XAI (SSM-GradCAM)

Beyond standard GradCAM, the model produces **two complementary saliency maps** per image using the `[B, N, D]` SSM activations of the last VSSBlock:

| Map | Gradient target | What it highlights |
|---|---|---|
| **Class saliency** | `logits[predicted_class]` | Local lesion / class-specific structure |
| **Domain saliency** | `cos(features, prototype[domain])` | Global texture / acquisition signatures |

When class and domain disentanglement is effective, these two maps highlight *different* spatial regions — the empirical divergence between them is the scientific finding.

---

## Datasets

All datasets are sourced from **MedMNIST+ v2** at 224×224 resolution.

| Dataset | Modality | Classes | Train Size | Imbalance Note |
|---|---|---|---|---|
| PathMNIST | Colon Pathology | 9 | 89,996 | Moderate class imbalance |
| DermaMNIST | Dermatoscopy | 7 | 7,007 | Severe class imbalance (melanoma under-represented) |
| BloodMNIST | Blood Cell Microscopy | 8 | 11,959 | Near-balanced |
| OCTMNIST | Retinal OCT | 4 | 97,477 | Moderate class imbalance |

---

## Installation

### Prerequisites

- Python ≥ 3.9
- CUDA ≥ 11.8 (for GPU training)

### Steps

```bash
# Clone the repository
git clone https://github.com/yusufafify/MedMamba-XAI.git
cd MedMamba-XAI

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install the package in editable mode
pip install -e .

# Install the Mamba CUDA kernels (required)
pip install mamba-ssm --no-build-isolation

# Copy and fill in the environment variables
cp .env.example .env
````

> **Note:** `mamba-ssm` requires a CUDA-capable GPU and may need to be compiled from source on some systems. See the [mamba-ssm repository](https://github.com/state-spaces/mamba) for details.

---

## Usage

### (a) Single-Dataset Training

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data configs/data/pathmnist.yaml \
    --model configs/model/vmamba_tiny.yaml \
    --training configs/training/single_task.yaml
```

### (b) Multi-Task Training (standard)

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data configs/data/multitask.yaml \
    --model configs/model/vmamba_small.yaml \
    --training configs/training/multi_task.yaml
```

### (c) Multi-Task Contrastive Training

Adds the Supervised Contrastive Domain Loss (SupCon) on top of the Kendall multi-task loss. After training, domain prototypes are computed automatically and baked into the checkpoint, enabling autonomous inference without a `task_id`.

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data configs/data/multitask.yaml \
    --model configs/model/vmamba_small.yaml \
    --training configs/training/multi_task_contrastive.yaml
```

Key config knobs in `configs/training/multi_task_contrastive.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `use_contrastive` | `true` | Enable SupCon domain loss |
| `contrastive_lambda` | `0.1` | Weight of contrastive term relative to Kendall loss |
| `contrastive_temp` | `0.07` | SupCon temperature $\tau$ |
| `contrastive_warmup` | `10` | Epochs before contrastive term is activated |
| `compute_prototypes_after_training` | `true` | Auto-compute and save domain prototypes post-training |

### (d) Recompute Domain Prototypes (optional)

If you need to recompute prototypes for an existing checkpoint (e.g. after fine-tuning):

```bash
python scripts/recompute_prototypes.py \
    --checkpoint outputs/best_model.pt \
    --data configs/data/multitask.yaml
```

### (e) Autonomous Prediction (no task_id required)

```bash
python scripts/predict.py \
    --checkpoint outputs/best_model.pt \
    --image path/to/sample.png
```

The model routes the image to the correct task head via cosine similarity against the stored domain prototypes, then outputs the task name, predicted class, and softmax confidence.

### (f) Test-Set Evaluation

```bash
python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint outputs/best_model.pt \
    --data configs/data/pathmnist.yaml
```

### (g) XAI — Dual-Pathway Saliency

```bash
python scripts/explainability.py \
    --checkpoint outputs/best_model.pt \
    --image path/to/sample.png \
    --output outputs/xai/
```

Produces both the **class saliency map** (what drives the class prediction) and the **domain saliency map** (what drives the modality assignment) side-by-side. For a combined overlay visualisation:

```bash
python scripts/visualize_dual_saliency.py \
    --checkpoint outputs/best_model.pt \
    --image path/to/sample.png \
    --output outputs/xai/
```

---

## Configuration System

The project uses a **YAML-based** hierarchical config system. Configs are merged in priority order:

1. `configs/default.yaml` — base defaults
2. `configs/model/<variant>.yaml` — model-size overrides
3. `configs/data/<dataset>.yaml` — dataset-specific settings
4. `configs/training/<mode>.yaml` — training-regime settings

To switch model sizes, simply change the `--model` flag:

```bash
# Tiny (≈ 28M params)
--model configs/model/vmamba_tiny.yaml

# Small (≈ 50M params)
--model configs/model/vmamba_small.yaml

# Base (≈ 89M params)
--model configs/model/vmamba_base.yaml
```

---

## Project Structure

```
MedMamba-XAI/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── configs/
│   ├── default.yaml
│   ├── model/           (vmamba_tiny, vmamba_small, vmamba_base, resnet50, vit)
│   ├── data/            (pathmnist, dermamnist, bloodmnist, octmnist, multitask)
│   └── training/        (single_task, multi_task, multi_task_contrastive)
├── src/medical_mamba/
│   ├── data/            (dataset, transforms, samplers, constants)
│   ├── models/          (backbone, blocks, heads, medical_vmamba,
│   │                     resnet_baseline, vit_baseline)
│   ├── training/        (trainer, losses, metrics, schedulers)
│   │                     — KendallMultiTaskLoss + ContrastiveDomainLoss
│   ├── xai/             (gradcam, dual_saliency, visualize)
│   └── utils/           (checkpoint, logging, seed)
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py                  ← autonomous inference (no task_id)
│   ├── recompute_prototypes.py     ← re-compute domain prototypes
│   ├── explainability.py
│   ├── visualize_dual_saliency.py  ← side-by-side class + domain maps
│   ├── finetune_contrastive.py
│   ├── eval_ood.py
│   └── explode_npz.py
├── notebooks/           (00_eda … 07_xai_analysis, kaggle_finetune)
├── tests/               (test_dataset, test_model, test_losses, test_xai)
├── runs/                (gitignored — TensorBoard / metric CSVs)
├── outputs/             (gitignored — checkpoints)
└── docs/                (architecture.md, xai_methodology.md)
```

