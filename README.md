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
7. [Expected Results](#expected-results)
8. [Project Structure](#project-structure)
9. [Citation](#citation)
10. [License](#license)

---

## Motivation

Vision Transformers (ViTs) have set the state-of-the-art in many medical imaging tasks, but their **O(n²)** self-attention cost becomes prohibitive at high resolutions. Selective State-Space Models (SSMs), exemplified by **Mamba**, offer **linear-time** sequence modelling with strong long-range dependency capture. This project adapts the **VMamba** architecture—originally designed for natural images—to the medical imaging domain and pairs it with a novel **SSM-GradCAM** explainability pipeline so clinicians can inspect *why* a model makes a particular prediction.

Key advantages over ViT baselines:

| Property | ViT | VMamba (ours) |
|---|---|---|
| Sequence complexity | O(n²) | **O(n)** |
| Global receptive field | ✓ (via attention) | ✓ (via state-space) |
| Interpretable gradients | Attention rollout | **SSM-GradCAM** |
| Multi-task friendly | ✓ | ✓ |

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                   MedicalVMamba  Pipeline                          │
│                                                                    │
│   Input (3×224×224)                                                │
│       │                                                            │
│       ▼                                                            │
│   ┌──────────────┐                                                 │
│   │  PatchEmbed   │  → (B, C₀, H/4, W/4)                          │
│   └──────┬───────┘                                                 │
│          ▼                                                         │
│   ┌──────────────────┐                                             │
│   │ Stage 1: VSSBlock │  × N₁   → (B, C₁, H/8, W/8)              │
│   │  + PatchMerging   │                                            │
│   └──────┬───────────┘                                             │
│          ▼                                                         │
│   ┌──────────────────┐                                             │
│   │ Stage 2: VSSBlock │  × N₂   → (B, C₂, H/16, W/16)            │
│   │  + PatchMerging   │                                            │
│   └──────┬───────────┘                                             │
│          ▼                                                         │
│   ┌──────────────────┐                                             │
│   │ Stage 3: VSSBlock │  × N₃   → (B, C₃, H/32, W/32)            │
│   │  + PatchMerging   │                                            │
│   └──────┬───────────┘                                             │
│          ▼                                                         │
│   ┌──────────────────┐                                             │
│   │ Stage 4: VSSBlock │  × N₄   → (B, C₄, H/32, W/32)            │
│   └──────┬───────────┘                                             │
│          ▼                                                         │
│   ┌──────────────┐                                                 │
│   │     GAP       │  → (B, C₄)                                     │
│   └──────┬───────┘                                                 │
│          ▼                                                         │
│   ┌──────────────────────────────────┐                             │
│   │  Task Heads (one per dataset)     │                            │
│   │  PathMNIST │ Derma │ Blood │ OCT  │                            │
│   └──────────────────────────────────┘                             │
└────────────────────────────────────────────────────────────────────┘
```

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
git clone https://github.com/<your-username>/MedMamba-XAI.git
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
```

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

### (b) Multi-Task Training

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data configs/data/multitask.yaml \
    --model configs/model/vmamba_small.yaml \
    --training configs/training/multi_task.yaml
```

### (c) Test-Set Evaluation

```bash
python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint outputs/best_model.pt \
    --data configs/data/pathmnist.yaml
```

### (d) XAI Visualization

```bash
python scripts/explainability.py \
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

## Expected Results

Reported on MedMNIST+ v2 test splits (224×224, single-task, VMamba-Tiny):

| Dataset | Accuracy (%) | F1-Macro (%) | AUC (%) |
|---|---|---|---|
| PathMNIST | 88 – 92 | 85 – 90 | 96 – 98 |
| DermaMNIST | 74 – 78 | 70 – 75 | 90 – 93 |
| BloodMNIST | 95 – 97 | 94 – 96 | 99+ |
| OCTMNIST | 76 – 80 | 73 – 78 | 95 – 97 |

> These ranges are indicative targets based on comparable architectures. Exact numbers depend on hyperparameter tuning.

---

## Project Structure

```
MedMamba-XAI/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── configs/
│   ├── default.yaml
│   ├── model/           (vmamba_tiny, vmamba_small, vmamba_base)
│   ├── data/            (pathmnist, dermamnist, bloodmnist, octmnist, multitask)
│   └── training/        (single_task, multi_task)
├── src/medical_mamba/
│   ├── data/            (dataset, transforms, samplers, constants)
│   ├── models/          (backbone, blocks, heads, medical_vmamba)
│   ├── training/        (trainer, losses, metrics, schedulers)
│   ├── xai/             (gradcam, visualize)
│   └── utils/           (checkpoint, logging, seed)
├── scripts/             (train, evaluate, explainability, explode_npz)
├── notebooks/           (00_eda, 01_data_pipeline, 02_architecture, 03_xai)
├── tests/               (test_dataset, test_model, test_losses, test_xai)
├── outputs/             (gitignored — run artifacts)
└── docs/                (architecture.md, xai_methodology.md)
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{medmamba_xai_2025,
    title   = {Interpretable Mamba Models for High-Fidelity Medical Image Classification},
    author  = {<Your Name>},
    year    = {2025},
    note    = {Graduate Research Project},
    url     = {https://github.com/<your-username>/MedMamba-XAI}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
