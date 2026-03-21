# Architecture: VMamba Adaptation for 2D Medical Vision

## Overview

This document describes how the VMamba (Visual State-Space Model) architecture
is adapted from natural-image classification to high-fidelity medical image
analysis on MedMNIST+ datasets.

---

## Background: From Mamba to VMamba

**Mamba** is a Selective State-Space Model (SSM) for 1-D sequences that achieves
linear-time complexity in sequence length while retaining strong long-range
dependency modelling. The key insight is *input-dependent* (selective) gating
of the continuous state-space transition matrices (A, B, C, Δ).

**VMamba** extends Mamba to 2-D images by:

1. **Patch Embedding** — Splitting the image into non-overlapping patches
   (analogous to ViT) and projecting each patch to a hidden dimension.
2. **Cross-Scan Module (CSM)** — Serialising the 2-D token grid into *four*
   1-D sequences (row-major, row-major reversed, column-major, column-major
   reversed), processing each with the SSM, and aggregating results. This
   ensures every spatial position has access to context from all four
   cardinal directions.
3. **Hierarchical stages** — Mirror the multi-scale structure of CNNs /
   Swin Transformers with PatchMerging layers between stages.

---

## Our Adaptation for Medical Imaging

### Modality-Aware Input Handling

Medical datasets span various modalities:

| Dataset     | Channels | Domain           |
|-------------|----------|------------------|
| PathMNIST   | 3 (RGB)  | Colon pathology   |
| DermaMNIST  | 3 (RGB)  | Dermatoscopy      |
| BloodMNIST  | 3 (RGB)  | Blood microscopy  |
| OCTMNIST    | 1 (Gray) | Retinal OCT       |

Grayscale images (OCTMNIST) are replicated to 3 channels at the transform
level so that a single backbone handles all datasets uniformly.

### Backbone: VMambaBackbone

```
Input (B, 3, 224, 224)
    ↓
PatchEmbed(patch_size=4)  →  (B, C₀, 56, 56)
    ↓
Stage 1: VSSBlock × N₁   →  (B, C₁, 28, 28)   via PatchMerging
Stage 2: VSSBlock × N₂   →  (B, C₂, 14, 14)   via PatchMerging
Stage 3: VSSBlock × N₃   →  (B, C₃,  7,  7)   via PatchMerging
Stage 4: VSSBlock × N₄   →  (B, C₄,  7,  7)   (no merging)
    ↓
LayerNorm + GAP           →  (B, C₄)
```

### VSSBlock Internals

Each VSSBlock performs:

1. **LayerNorm** on the input
2. **Linear projection** into a gated + SSM branch (2× expansion)
3. **Depthwise convolution** for local feature mixing
4. **Cross-Scan Aggregation** — four directional scans, averaged
5. **SiLU gating** from the gate branch
6. **Linear projection** back to the original dimension
7. **Residual connection** with stochastic depth

### Multi-Task Heads

A shared backbone feeds into separate `ClassificationHead` modules — one per
dataset. Each head is `LayerNorm → Dropout → Linear(C₄, num_classes)`.

In multi-task training, each mini-batch contains samples from multiple
datasets; the model's `forward_multitask` method routes samples to the correct
head based on `task_id`.

### Stochastic Depth

Drop-path rates increase linearly across blocks (from 0 to `drop_path_rate`),
following the convention from DeiT / Swin Transformer.

---

## Model Configurations

| Variant | Depths         | Dims                  | ~Params | VRAM (est.) |
|---------|----------------|-----------------------|---------|-------------|
| Tiny    | [2, 2, 9, 2]  | [96, 192, 384, 768]   | 28M     | ~10 GB      |
| Small   | [2, 2, 27, 2] | [96, 192, 384, 768]   | 50M     | ~16 GB      |
| Base    | [2, 2, 27, 2] | [128, 256, 512, 1024] | 89M     | ~24 GB      |

---

## Implementation Notes

- The current `VSSBlock` uses a **reference Python implementation** of the
  cross-scan aggregation. For production speed, swap in the fused CUDA kernel
  from the [`mamba-ssm`](https://github.com/state-spaces/mamba) package.
- Intermediate feature maps are returned by the backbone to support the
  SSM-GradCAM explainability pipeline (see `docs/xai_methodology.md`).
