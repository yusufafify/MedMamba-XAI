"""Per-dataset metadata constants — single source of truth for MedMNIST+.

Merge decisions vs the two versions
-------------------------------------
description  : Kept from agent — useful for logging and EDA notebooks.
in_channels  : Kept from agent (cleaner than our ``n_channels``).
train_size   : Kept from agent — used in EDA and progress logging.
mean / std   : Kept from our version for OCTMNIST.  Agent had a
               single-element list ``[0.1916]`` which crashes
               ``transforms.Normalize`` after grayscale→RGB conversion.
               All datasets keep 3-channel means so the pipeline is
               uniform regardless of source modality.
task/modality: Dropped (not used downstream).

IMPORTANT — OCTMNIST note
--------------------------
OCTMNIST source images are grayscale (1 channel).  The transform pipeline
converts them to RGB via ``x.repeat(3, 1, 1)`` AFTER ``ToTensor``.
Therefore mean/std must have 3 values matching the replicated channels.
"""

from __future__ import annotations

from typing import Any, Dict

DATASET_META: Dict[str, Dict[str, Any]] = {
    "pathmnist": {
        "description": "Colon Pathology — 9 tissue types",
        "num_classes": 9,
        "in_channels": 3,
        "mean": [0.7405, 0.5330, 0.7058],
        "std":  [0.1237, 0.1768, 0.1244],
        "train_size": 89_996,
    },
    "dermamnist": {
        "description": "Dermatoscopy — 7 skin-lesion categories",
        "num_classes": 7,
        "in_channels": 3,
        "mean": [0.7632, 0.5381, 0.5614],
        "std":  [0.1419, 0.1528, 0.1692],
        "train_size": 7_007,
    },
    "bloodmnist": {
        "description": "Blood Cell Microscopy — 8 cell types",
        "num_classes": 8,
        "in_channels": 3,
        "mean": [0.7943, 0.6597, 0.6962],
        "std":  [0.2153, 0.2424, 0.1978],
        "train_size": 11_959,
    },
    "octmnist": {
        "description": "Retinal OCT — 4 pathology classes",
        "num_classes": 4,
        "in_channels": 1,                    # source is grayscale
        # 3-channel values after grayscale→RGB repeat in transform pipeline
        "mean": [0.1889, 0.1889, 0.1889],
        "std":  [0.1973, 0.1973, 0.1973],
        "train_size": 97_477,
    },
}

# Ordered list — position = default task_id when all datasets are active
DATASET_NAMES = list(DATASET_META.keys())
