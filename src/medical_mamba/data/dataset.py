"""
dataset.py — MedMNIST+ Lazy-Loading Dataset with Class Imbalance Handling
--------------------------------------------------------------------------
Supports TWO folder layouts automatically detected at runtime:

Layout A — class-subfolder (our explode_npz.py output):
    <root>/<split>/<class_idx>/<image>.jpg

Layout B — flat with label-in-filename (your actual data):
    <root>/<split>/<index>_<label>.png   e.g. 1042_3.png
    <root>/<split>/<index>_<label>.jpg   e.g. 1042_3.jpg

Detection logic: if the split directory contains image files directly
(no integer-named subdirectories), Layout B is assumed and the label
is parsed from the filename suffix ``_{label}.ext``.

Both .jpg and .png extensions are supported in both layouts.

Design decisions
----------------
- Folder-based (not .npz) loading: avoids the RAM wall for 12+ GB archives.
- External ``transform`` injection: callers (e.g. transforms.py) own the
  augmentation logic; this class stays a pure data source.
- Dict return type ``{image, label, task_id}``: downstream model and trainer
  code can always find ``task_id`` without tuple unpacking conventions.
- Dual-level WeightedRandomSampler: balances class imbalance *within* each
  dataset AND dataset-level frequency imbalance *across* datasets.
- Epsilon guard on sampler weights: prevents division-by-zero for classes
  that may be missing in a split (rare but possible in small val/test sets).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from medical_mamba.data.constants import DATASET_META


# ─────────────────────────────────────────────────────────────────────────────
# Single-dataset
# ─────────────────────────────────────────────────────────────────────────────

class MedMNISTFolder(Dataset):
    """Single MedMNIST dataset loaded lazily from an exploded JPEG folder.

    Expected folder layout::

        root/
          train/
            0/          ← integer class index
              img_000.jpg
              ...
            1/
            ...
          val/
          test/

    Parameters
    ----------
    dataset_name:
        One of ``pathmnist``, ``dermamnist``, ``bloodmnist``, ``octmnist``.
        Must be a key in ``DATASET_META``.
    root:
        Path to the dataset root directory (contains ``train/``, ``val/``,
        ``test/`` sub-directories).
    split:
        One of ``"train"``, ``"val"``, ``"test"``.
    transform:
        Torchvision-style callable applied to each ``PIL.Image`` before it is
        returned.  Pass ``None`` only for debugging — production code should
        always supply a transform (see ``medical_mamba.data.transforms``).
    task_id:
        Integer identifier for this dataset in multi-task mode.  Embedded in
        every returned batch dict so the model knows which head to route to.
    """

    def __init__(
        self,
        dataset_name: str,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        task_id: int = 0,
    ) -> None:
        super().__init__()

        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        if dataset_name.lower() not in DATASET_META:
            raise KeyError(
                f"Unknown dataset '{dataset_name}'. "
                f"Valid options: {list(DATASET_META.keys())}"
            )

        self.dataset_name = dataset_name.lower()
        self.root         = Path(root) / split
        self.split        = split
        self.transform    = transform
        self.task_id      = task_id
        self.meta         = DATASET_META[self.dataset_name]
        self.num_classes: int = self.meta["num_classes"]

        self.samples: List[Tuple[Path, int]] = self._build_sample_list()

        if not self.samples:
            raise RuntimeError(
                f"No images found under {self.root}. "
                "Run scripts/explode_npz.py first and verify the folder layout."
            )

        self._label_counts: Counter = Counter(lbl for _, lbl in self.samples)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_sample_list(self) -> List[Tuple[Path, int]]:
        """Walk the split directory and collect (path, label) pairs.

        Detects layout automatically:

        Layout A — integer-named class subdirectories:
            train/0/img_000.jpg  or  train/0/img_000.png

        Layout B — flat directory, label encoded in filename:
            train/1042_3.png  ->  label 3
            train/0055_0.jpg  ->  label 0

        Both .jpg and .png are supported in both layouts.
        """
        samples: List[Tuple[Path, int]] = []

        # Detect layout: Layout A has integer-named subdirectories
        has_class_dirs = any(
            p.is_dir() and p.name.isdigit()
            for p in self.root.iterdir()
        )

        if has_class_dirs:
            # Layout A: class-subfolder hierarchy
            for class_dir in sorted(self.root.iterdir()):
                if not class_dir.is_dir():
                    continue
                try:
                    label = int(class_dir.name)
                except ValueError:
                    continue
                for ext in ("*.jpg", "*.jpeg", "*.png"):
                    for img_path in sorted(class_dir.glob(ext)):
                        samples.append((img_path, label))
        else:
            # Layout B: flat, label encoded as {idx}_{label}.ext
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in sorted(self.root.glob(ext)):
                    stem = img_path.stem          # e.g. "1042_3"
                    parts = stem.rsplit("_", 1)   # split on LAST underscore
                    if len(parts) != 2:
                        continue
                    try:
                        label = int(parts[1])
                    except ValueError:
                        continue
                    samples.append((img_path, label))

        return samples

    # ------------------------------------------------------------------ #
    #  Sampler support                                                     #
    # ------------------------------------------------------------------ #

    def get_sampler_weights(self) -> torch.Tensor:
        """Per-sample weights for ``WeightedRandomSampler``.

        Weight = ``1 / (class_count + ε)`` so minority classes are sampled
        more often.  The ``ε = 1e-6`` guard prevents division-by-zero for
        classes absent from this split.

        Returns
        -------
        torch.Tensor
            Shape ``(len(self),)``, dtype ``float32``.
        """
        class_weights = {
            cls: 1.0 / (count + 1e-6)
            for cls, count in self._label_counts.items()
        }
        # Fill missing classes (absent from this split) with the smallest weight
        min_w = min(class_weights.values()) if class_weights else 1e-6
        weights = torch.tensor(
            [class_weights.get(lbl, min_w) for _, lbl in self.samples],
            dtype=torch.float32,
        )
        return weights

    def get_sampler(self, replacement: bool = True) -> WeightedRandomSampler:
        """Return a ready-to-use ``WeightedRandomSampler`` for this dataset."""
        weights = self.get_sampler_weights()
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=replacement,
        )

    # ------------------------------------------------------------------ #
    #  Dataset interface                                                   #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, label = self.samples[idx]

        # PIL is more robust than torchvision.io for JPEG edge-cases
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return {
            "image":   img,                                    # [3, H, W] after transform
            "label":   torch.tensor(label, dtype=torch.long),
            "task_id": torch.tensor(self.task_id, dtype=torch.long),
        }

    def __repr__(self) -> str:
        dist = dict(sorted(self._label_counts.items()))
        return (
            f"MedMNISTFolder("
            f"dataset={self.dataset_name}, split={self.split}, "
            f"n={len(self.samples)}, class_dist={dist})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-task wrapper
# ─────────────────────────────────────────────────────────────────────────────

class MultiTaskMedDataset(ConcatDataset):
    """Concatenates multiple :class:`MedMNISTFolder` datasets for multi-task training.

    Each sample carries its own ``task_id`` so the model can route it to the
    correct classification head.

    The dual-level sampler produced by :meth:`get_sampler` simultaneously:

    1. Balances class imbalance *within* each dataset (via per-class weights).
    2. Balances dataset-level frequency *across* datasets — each dataset
       contributes roughly equally regardless of its raw size.

    Parameters
    ----------
    datasets:
        Ordered list of :class:`MedMNISTFolder` instances.  The position in
        this list must match each dataset's ``task_id`` attribute.
    """

    def __init__(self, datasets: List[MedMNISTFolder]) -> None:
        super().__init__(datasets)
        self.sub_datasets: List[MedMNISTFolder] = datasets

    def get_sampler(self, replacement: bool = True) -> WeightedRandomSampler:
        """Dual-level ``WeightedRandomSampler`` across all sub-datasets.

        Within each sub-dataset the weights are normalised to sum to 1, giving
        every dataset equal total probability mass regardless of size.  The
        per-class imbalance correction is preserved within each block.
        """
        all_weights: List[torch.Tensor] = []
        for ds in self.sub_datasets:
            w = ds.get_sampler_weights()
            w = w / (w.sum() + 1e-12)   # normalise to unit mass per dataset
            all_weights.append(w)

        combined = torch.cat(all_weights, dim=0)
        return WeightedRandomSampler(
            weights=combined,
            num_samples=len(combined),
            replacement=replacement,
        )


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    dataset_roots: Dict[str, str],
    transforms_map: Dict[str, Dict[str, Callable]],
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    single_dataset: Optional[str] = None,
) -> Dict[str, DataLoader]:
    """Build ``train`` / ``val`` / ``test`` DataLoaders.

    Parameters
    ----------
    dataset_roots:
        Mapping ``{dataset_name: folder_path}``.
        Example: ``{"pathmnist": "D:/medmnist/pathmnist"}``.
    transforms_map:
        Nested mapping ``{dataset_name: {split: transform}}``.
        Produced by ``medical_mamba.data.transforms.build_transforms_map()``.
        Separating transform construction from dataset construction keeps this
        factory free of augmentation logic.
    batch_size:
        Samples per batch (per GPU).
    num_workers:
        DataLoader worker processes.  Set to ``0`` on Windows if you encounter
        multiprocessing errors with ``spawn`` start method.
    pin_memory:
        Enable for CUDA training to speed up host→device transfer.
    single_dataset:
        If given, train/evaluate on this dataset only (ignores the rest of
        ``dataset_roots``).  The project's "safety net" mode.

    Returns
    -------
    Dict[str, DataLoader]
        Keys: ``"train"``, ``"val"``, ``"test"``.
    """
    if single_dataset is not None:
        if single_dataset not in dataset_roots:
            raise KeyError(
                f"'{single_dataset}' not found in dataset_roots. "
                f"Available: {list(dataset_roots.keys())}"
            )
        active: Dict[str, str] = {single_dataset: dataset_roots[single_dataset]}
    else:
        active = dataset_roots

    loaders: Dict[str, DataLoader] = {}

    for split in ("train", "val", "test"):
        is_train = split == "train"

        sub_datasets: List[MedMNISTFolder] = [
            MedMNISTFolder(
                dataset_name=name,
                root=path,
                split=split,
                transform=transforms_map[name][split],
                task_id=tid,
            )
            for tid, (name, path) in enumerate(active.items())
        ]

        if len(sub_datasets) == 1:
            dataset: Dataset = sub_datasets[0]
            sampler = sub_datasets[0].get_sampler() if is_train else None
        else:
            dataset = MultiTaskMedDataset(sub_datasets)
            sampler = dataset.get_sampler() if is_train else None  # type: ignore[union-attr]

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            # shuffle is mutually exclusive with sampler
            shuffle=(sampler is None and is_train),
            num_workers=num_workers,
            pin_memory=pin_memory,
            # drop_last avoids a tiny final batch that can destabilise BatchNorm
            drop_last=is_train,
            persistent_workers=(num_workers > 0),
        )

    return loaders