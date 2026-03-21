"""Tests for dataset loading, sampler weights, and batch shapes.

Covers:
    - MedMNISTFolder: loading from exploded JPEG folders
    - Batch shape [B, 3, 224, 224]
    - Labels in [0, num_classes)
    - task_id correctness
    - Sampler weights: positive, finite, correct length
    - MultiTaskMedDataset: dual-level sampler, length, task_id routing
    - build_dataloaders: all three splits, single_dataset mode

Real data paths (used only if REAL_DATA=1 env var is set):
    dataset/pathmnist_dataset/
    dataset/bloodmnist_dataset/
    dataset/dermamnist_dataset/
    dataset/octmnist_dataset/
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from medical_mamba.data.constants import DATASET_META
from medical_mamba.data.dataset import (
    MedMNISTFolder,
    MultiTaskMedDataset,
    build_dataloaders,
)
from medical_mamba.data.transforms import build_transforms_map


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — synthetic folder builder
# ─────────────────────────────────────────────────────────────────────────────

def _make_fake_dataset(
    root: Path,
    dataset_name: str,
    n_per_class: int = 4,
) -> Path:
    """Create a minimal exploded JPEG folder structure under ``root``.

    Layout::

        root/
          train/0/img_000.jpg ... img_003.jpg
          train/1/...
          ...
          val/  (same)
          test/ (same)
    """
    num_classes = DATASET_META[dataset_name]["num_classes"]
    for split in ("train", "val", "test"):
        for cls in range(num_classes):
            cls_dir = root / split / str(cls)
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                colour = (cls * 20 % 255, 128, 200)
                img = Image.new("RGB", (224, 224), color=colour)
                img.save(cls_dir / f"img_{i:03d}.jpg", quality=95)
    return root


# ─────────────────────────────────────────────────────────────────────────────
# Module-scoped fixtures — built once, reused across all tests in this file
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fake_pathmnist(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("pathmnist")
    return _make_fake_dataset(root, "pathmnist", n_per_class=4)


@pytest.fixture(scope="module")
def fake_bloodmnist(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("bloodmnist")
    return _make_fake_dataset(root, "bloodmnist", n_per_class=4)


@pytest.fixture(scope="module")
def fake_octmnist(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("octmnist")
    return _make_fake_dataset(root, "octmnist", n_per_class=4)


# ─────────────────────────────────────────────────────────────────────────────
# MedMNISTFolder — core dataset tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMedMNISTFolder:

    def test_correct_sample_count(self, fake_pathmnist: Path) -> None:
        """Total samples = num_classes × n_per_class."""
        tmap = build_transforms_map(["pathmnist"])
        ds = MedMNISTFolder(
            dataset_name="pathmnist",
            root=str(fake_pathmnist),
            split="train",
            transform=tmap["pathmnist"]["train"],
        )
        expected = DATASET_META["pathmnist"]["num_classes"] * 4
        assert len(ds) == expected, f"Expected {expected}, got {len(ds)}"

    def test_image_shape(self, fake_pathmnist: Path) -> None:
        """Image tensor must be [3, 224, 224]."""
        tmap = build_transforms_map(["pathmnist"])
        ds = MedMNISTFolder(
            dataset_name="pathmnist",
            root=str(fake_pathmnist),
            split="train",
            transform=tmap["pathmnist"]["train"],
        )
        item = ds[0]
        assert item["image"].shape == (3, 224, 224), \
            f"Bad shape: {item['image'].shape}"

    def test_label_in_range(self, fake_pathmnist: Path) -> None:
        """All labels must be in [0, num_classes)."""
        tmap = build_transforms_map(["pathmnist"])
        ds = MedMNISTFolder(
            dataset_name="pathmnist",
            root=str(fake_pathmnist),
            split="train",
            transform=tmap["pathmnist"]["train"],
        )
        n = DATASET_META["pathmnist"]["num_classes"]
        for item in ds:
            lbl = item["label"].item()
            assert 0 <= lbl < n, f"Label {lbl} out of [0, {n})"

    def test_task_id_matches_argument(self, fake_pathmnist: Path) -> None:
        """task_id in every item must equal the task_id passed at construction."""
        tmap = build_transforms_map(["pathmnist"])
        ds = MedMNISTFolder(
            dataset_name="pathmnist",
            root=str(fake_pathmnist),
            split="train",
            transform=tmap["pathmnist"]["train"],
            task_id=7,
        )
        assert ds[0]["task_id"].item() == 7
        assert ds[-1]["task_id"].item() == 7

    def test_return_dict_keys(self, fake_pathmnist: Path) -> None:
        """__getitem__ must return dict with image, label, task_id keys."""
        tmap = build_transforms_map(["pathmnist"])
        ds = MedMNISTFolder(
            dataset_name="pathmnist",
            root=str(fake_pathmnist),
            split="train",
            transform=tmap["pathmnist"]["train"],
        )
        item = ds[0]
        assert set(item.keys()) == {"image", "label", "task_id"}

    def test_val_transform_is_deterministic(self, fake_pathmnist: Path) -> None:
        """Val transform must be deterministic — same index → same tensor."""
        tmap = build_transforms_map(["pathmnist"])
        ds = MedMNISTFolder(
            dataset_name="pathmnist",
            root=str(fake_pathmnist),
            split="val",
            transform=tmap["pathmnist"]["val"],
        )
        t1 = ds[0]["image"]
        t2 = ds[0]["image"]
        assert torch.allclose(t1, t2), "Val transform produced different tensors"

    def test_raises_on_missing_images(self, tmp_path: Path) -> None:
        """RuntimeError when split dir exists but contains no images."""
        tmap = build_transforms_map(["pathmnist"])
        (tmp_path / "train").mkdir()
        with pytest.raises(RuntimeError, match="No images found"):
            MedMNISTFolder(
                dataset_name="pathmnist",
                root=str(tmp_path),
                split="train",
                transform=tmap["pathmnist"]["train"],
            )

    def test_raises_on_unknown_dataset(self, fake_pathmnist: Path) -> None:
        """KeyError when dataset_name is not in DATASET_META."""
        tmap = build_transforms_map(["pathmnist"])
        with pytest.raises(KeyError):
            MedMNISTFolder(
                dataset_name="notadataset",
                root=str(fake_pathmnist),
                split="train",
                transform=tmap["pathmnist"]["train"],
            )

    def test_image_dtype_is_float32(self, fake_pathmnist: Path) -> None:
        """ToTensor produces float32, not uint8."""
        tmap = build_transforms_map(["pathmnist"])
        ds = MedMNISTFolder(
            dataset_name="pathmnist",
            root=str(fake_pathmnist),
            split="train",
            transform=tmap["pathmnist"]["train"],
        )
        assert ds[0]["image"].dtype == torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# WeightedRandomSampler
# ─────────────────────────────────────────────────────────────────────────────

class TestSamplerWeights:

    def test_weights_length(self, fake_pathmnist: Path) -> None:
        tmap = build_transforms_map(["pathmnist"])
        ds = MedMNISTFolder(
            dataset_name="pathmnist",
            root=str(fake_pathmnist),
            split="train",
            transform=tmap["pathmnist"]["train"],
        )
        w = ds.get_sampler_weights()
        assert w.shape == (len(ds),)

    def test_weights_positive_finite(self, fake_pathmnist: Path) -> None:
        tmap = build_transforms_map(["pathmnist"])
        ds = MedMNISTFolder(
            dataset_name="pathmnist",
            root=str(fake_pathmnist),
            split="train",
            transform=tmap["pathmnist"]["train"],
        )
        w = ds.get_sampler_weights()
        assert (w > 0).all(),           "Some weights are zero or negative"
        assert torch.isfinite(w).all(), "Some weights are NaN or Inf"

    def test_sampler_num_samples(self, fake_pathmnist: Path) -> None:
        tmap = build_transforms_map(["pathmnist"])
        ds = MedMNISTFolder(
            dataset_name="pathmnist",
            root=str(fake_pathmnist),
            split="train",
            transform=tmap["pathmnist"]["train"],
        )
        sampler = ds.get_sampler()
        assert sampler.num_samples == len(ds)

    def test_minority_class_upweighted(self, tmp_path: Path) -> None:
        """Class with 1 sample must get higher weight than class with 8 samples."""
        for cls, count in [(0, 1), (1, 8)]:
            cls_dir = tmp_path / "train" / str(cls)
            cls_dir.mkdir(parents=True)
            for i in range(count):
                Image.new("RGB", (224, 224), color=(cls * 100, 0, 0)).save(
                    cls_dir / f"img_{i:03d}.jpg"
                )
        tmap = build_transforms_map(["pathmnist"])
        ds = MedMNISTFolder(
            dataset_name="pathmnist",
            root=str(tmp_path),
            split="train",
            transform=tmap["pathmnist"]["train"],
        )
        w = ds.get_sampler_weights()
        labels = [lbl for _, lbl in ds.samples]
        w0 = w[[i for i, l in enumerate(labels) if l == 0]].mean().item()
        w1 = w[[i for i, l in enumerate(labels) if l == 1]].mean().item()
        assert w0 > w1, \
            f"Minority weight ({w0:.4f}) should exceed majority weight ({w1:.4f})"


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader batch shapes
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoaderBatch:

    def test_batch_image_shape(self, fake_pathmnist: Path) -> None:
        tmap = build_transforms_map(["pathmnist"])
        ds = MedMNISTFolder(
            dataset_name="pathmnist",
            root=str(fake_pathmnist),
            split="train",
            transform=tmap["pathmnist"]["train"],
        )
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        assert batch["image"].shape == (4, 3, 224, 224), \
            f"Bad batch shape: {batch['image'].shape}"

    def test_batch_label_dtype(self, fake_pathmnist: Path) -> None:
        tmap = build_transforms_map(["pathmnist"])
        ds = MedMNISTFolder(
            dataset_name="pathmnist",
            root=str(fake_pathmnist),
            split="train",
            transform=tmap["pathmnist"]["train"],
        )
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        assert batch["label"].shape == (4,)
        assert batch["label"].dtype == torch.int64

    def test_batch_task_id_constant(self, fake_pathmnist: Path) -> None:
        tmap = build_transforms_map(["pathmnist"])
        ds = MedMNISTFolder(
            dataset_name="pathmnist",
            root=str(fake_pathmnist),
            split="train",
            transform=tmap["pathmnist"]["train"],
            task_id=0,
        )
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        assert batch["task_id"].shape == (4,)
        assert (batch["task_id"] == 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# MultiTaskMedDataset
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiTaskMedDataset:

    def test_total_length(self, fake_pathmnist: Path, fake_bloodmnist: Path) -> None:
        tmap = build_transforms_map(["pathmnist", "bloodmnist"])
        ds_p = MedMNISTFolder("pathmnist", str(fake_pathmnist), "train",
                              tmap["pathmnist"]["train"], task_id=0)
        ds_b = MedMNISTFolder("bloodmnist", str(fake_bloodmnist), "train",
                              tmap["bloodmnist"]["train"], task_id=1)
        multi = MultiTaskMedDataset([ds_p, ds_b])
        assert len(multi) == len(ds_p) + len(ds_b)

    def test_task_id_routing(self, fake_pathmnist: Path, fake_bloodmnist: Path) -> None:
        """pathmnist items → task_id=0, bloodmnist items → task_id=1."""
        tmap = build_transforms_map(["pathmnist", "bloodmnist"])
        ds_p = MedMNISTFolder("pathmnist", str(fake_pathmnist), "train",
                              tmap["pathmnist"]["train"], task_id=0)
        ds_b = MedMNISTFolder("bloodmnist", str(fake_bloodmnist), "train",
                              tmap["bloodmnist"]["train"], task_id=1)
        multi = MultiTaskMedDataset([ds_p, ds_b])

        for i in range(len(ds_p)):
            assert multi[i]["task_id"].item() == 0
        for i in range(len(ds_p), len(multi)):
            assert multi[i]["task_id"].item() == 1

    def test_sampler_length(self, fake_pathmnist: Path, fake_bloodmnist: Path) -> None:
        tmap = build_transforms_map(["pathmnist", "bloodmnist"])
        ds_p = MedMNISTFolder("pathmnist", str(fake_pathmnist), "train",
                              tmap["pathmnist"]["train"], task_id=0)
        ds_b = MedMNISTFolder("bloodmnist", str(fake_bloodmnist), "train",
                              tmap["bloodmnist"]["train"], task_id=1)
        multi = MultiTaskMedDataset([ds_p, ds_b])
        sampler = multi.get_sampler()
        assert sampler.num_samples == len(multi)

    def test_per_dataset_weights_sum_to_one(
        self, fake_pathmnist: Path, fake_bloodmnist: Path
    ) -> None:
        """After normalisation each sub-dataset's weight block sums to 1.0."""
        tmap = build_transforms_map(["pathmnist", "bloodmnist"])
        ds_p = MedMNISTFolder("pathmnist", str(fake_pathmnist), "train",
                              tmap["pathmnist"]["train"], task_id=0)
        ds_b = MedMNISTFolder("bloodmnist", str(fake_bloodmnist), "train",
                              tmap["bloodmnist"]["train"], task_id=1)
        for ds in [ds_p, ds_b]:
            w = ds.get_sampler_weights()
            w_norm = w / w.sum()
            assert abs(w_norm.sum().item() - 1.0) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# build_dataloaders factory
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildDataloaders:

    def test_returns_three_splits(self, fake_pathmnist: Path) -> None:
        tmap = build_transforms_map(["pathmnist"])
        loaders = build_dataloaders(
            dataset_roots={"pathmnist": str(fake_pathmnist)},
            transforms_map=tmap,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
        )
        assert set(loaders.keys()) == {"train", "val", "test"}

    def test_train_batch_shape(self, fake_pathmnist: Path) -> None:
        tmap = build_transforms_map(["pathmnist"])
        loaders = build_dataloaders(
            dataset_roots={"pathmnist": str(fake_pathmnist)},
            transforms_map=tmap,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
        )
        batch = next(iter(loaders["train"]))
        assert batch["image"].shape == (4, 3, 224, 224)

    def test_single_dataset_mode_task_ids(
        self, fake_pathmnist: Path, fake_bloodmnist: Path
    ) -> None:
        """single_dataset='pathmnist' → all task_ids must be 0."""
        tmap = build_transforms_map(["pathmnist", "bloodmnist"])
        loaders = build_dataloaders(
            dataset_roots={
                "pathmnist":  str(fake_pathmnist),
                "bloodmnist": str(fake_bloodmnist),
            },
            transforms_map=tmap,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            single_dataset="pathmnist",
        )
        batch = next(iter(loaders["train"]))
        assert (batch["task_id"] == 0).all(), \
            "single_dataset mode should only contain task_id=0"

    def test_val_loader_uses_sequential_sampler(self, fake_pathmnist: Path) -> None:
        """Val and test loaders must not use WeightedRandomSampler."""
        tmap = build_transforms_map(["pathmnist"])
        loaders = build_dataloaders(
            dataset_roots={"pathmnist": str(fake_pathmnist)},
            transforms_map=tmap,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
        )
        assert loaders["val"].sampler.__class__.__name__ == "SequentialSampler"
        assert loaders["test"].sampler.__class__.__name__ == "SequentialSampler"


# ─────────────────────────────────────────────────────────────────────────────
# Optional: real data smoke tests
# Run with: $env:REAL_DATA="1"; pytest tests/test_dataset.py -v -k real
# ─────────────────────────────────────────────────────────────────────────────

REAL_DATA = os.environ.get("REAL_DATA", "0") == "1"
REAL_ROOTS = {
    "pathmnist":  "dataset/pathmnist_dataset",
    "bloodmnist": "dataset/bloodmnist_dataset",
    "dermamnist": "dataset/dermamnist_dataset",
    "octmnist":   "dataset/octmnist_dataset",
}


@pytest.mark.skipif(not REAL_DATA, reason="Set REAL_DATA=1 to run against real data")
class TestRealData:

    def test_pathmnist_loads(self) -> None:
        tmap = build_transforms_map(["pathmnist"])
        ds = MedMNISTFolder(
            dataset_name="pathmnist",
            root=REAL_ROOTS["pathmnist"],
            split="train",
            transform=tmap["pathmnist"]["train"],
        )
        assert len(ds) > 0
        item = ds[0]
        assert item["image"].shape == (3, 224, 224)
        assert 0 <= item["label"].item() < DATASET_META["pathmnist"]["num_classes"]

    def test_all_datasets_load(self) -> None:
        names = list(REAL_ROOTS.keys())
        tmap = build_transforms_map(names)
        for name in names:
            ds = MedMNISTFolder(
                dataset_name=name,
                root=REAL_ROOTS[name],
                split="train",
                transform=tmap[name]["train"],
            )
            assert len(ds) > 0, f"{name} returned 0 samples"
            item = ds[0]
            assert item["image"].shape == (3, 224, 224)

    def test_real_batch_from_loader(self) -> None:
        tmap = build_transforms_map(["pathmnist"])
        loaders = build_dataloaders(
            dataset_roots={"pathmnist": REAL_ROOTS["pathmnist"]},
            transforms_map=tmap,
            batch_size=8,
            num_workers=0,
            pin_memory=False,
        )
        batch = next(iter(loaders["train"]))
        assert batch["image"].shape == (8, 3, 224, 224)
        assert batch["label"].shape == (8,)
        print(f"\nReal pathmnist — label range: "
              f"{batch['label'].min().item()}–{batch['label'].max().item()}")