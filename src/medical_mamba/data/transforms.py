"""Modality-aware transform pipelines for MedMNIST+ datasets.

Design decisions
----------------
API
    ``build_transforms_map()`` is the single public entry point.
    ``get_train_transforms`` / ``get_val_transforms`` are also exposed
    for direct use in notebooks and backward compatibility.

Grayscale -> RGB
    MedMNISTFolder.__getitem__ calls ``Image.open(...).convert("RGB")``
    on every image before passing it to the transform.  This means the
    transform always receives a 3-channel PIL image regardless of source
    modality (OCTMNIST is L-mode on disk but RGB by the time it reaches
    ToTensor).  Therefore NO ``repeat(3, 1, 1)`` is needed here — adding
    it would triple the channels a second time producing [9, 224, 224].

    The ``in_channels`` field in DATASET_META records the source channel
    count for reference only.  The pipeline output is always 3-channel.

ColorJitter
    Per-dataset configs. OCTMNIST gets None (jitter on greyscale content,
    even as RGB, is meaningless and wastes augmentation budget).

RandomAffine
    Mild translation + scale helps generalise on small datasets like
    DermaMNIST (7k training samples).

Resize strategy
    Val/test: Resize(256) then CenterCrop(224) — standard ImageNet
    practice, avoids border distortion from direct resize.
    Train: RandomResizedCrop already handles spatial variation.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import torchvision.transforms as T

from medical_mamba.data.constants import DATASET_META


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset colour jitter configs
# None = disabled (OCTMNIST: jitter on greyscale-as-RGB is meaningless)
# ─────────────────────────────────────────────────────────────────────────────

_JITTER_CFG: Dict[str, Dict[str, float]] = {
    "pathmnist":  {"brightness": 0.3, "contrast": 0.3, "saturation": 0.2, "hue": 0.05},
    "dermamnist": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.1, "hue": 0.03},
    "bloodmnist": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.15, "hue": 0.04},
    "octmnist":   None,
}


def get_train_transforms(
    dataset_name: str,
    img_size: int = 224,
) -> T.Compose:
    """Training augmentation pipeline for ``dataset_name``.

    Pipeline order:
        RandomResizedCrop  -> spatial variation
        RandomHorizontalFlip / RandomVerticalFlip
        RandomRotation(15)
        RandomAffine       -> mild translation + scale
        ColorJitter        -> colour variation (RGB datasets only)
        ToTensor           -> [0, 1] float32, shape (3, H, W)
        Normalize          -> per-dataset channel statistics

    Note: no grayscale->RGB repeat needed — PIL .convert("RGB") in
    MedMNISTFolder.__getitem__ guarantees 3-channel input to this pipeline.

    Parameters
    ----------
    dataset_name : str
        Key in DATASET_META.
    img_size : int
        Output spatial resolution (default 224).

    Returns
    -------
    torchvision.transforms.Compose
    """
    meta       = DATASET_META[dataset_name]
    jitter_cfg = _JITTER_CFG.get(dataset_name)

    aug: list = [
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    ]

    if jitter_cfg is not None:
        aug.append(T.RandomApply([T.ColorJitter(**jitter_cfg)], p=0.5))

    aug += [
        T.ToTensor(),
        T.Normalize(mean=meta["mean"], std=meta["std"]),
    ]

    return T.Compose(aug)


def get_val_transforms(
    dataset_name: str,
    img_size: int = 224,
) -> T.Compose:
    """Deterministic val/test pipeline — no augmentation.

    Pipeline: Resize(256) -> CenterCrop(img_size) -> ToTensor -> Normalize

    Parameters
    ----------
    dataset_name : str
        Key in DATASET_META.
    img_size : int
        Output spatial resolution (default 224).

    Returns
    -------
    torchvision.transforms.Compose
    """
    meta = DATASET_META[dataset_name]
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=meta["mean"], std=meta["std"]),
    ])


def build_transforms_map(
    dataset_names: List[str],
    img_size: int = 224,
) -> Dict[str, Dict[str, Callable]]:
    """Build train / val / test transforms for each dataset.

    Parameters
    ----------
    dataset_names : List[str]
        e.g. ["pathmnist", "octmnist"]
    img_size : int
        Target spatial resolution (default 224).

    Returns
    -------
    Dict[str, Dict[str, Callable]]
        {dataset_name: {"train": transform, "val": transform, "test": transform}}

    Example
    -------
        tmap = build_transforms_map(["pathmnist", "octmnist"])
        ds = MedMNISTFolder(
            dataset_name="pathmnist",
            root="dataset/pathmnist_dataset",
            split="train",
            transform=tmap["pathmnist"]["train"],
        )
    """
    result: Dict[str, Dict[str, Callable]] = {}
    for name in dataset_names:
        if name not in DATASET_META:
            raise KeyError(
                f"Unknown dataset '{name}'. "
                f"Valid options: {list(DATASET_META.keys())}"
            )
        val_t = get_val_transforms(name, img_size)
        result[name] = {
            "train": get_train_transforms(name, img_size),
            "val":   val_t,
            "test":  val_t,
        }
    return result