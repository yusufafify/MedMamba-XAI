"""Explode MedMNIST .npz archives into a folder of individual JPEG files.

MedMNIST datasets are distributed as compressed NumPy archives (``.npz``).
This script extracts every image into a directory of individual JPEG files,
which can be more convenient for ad-hoc inspection and some data loaders.

Usage::

    python scripts/explode_npz.py \\
        --npz data/pathmnist.npz \\
        --output data/pathmnist_images/ \\
        --split train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def explode(
    npz_path: str | Path,
    output_dir: str | Path,
    split: str = "train",
) -> int:
    """Extract images from an ``.npz`` archive into individual JPEG files.

    Parameters
    ----------
    npz_path : str | Path
        Path to the ``.npz`` file.
    output_dir : str | Path
        Destination directory.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.

    Returns
    -------
    int
        Number of images extracted.
    """
    npz_path = Path(npz_path)
    output_dir = Path(output_dir) / split
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(str(npz_path))
    images = data[f"{split}_images"]
    labels = data[f"{split}_labels"].squeeze()

    for idx in tqdm(range(len(images)), desc=f"Extracting {split}"):
        img = images[idx]
        label = int(labels[idx])

        # Create class sub-directory
        class_dir = output_dir / str(label)
        class_dir.mkdir(exist_ok=True)

        if img.ndim == 2:
            pil_img = Image.fromarray(img, mode="L")
        else:
            pil_img = Image.fromarray(img, mode="RGB")

        pil_img.save(str(class_dir / f"{idx:06d}.jpg"), quality=95)

    return len(images)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Explode MedMNIST .npz to JPEG folder")
    parser.add_argument("--npz", type=str, required=True, help="Path to .npz file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Which split(s) to extract",
    )
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    total = 0
    for split in splits:
        n = explode(args.npz, args.output, split)
        total += n
        print(f"  {split}: {n} images")

    print(f"\n✓ Extracted {total} total images to {args.output}")


if __name__ == "__main__":
    main()
