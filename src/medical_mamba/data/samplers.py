"""Weighted random sampler factory — standalone convenience wrapper.

The primary sampler API lives on the dataset class itself:
    ``ds.get_sampler()``          single-dataset
    ``multi_ds.get_sampler()``    multi-task dual-level

This module provides ``build_weighted_sampler()`` as a standalone function
for backward compatibility with agent-generated scripts that import it
directly, and for use in notebooks.
"""

from __future__ import annotations

from torch.utils.data import WeightedRandomSampler

from medical_mamba.data.dataset import MedMNISTFolder


def build_weighted_sampler(
    dataset: MedMNISTFolder,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """Create a ``WeightedRandomSampler`` from dataset class frequencies.

    Delegates to ``dataset.get_sampler()`` which uses inverse-frequency
    weights with an epsilon guard against zero-count classes.

    Parameters
    ----------
    dataset : MedMNISTFolder
    replacement : bool
        Sample with replacement (default ``True``).

    Returns
    -------
    WeightedRandomSampler
    """
    return dataset.get_sampler(replacement=replacement)
