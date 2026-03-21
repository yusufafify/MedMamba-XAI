"""Checkpoint save/load helpers.

Provides a uniform interface for persisting and restoring model weights,
optimizer state, epoch, and validation metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: str | Path,
) -> Path:
    """Save a training checkpoint.

    Parameters
    ----------
    model : nn.Module
        Model to save.
    optimizer : Optimizer
        Optimizer whose state dict is persisted.
    epoch : int
        Current epoch number.
    metrics : Dict[str, float]
        Validation metrics at the time of saving.
    filepath : str | Path
        Destination file path.

    Returns
    -------
    Path
        Absolute path to the saved checkpoint.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    torch.save(checkpoint, str(filepath))
    return filepath.resolve()


def load_checkpoint(
    filepath: str | Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    device: str | torch.device = "cpu",
) -> Tuple[nn.Module, Optional[Optimizer], int, Dict[str, Any]]:
    """Load a training checkpoint.

    Parameters
    ----------
    filepath : str | Path
        Path to the ``.pt`` checkpoint file.
    model : nn.Module
        Model to load weights into.
    optimizer : Optimizer | None
        Optimizer to restore state into.  ``None`` to skip.
    device : str | torch.device
        Device to map tensors to.

    Returns
    -------
    Tuple
        ``(model, optimizer, epoch, metrics)``
    """
    checkpoint = torch.load(str(filepath), map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})

    return model, optimizer, epoch, metrics
