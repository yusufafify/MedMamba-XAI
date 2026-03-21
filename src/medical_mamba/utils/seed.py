"""Reproducibility seed helper.

Sets seeds for ``random``, ``numpy``, ``torch`` (CPU + CUDA), and
configures cuDNN for deterministic behaviour.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to use everywhere.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN determinism (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
