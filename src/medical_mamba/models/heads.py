"""Classification heads for single-task and multi-task setups.

Both source versions were essentially identical here.  The only meaningful
differences:

- model.py used ``dropout=0.3`` as default; agent used ``0.1``.
  Merged default is ``0.1`` (less aggressive, easier to tune up).
- model.py omitted the leading ``LayerNorm``; agent included it.
  The ``LayerNorm`` is kept — it stabilises training when the backbone
  output distribution shifts during the early warmup phase.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Lightweight MLP head: LayerNorm → Dropout → Linear.

    Parameters
    ----------
    in_features : int
        Backbone output dimension (``VMambaBackbone.out_dim``).
    num_classes : int
        Number of target classes for this task.
    dropout : float
        Dropout probability applied before the linear projection.
        Default ``0.1`` — increase to ``0.3`` if overfitting.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map pooled features to class logits.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, in_features)``

        Returns
        -------
        torch.Tensor
            ``(B, num_classes)`` — raw logits (no softmax).
        """
        return self.head(x)

    def __repr__(self) -> str:
        return (
            f"ClassificationHead("
            f"in={self.in_features}, "
            f"out={self.num_classes}, "
            f"dropout={self.head[1].p:.2f})"
        )
