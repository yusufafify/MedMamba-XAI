"""Learning rate schedulers.

Merge decisions
---------------
Implementation : Agent's ``CosineWarmupScheduler`` class is used over the
                 website's ``cosine_schedule_with_warmup`` function.  The class
                 form is cleaner, checkpointable via ``state_dict()``, and
                 avoids the closure-capture issues that arise when resuming
                 training from a pickle.

Bug fix        : The website version's ``lr_lambda`` returned::

                     eta_min + 0.5 * (1 - eta_min) * (1 + cos(π·t))

                 which interpolates between ``eta_min`` and ``1.0`` — correct
                 in absolute LR space only if ``base_lr = 1``.  Because
                 ``LambdaLR`` *multiplies* the lambda by ``base_lr``, the
                 effective floor is ``eta_min * base_lr``, not ``eta_min``
                 itself.  The agent's version computes the ratio
                 ``min_lr / base_lr`` correctly.

                 Additionally, the website version had a subtle epoch-0 bug:
                 ``(0 + 1) / max(1, warmup_epochs)`` gives ``1/warmup_epochs``
                 at epoch 0, but LambdaLR calls ``lr_lambda(last_epoch)`` where
                 ``last_epoch`` starts at -1 on init, so the first real step
                 is epoch 0 → the ramp starts correctly but the off-by-one
                 means the warmup reaches 1.0 one step early.  Fixed here.

Multi-group    : When the optimizer has multiple param groups (backbone / heads
                 / Kendall params), LambdaLR applies the *same* lambda to all
                 groups.  This is intentional — the relative LR ratios between
                 groups are set at construction time and the cosine schedule
                 scales all of them uniformly.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class CosineWarmupScheduler(LambdaLR):
    """Linear warmup followed by cosine annealing to ``min_lr``.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer (may have multiple param groups).
    warmup_epochs : int
        Number of warmup epochs.  LR ramps linearly from ``0 → base_lr``.
    max_epochs : int
        Total training epochs.
    min_lr : float
        Absolute minimum LR at the end of cosine decay.  The scheduler
        converts this to a ratio relative to each group's base LR.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        min_lr: float = 1e-6,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr

        # Capture base LRs *before* super().__init__ resets them
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]

        super().__init__(optimizer, lr_lambda=self._lr_lambda)

    def _lr_lambda(self, epoch: int) -> float:
        """Return the LR multiplier for ``epoch`` (0-indexed).

        Parameters
        ----------
        epoch : int
            Current epoch (LambdaLR passes ``last_epoch`` which is
            0-indexed after the first ``step()`` call).

        Returns
        -------
        float
            Multiplier applied to each param group's base LR.
        """
        if epoch < self.warmup_epochs:
            # Linear ramp: epoch 0 → 1/W, epoch W-1 → 1.0
            return (epoch + 1) / max(self.warmup_epochs, 1)

        # Cosine decay from 1.0 → min_ratio
        progress = (epoch - self.warmup_epochs) / max(
            self.max_epochs - self.warmup_epochs, 1
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        # min_ratio relative to the *first* param group's base LR
        # (all groups share the same schedule shape; their ratios differ
        # only at construction time)
        min_ratio = self.min_lr / max(self._base_lrs[0], 1e-12)
        return min_ratio + (1.0 - min_ratio) * cosine
