"""Loss functions for single-task and multi-task training.

Kendall multi-task loss
-----------------------
Implements the *homoscedastic uncertainty* weighting from Kendall et al.
(2018).  Each task's cross-entropy loss is weighted by a learnable
log-standard-deviation parameter ``log_σᵢ``::

    L_total = Σᵢ  exp(-2·log_σᵢ) · L_i  +  log_σᵢ

This is equivalent to::

    L_total = Σᵢ  (1 / 2σᵢ²) · L_i  +  log(σᵢ)

because ``exp(-2·log_σ) = 1/σ²`` and the ``1/2`` is absorbed into the
learned parameter at convergence.

Merge decisions
---------------
API signature  : Agent version (``task_logits`` dict + ``labels`` + ``task_ids``
                 + ``task_names``).  Website version took parallel lists which
                 required the caller to maintain list order — error-prone.
Math           : Website version.  Agent used ``exp(-log_vars)`` which is
                 ``1/σ`` not ``1/σ²`` — missing the factor of 2 in the exponent.
                 Concrete impact: the learned weighting is off by a square root,
                 causing the σ values to converge at wrong magnitudes.
Parameter name : ``log_sigma`` (website) over ``log_vars`` (agent) — more
                 explicit about what the parameter represents.
task_losses    : Returned as a ``Dict[str, float]`` (merged from both) so the
                 trainer can log per-task losses without extra bookkeeping.

References
----------
Kendall, A., Gal, Y., & Cipolla, R. (2018). *Multi-Task Learning Using
Uncertainty to Weigh Losses for Scene Geometry and Semantics.* CVPR.
https://arxiv.org/abs/1705.07115
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class KendallMultiTaskLoss(nn.Module):
    """Multi-task loss with learned homoscedastic uncertainty weights.

    Parameters
    ----------
    task_names : List[str]
        Ordered task names.  Position = task_id from the DataLoader.
    label_smoothing : float
        Applied to every per-task cross-entropy.
    """

    def __init__(
        self,
        task_names: List[str],
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.task_names = task_names
        # log(σ) = 0  →  σ = 1  →  equal initial weighting across tasks
        self.log_sigma = nn.Parameter(torch.zeros(len(task_names)))
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        task_logits: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the weighted multi-task loss.

        Parameters
        ----------
        task_logits : Dict[str, torch.Tensor]
            ``{task_name: (N_task, C_task)}`` — only tasks present in the
            current batch need to be included.
        labels : torch.Tensor
            Full batch labels ``(B,)``.
        task_ids : torch.Tensor
            Integer task index per sample ``(B,)``.

        Returns
        -------
        total_loss : torch.Tensor
            Scalar weighted loss ready for ``.backward()``.
        task_losses : Dict[str, float]
            Detached per-task CE losses for logging (tasks absent from the
            batch are omitted).
        """
        total_loss = torch.zeros(1, device=labels.device).squeeze()
        task_losses: Dict[str, float] = {}

        for tid, name in enumerate(self.task_names):
            if name not in task_logits:
                continue

            mask = task_ids == tid
            if not mask.any():
                continue

            ce = self.ce(task_logits[name], labels[mask])
            task_losses[name] = ce.item()

            # Correct Kendall formula: exp(-2·log_σ) · L + log_σ
            # exp(-2·log_σ) = 1/σ² — the precision term
            precision = torch.exp(-2.0 * self.log_sigma[tid])
            total_loss = total_loss + precision * ce + self.log_sigma[tid]

        return total_loss, task_losses

    def sigma_values(self) -> Dict[str, float]:
        """Return current σ values per task (diagnostic / TensorBoard logging)."""
        return {
            name: torch.exp(self.log_sigma[i]).item()
            for i, name in enumerate(self.task_names)
        }
