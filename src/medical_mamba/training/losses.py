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


# ─────────────────────────────────────────────────────────────────────────────
# Supervised Contrastive Domain Loss
# ─────────────────────────────────────────────────────────────────────────────

class ContrastiveDomainLoss(nn.Module):
    """Supervised contrastive loss for domain (dataset) separation.

    Uses ``task_id`` as the positive/negative label — all images from the
    same dataset are positives to each other, all others are negatives.
    Combined with Kendall multi-task CE, this yields a backbone that is
    simultaneously class-discriminative (within each dataset) and
    domain-discriminative (across the 4 medical imaging modalities),
    enabling nearest-prototype autonomous inference.

    Reference
    ---------
    Khosla et al. (2020). *Supervised Contrastive Learning.* arXiv:2004.11362.

    Math
    ----
    For each anchor ``i``::

        SupCon(i) = -1/|P(i)| · Σ_{p∈P(i)} log [
            exp(z_i · z_p / τ) / Σ_{a∈A(i)} exp(z_i · z_a / τ)
        ]

    where ``z_i`` is the L2-normalised projection, ``P(i)`` is the set of
    in-batch positives (same task_id, excluding ``i``), and ``A(i)`` is
    all other samples in the batch (excluding ``i``).

    Parameters
    ----------
    temperature : float
        Logit scaling temperature. Lower → sharper separation.
        Recommended range 0.05–0.2. Default 0.07 (matches the paper).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        task_ids:   torch.Tensor,
    ) -> torch.Tensor:
        """Compute SupCon loss with log-sum-exp numerical stability.

        Parameters
        ----------
        embeddings : torch.Tensor
            L2-normalised projections ``(B, D)``. The caller is responsible
            for the L2 normalisation — this loss does NOT re-normalise.
        task_ids : torch.Tensor
            Integer dataset index per sample ``(B,)``.

        Returns
        -------
        torch.Tensor
            Scalar loss. Returns ``0.0`` (no gradient) when the batch
            contains only one unique task_id, or when no anchor in the
            batch has at least one positive.
        """
        # Force float32 — AMP float16 causes NaN in the exp step below
        z = embeddings.float()
        device = z.device
        B = z.size(0)

        # Single-task batch → no negatives at all → loss is undefined.
        # Return a zero scalar that's connected to nothing (no grad).
        if task_ids.unique().numel() < 2:
            return torch.zeros((), device=device, dtype=torch.float32)

        # Similarity matrix with log-sum-exp trick: subtract per-row max
        # before exp to avoid overflow with large logits.
        logits = (z @ z.T) / self.temperature
        logits_max = logits.detach().max(dim=1, keepdim=True).values
        logits = logits - logits_max

        # Masks
        eye = torch.eye(B, device=device, dtype=torch.bool)
        same_task  = task_ids.unsqueeze(0) == task_ids.unsqueeze(1)  # (B,B)
        pos_mask   = same_task & ~eye                                # positives (exclude self)
        valid_mask = ~eye                                            # denom: everything except self

        exp_logits = torch.exp(logits) * valid_mask
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
        )

        # Per-anchor mean log-prob over its positives.
        # clamp_min(1) prevents /0 for anchors with no positives — those
        # rows are masked out below anyway.
        pos_counts = pos_mask.sum(dim=1).clamp_min(1)
        mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1) / pos_counts

        # Anchors with no positives (single-sample-from-a-task) are skipped.
        has_pos = pos_mask.any(dim=1)
        if not has_pos.any():
            return torch.zeros((), device=device, dtype=torch.float32)

        loss = -mean_log_prob_pos[has_pos].mean()
        return loss
