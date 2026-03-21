"""Metric tracking for training and evaluation.

Merge decisions
---------------
Loss tracking  : Website version tracked per-task losses inside the tracker.
                 Agent did not.  Kept — essential for diagnosing which task
                 is struggling without adding extra state in the trainer.
avg_f1_macro   : Agent version computed an average F1 across tasks in
                 ``compute()``.  Kept — this is the primary scalar to watch
                 in multi-task TensorBoard curves.
update API     : Agent's ``update`` / ``update_multitask`` split is cleaner
                 than the website's single ``update(task_name, ...)`` that
                 required the caller to loop.  Kept.
Return shape   : Website returned nested ``{task: {acc, f1, loss}}``.  Agent
                 returned flat ``{task_acc, task_f1}``.  Merged to a flat dict
                 (agent style) — easier to pass directly to TensorBoard and CSV
                 writers — but with the loss keys added from the website version.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


class TaskMetricTracker:
    """Accumulate predictions / labels / losses and compute epoch metrics.

    Parameters
    ----------
    task_names : List[str], optional
        Task names for multi-task mode.  ``None`` for single-task (stored
        under the ``"default"`` key internally).
    """

    def __init__(self, task_names: Optional[List[str]] = None) -> None:
        self.task_names = task_names
        self._keys = task_names if task_names else ["default"]
        self.reset()

    def reset(self) -> None:
        """Clear accumulated state for a new epoch."""
        self._preds:  Dict[str, List[int]]   = {k: [] for k in self._keys}
        self._labels: Dict[str, List[int]]   = {k: [] for k in self._keys}
        self._losses: Dict[str, List[float]] = {k: [] for k in self._keys}

    # ------------------------------------------------------------------ #
    #  Update methods                                                      #
    # ------------------------------------------------------------------ #

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss: Optional[float] = None,
    ) -> None:
        """Accumulate one single-task batch.

        Parameters
        ----------
        logits : torch.Tensor
            ``(B, C)`` raw logits.
        labels : torch.Tensor
            ``(B,)`` ground-truth class indices.
        loss : float, optional
            Scalar loss value for this batch.
        """
        preds = logits.argmax(dim=1).cpu().tolist()
        self._preds["default"].extend(preds)
        self._labels["default"].extend(labels.cpu().tolist())
        if loss is not None:
            self._losses["default"].append(loss)

    def update_multitask(
        self,
        task_logits: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        task_ids: torch.Tensor,
        task_losses: Optional[Dict[str, float]] = None,
    ) -> None:
        """Accumulate one multi-task batch.

        Parameters
        ----------
        task_logits : Dict[str, torch.Tensor]
            Per-task logits from ``model.forward_multi()``.
        labels : torch.Tensor
            Full batch labels ``(B,)``.
        task_ids : torch.Tensor
            Per-sample task index ``(B,)``.
        task_losses : Dict[str, float], optional
            Per-task CE losses from ``KendallMultiTaskLoss.forward()``.
        """
        task_names = self.task_names or []
        for tid, name in enumerate(task_names):
            if name not in task_logits:
                continue
            mask = task_ids == tid
            if not mask.any():
                continue

            preds = task_logits[name].argmax(dim=1).cpu().tolist()
            self._preds[name].extend(preds)
            self._labels[name].extend(labels[mask].cpu().tolist())

            if task_losses and name in task_losses:
                self._losses[name].append(task_losses[name])

    # ------------------------------------------------------------------ #
    #  Compute                                                             #
    # ------------------------------------------------------------------ #

    def compute(self) -> Dict[str, float]:
        """Compute accuracy, F1-macro, and mean loss for each task.

        Returns
        -------
        Dict[str, float]
            Flat metric dict.  Keys:

            - Single-task: ``accuracy``, ``f1_macro``, ``loss``
            - Multi-task:  ``{task}_accuracy``, ``{task}_f1_macro``,
              ``{task}_loss``, ``avg_f1_macro``
        """
        metrics: Dict[str, float] = {}

        for name in self._keys:
            preds  = np.array(self._preds[name])
            labels = np.array(self._labels[name])
            if len(preds) == 0:
                continue

            acc = float(accuracy_score(labels, preds))
            f1  = float(f1_score(labels, preds, average="macro", zero_division=0))
            avg_loss = float(np.mean(self._losses[name])) if self._losses[name] else 0.0

            prefix = "" if name == "default" else f"{name}_"
            metrics[f"{prefix}accuracy"] = acc
            metrics[f"{prefix}f1_macro"] = f1
            metrics[f"{prefix}loss"]     = avg_loss

        # Average F1 across all tasks (primary multi-task scalar for TB)
        if self.task_names:
            f1_vals = [
                metrics[f"{n}_f1_macro"]
                for n in self.task_names
                if f"{n}_f1_macro" in metrics
            ]
            if f1_vals:
                metrics["avg_f1_macro"] = float(np.mean(f1_vals))

        return metrics
