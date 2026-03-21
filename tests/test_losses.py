"""Tests for loss functions — Kendall multi-task loss.

Fixes vs agent version
----------------------
Constructor  : Our ``KendallMultiTaskLoss`` takes ``task_names: List[str]``
               not ``num_tasks: int``.
forward API  : Signature is ``(task_logits, labels, task_ids)`` — task_names
               are stored on the object at construction, not passed per call.
Parameter    : Our learnable parameter is ``log_sigma``, not ``log_vars``.
Return value : ``forward()`` returns ``(total_loss, task_losses_dict)`` —
               tests must unpack the tuple.
"""

from __future__ import annotations

import pytest
import torch

from medical_mamba.training.losses import KendallMultiTaskLoss


class TestKendallMultiTaskLoss:

    @pytest.fixture
    def loss_fn(self) -> KendallMultiTaskLoss:
        """Two-task loss with no label smoothing."""
        return KendallMultiTaskLoss(
            task_names=["task_a", "task_b"],
            label_smoothing=0.0,
        )

    # ------------------------------------------------------------------ #
    #  Return shape                                                        #
    # ------------------------------------------------------------------ #

    def test_output_is_scalar(self, loss_fn: KendallMultiTaskLoss) -> None:
        """Total loss must be a 0-dim scalar tensor."""
        task_logits = {
            "task_a": torch.randn(4, 5),
            "task_b": torch.randn(6, 3),
        }
        labels  = torch.cat([torch.randint(0, 5, (4,)), torch.randint(0, 3, (6,))])
        task_ids = torch.cat([torch.zeros(4), torch.ones(6)]).long()

        loss, task_losses = loss_fn(task_logits, labels, task_ids)

        assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"

    def test_returns_per_task_losses(self, loss_fn: KendallMultiTaskLoss) -> None:
        """forward() must return a dict of per-task CE losses."""
        task_logits = {
            "task_a": torch.randn(4, 5),
            "task_b": torch.randn(4, 3),
        }
        labels   = torch.cat([torch.randint(0, 5, (4,)), torch.randint(0, 3, (4,))])
        task_ids = torch.cat([torch.zeros(4), torch.ones(4)]).long()

        _, task_losses = loss_fn(task_logits, labels, task_ids)

        assert "task_a" in task_losses
        assert "task_b" in task_losses
        assert isinstance(task_losses["task_a"], float)
        assert isinstance(task_losses["task_b"], float)

    # ------------------------------------------------------------------ #
    #  Gradient flow                                                       #
    # ------------------------------------------------------------------ #

    def test_gradient_flows_to_log_sigma(self, loss_fn: KendallMultiTaskLoss) -> None:
        """Gradients must reach the learnable log_sigma parameter."""
        task_logits = {
            "task_a": torch.randn(4, 5),
            "task_b": torch.randn(4, 3),
        }
        labels   = torch.cat([torch.randint(0, 5, (4,)), torch.randint(0, 3, (4,))])
        task_ids = torch.cat([torch.zeros(4), torch.ones(4)]).long()

        loss, _ = loss_fn(task_logits, labels, task_ids)
        loss.backward()

        assert loss_fn.log_sigma.grad is not None, \
            "No gradient reached log_sigma"
        assert loss_fn.log_sigma.grad.abs().sum() > 0, \
            "log_sigma gradient is all zeros"

    def test_gradient_flows_to_logits(self, loss_fn: KendallMultiTaskLoss) -> None:
        """Gradients must also reach the input logits."""
        logits_a = torch.randn(4, 5, requires_grad=True)
        logits_b = torch.randn(4, 3, requires_grad=True)
        task_logits = {"task_a": logits_a, "task_b": logits_b}

        labels   = torch.cat([torch.randint(0, 5, (4,)), torch.randint(0, 3, (4,))])
        task_ids = torch.cat([torch.zeros(4), torch.ones(4)]).long()

        loss, _ = loss_fn(task_logits, labels, task_ids)
        loss.backward()

        assert logits_a.grad is not None, "No gradient reached task_a logits"
        assert logits_b.grad is not None, "No gradient reached task_b logits"

    # ------------------------------------------------------------------ #
    #  Numerical properties                                                #
    # ------------------------------------------------------------------ #

    def test_loss_is_positive(self, loss_fn: KendallMultiTaskLoss) -> None:
        """With log_sigma=0 (σ=1), loss = CE + 0 which must be positive."""
        task_logits = {"task_a": torch.randn(8, 5)}
        labels   = torch.randint(0, 5, (8,))
        task_ids = torch.zeros(8, dtype=torch.long)

        loss, _ = loss_fn(task_logits, labels, task_ids)
        assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"

    def test_loss_is_finite(self, loss_fn: KendallMultiTaskLoss) -> None:
        """Loss must never be NaN or Inf on normal inputs."""
        task_logits = {
            "task_a": torch.randn(4, 5),
            "task_b": torch.randn(4, 3),
        }
        labels   = torch.cat([torch.randint(0, 5, (4,)), torch.randint(0, 3, (4,))])
        task_ids = torch.cat([torch.zeros(4), torch.ones(4)]).long()

        loss, _ = loss_fn(task_logits, labels, task_ids)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_absent_task_ignored(self, loss_fn: KendallMultiTaskLoss) -> None:
        """Tasks absent from task_logits dict must not contribute to loss
        and must not cause a KeyError."""
        # Only task_a is present — task_b has no samples in this batch
        task_logits = {"task_a": torch.randn(8, 5)}
        labels   = torch.randint(0, 5, (8,))
        task_ids = torch.zeros(8, dtype=torch.long)

        loss, task_losses = loss_fn(task_logits, labels, task_ids)

        assert torch.isfinite(loss)
        assert "task_a" in task_losses
        assert "task_b" not in task_losses   # absent task must not appear

    def test_single_task_mode(self) -> None:
        """Single-task KendallMultiTaskLoss must degenerate to plain CE."""
        loss_fn = KendallMultiTaskLoss(
            task_names=["pathmnist"],
            label_smoothing=0.0,
        )
        logits   = torch.randn(8, 9)
        labels   = torch.randint(0, 9, (8,))
        task_ids = torch.zeros(8, dtype=torch.long)

        loss, task_losses = loss_fn({"pathmnist": logits}, labels, task_ids)

        assert loss.ndim == 0
        assert "pathmnist" in task_losses
        assert torch.isfinite(loss)

    def test_sigma_values_helper(self, loss_fn: KendallMultiTaskLoss) -> None:
        """sigma_values() must return positive σ for each task."""
        sigmas = loss_fn.sigma_values()
        assert set(sigmas.keys()) == {"task_a", "task_b"}
        for name, sigma in sigmas.items():
            assert sigma > 0, f"σ for {name} is not positive: {sigma}"

    def test_kendall_weighting_math(self) -> None:
        """Verify the Kendall formula: exp(-2·log_σ)·L + log_σ.

        With log_σ = 0 (σ=1): weight = exp(0) = 1, regulariser = 0.
        With log_σ = 1 (σ=e): weight = exp(-2) ≈ 0.135, regulariser = 1.
        """
        loss_fn = KendallMultiTaskLoss(task_names=["task_a"], label_smoothing=0.0)

        # Force log_sigma to a known value
        with torch.no_grad():
            loss_fn.log_sigma.fill_(0.0)

        logits   = torch.randn(4, 5)
        labels   = torch.randint(0, 5, (4,))
        task_ids = torch.zeros(4, dtype=torch.long)

        loss, task_losses = loss_fn({"task_a": logits}, labels, task_ids)

        import torch.nn as nn
        ce = nn.CrossEntropyLoss()(logits, labels)
        # With log_σ=0: total = exp(0)*CE + 0 = 1*CE = CE
        assert abs(loss.item() - ce.item()) < 1e-5, \
            f"Expected loss≈CE={ce.item():.4f}, got {loss.item():.4f}"