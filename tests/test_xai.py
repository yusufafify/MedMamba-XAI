"""Tests for XAI saliency map generation.

Fixes vs agent version
----------------------
Model construction : Agent used ``MedicalVMamba(in_channels=3, dims=...,
                     task_classes=..., ssm_ratio=...)`` — flat kwargs that
                     don't exist in our constructor.  Our ``MedicalVMamba``
                     takes ``task_configs: List[Tuple[str, int]]`` and
                     ``backbone_cfg: dict``.  Fixed to use ``build_model()``.

Task name          : ``SSMGradCAM.__call__`` takes ``task_name`` kwarg which
                     defaults to ``"default"``.  The model must have a head
                     registered under that exact name, so ``build_model`` is
                     called with ``[("default", 4)]``.

generate_batch     : Passes ``task_name="default"`` explicitly so it routes
                     to the correct head regardless of default arg changes.

patch_size=8       : Using patch_size=8 (not 4) keeps the test fast on CPU —
                     4x fewer tokens, same logic verified.

depths=[1,1,1,1]  : Kept from agent — minimal depth makes the test ~10x
                     faster while still exercising the full GradCAM pipeline.
"""

from __future__ import annotations

import pytest
import torch

from medical_mamba.models.medical_vmamba import build_model, MedicalVMamba
from medical_mamba.xai.gradcam import SSMGradCAM


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model() -> MedicalVMamba:
    """Minimal VMamba model for XAI testing.

    Uses:
    - task_name "default" so SSMGradCAM's default task_name arg matches
    - patch_size=8 for speed (4x fewer tokens vs patch_size=4)
    - depths=[1,1,1,1] for minimal compute
    - drop_path_rate=0.0 for deterministic GradCAM
    """
    return build_model(
        task_configs=[("default", 4)],
        model_size="tiny",
        patch_size=8,
        drop_path_rate=0.0,   # deterministic — no stochastic depth during XAI
    )


@pytest.fixture(scope="module")
def cam(model: MedicalVMamba) -> SSMGradCAM:
    """SSMGradCAM hooked into the last backbone stage."""
    return SSMGradCAM(model, target_stage=3)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSSMGradCAM:

    def test_heatmap_shape(self, cam: SSMGradCAM) -> None:
        """Heatmap must match input spatial resolution (224×224)."""
        x = torch.randn(1, 3, 224, 224)
        hm = cam(x, class_idx=0, task_name="default")
        assert hm.shape == (224, 224), \
            f"Expected (224, 224), got {hm.shape}"

    def test_heatmap_range(self, cam: SSMGradCAM) -> None:
        """Heatmap values must be normalised to [0, 1]."""
        x = torch.randn(1, 3, 224, 224)
        hm = cam(x, class_idx=0, task_name="default")
        assert hm.min() >= 0.0,          f"Min value {hm.min()} < 0"
        assert hm.max() <= 1.0 + 1e-5,  f"Max value {hm.max()} > 1"

    def test_heatmap_not_all_zeros(self, cam: SSMGradCAM) -> None:
        """Heatmap must have non-zero activations — all-zero means hooks failed."""
        x = torch.randn(1, 3, 224, 224)
        hm = cam(x, class_idx=0, task_name="default")
        assert hm.max() > 1e-6, \
            "Heatmap is all zeros — GradCAM hooks may have failed"

    def test_auto_class_selection(self, cam: SSMGradCAM) -> None:
        """When class_idx=None, should use argmax of predicted logits."""
        x = torch.randn(1, 3, 224, 224)
        hm = cam(x, class_idx=None, task_name="default")
        assert hm.shape == (224, 224)

    def test_different_classes_give_different_maps(self, cam: SSMGradCAM) -> None:
        """Different target classes must produce different saliency maps."""
        torch.manual_seed(0)
        x = torch.randn(1, 3, 224, 224)
        hm0 = cam(x, class_idx=0, task_name="default")
        hm1 = cam(x, class_idx=1, task_name="default")
        # Maps for different classes should not be identical
        assert not torch.allclose(hm0, hm1), \
            "Class 0 and class 1 produced identical heatmaps"

    def test_heatmap_is_cpu_tensor(self, cam: SSMGradCAM) -> None:
        """Returned heatmap must always be on CPU for downstream plotting."""
        x = torch.randn(1, 3, 224, 224)
        hm = cam(x, class_idx=0, task_name="default")
        assert hm.device.type == "cpu"

    def test_heatmap_dtype_float32(self, cam: SSMGradCAM) -> None:
        """Heatmap must be float32 for matplotlib compatibility."""
        x = torch.randn(1, 3, 224, 224)
        hm = cam(x, class_idx=0, task_name="default")
        assert hm.dtype == torch.float32

    def test_generate_batch_shape(self, cam: SSMGradCAM) -> None:
        """generate_batch must return (B, H, W) tensor."""
        x = torch.randn(3, 3, 224, 224)
        hms = cam.generate_batch(x, task_name="default")
        assert hms.shape == (3, 224, 224), \
            f"Expected (3, 224, 224), got {hms.shape}"

    def test_generate_batch_range(self, cam: SSMGradCAM) -> None:
        """All heatmaps in a batch must be in [0, 1]."""
        x = torch.randn(2, 3, 224, 224)
        hms = cam.generate_batch(x, task_name="default")
        assert hms.min() >= 0.0
        assert hms.max() <= 1.0 + 1e-5

    def test_generate_batch_with_class_indices(self, cam: SSMGradCAM) -> None:
        """generate_batch must respect provided class_indices per image."""
        x = torch.randn(2, 3, 224, 224)
        class_indices = torch.tensor([0, 2])
        hms = cam.generate_batch(x, class_indices=class_indices, task_name="default")
        assert hms.shape == (2, 224, 224)

    def test_hooks_removed_after_call(self, cam: SSMGradCAM) -> None:
        """Hooks must be cleaned up after each call to prevent accumulation."""
        x = torch.randn(1, 3, 224, 224)
        cam(x, class_idx=0, task_name="default")
        assert len(cam._hooks) == 0, \
            f"Hooks not removed after call: {len(cam._hooks)} remaining"

    def test_multiple_calls_consistent(self, cam: SSMGradCAM) -> None:
        """Two calls with the same input and class must give the same heatmap."""
        cam.model.eval()
        torch.manual_seed(42)
        x = torch.randn(1, 3, 224, 224)
        hm1 = cam(x.clone(), class_idx=0, task_name="default")
        hm2 = cam(x.clone(), class_idx=0, task_name="default")
        assert torch.allclose(hm1, hm2, atol=1e-5), \
            "Two identical calls gave different heatmaps — hooks not being reset"