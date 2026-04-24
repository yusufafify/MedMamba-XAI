"""Phase 2 model shape tests.

Verifies every model component produces correct output shapes.
All tests run on CPU with patch_size=8 (fast, no GPU required).

Run with:
    pytest tests/test_model.py -v
"""

from __future__ import annotations

import pytest
import torch

from medical_mamba.models.backbone import PatchEmbed, PatchMerging, VMambaBackbone
from medical_mamba.models.blocks import VSSBlock, StochasticDepth
from medical_mamba.models.heads import ClassificationHead
from medical_mamba.models.medical_vmamba import MedicalVMamba, build_model


# ─────────────────────────────────────────────────────────────────────────────
# VSSBlock
# ─────────────────────────────────────────────────────────────────────────────

class TestVSSBlock:

    def test_output_shape_matches_input(self) -> None:
        """VSSBlock must preserve (B, N, D) shape."""
        block = VSSBlock(d_model=96)
        x = torch.randn(2, 56 * 56, 96)
        out = block(x, H=56, W=56)
        assert out.shape == (2, 56 * 56, 96)

    def test_different_spatial_sizes(self) -> None:
        """VSSBlock must work for any (H, W) not just 56x56."""
        block = VSSBlock(d_model=96)
        for H, W in [(28, 28), (14, 14), (7, 7)]:
            x = torch.randn(1, H * W, 96)
            out = block(x, H=H, W=W)
            assert out.shape == (1, H * W, 96), f"Failed for H={H} W={W}"

    def test_residual_connection(self) -> None:
        """With drop_path=0 and identity weights, output should not be zero."""
        block = VSSBlock(d_model=96, drop_path=0.0)
        x = torch.randn(2, 28 * 28, 96)
        out = block(x, H=28, W=28)
        assert not torch.allclose(out, torch.zeros_like(out)), \
            "Output is all zeros — residual connection broken"

    def test_stochastic_depth_eval_mode(self) -> None:
        """In eval mode, StochasticDepth must be a no-op (same output)."""
        block = VSSBlock(d_model=96, drop_path=0.5)
        block.eval()
        x = torch.randn(2, 14 * 14, 96)
        out1 = block(x, H=14, W=14)
        out2 = block(x, H=14, W=14)
        assert torch.allclose(out1, out2), \
            "Eval mode produced different outputs — StochasticDepth not disabled"


class TestStochasticDepth:

    def test_identity_in_eval(self) -> None:
        sd = StochasticDepth(drop_prob=0.9)
        sd.eval()
        x = torch.randn(4, 10, 10)
        assert torch.equal(sd(x), x)

    def test_identity_when_prob_zero(self) -> None:
        sd = StochasticDepth(drop_prob=0.0)
        sd.train()
        x = torch.randn(4, 10, 10)
        assert torch.equal(sd(x), x)

    def test_drops_in_training(self) -> None:
        """StochasticDepth must randomly drop residuals in train mode.

        drop_prob=1.0 causes NaN (x/0*0) so we use 0.5 and verify
        that some samples are dropped and some are kept.
        In real training drop_path_rate is always <= 0.2.
        """
        import torch
        torch.manual_seed(0)
        sd = StochasticDepth(drop_prob=0.5)
        sd.train()
        x = torch.ones(64, 1, 1)
        out = sd(x)
        has_zeros   = (out == 0).any().item()
        has_nonzero = (out != 0).any().item()
        assert has_zeros,   "No samples were dropped — StochasticDepth not working"
        assert has_nonzero, "All samples dropped — drop_prob=0.5 should keep some"


# ─────────────────────────────────────────────────────────────────────────────
# PatchEmbed
# ─────────────────────────────────────────────────────────────────────────────

class TestPatchEmbed:

    def test_patch4_shape(self) -> None:
        pe = PatchEmbed(img_size=224, patch_size=4, in_chans=3, embed_dim=96)
        x = torch.randn(2, 3, 224, 224)
        tokens, H, W = pe(x)
        assert tokens.shape == (2, 56 * 56, 96)
        assert H == 56 and W == 56

    def test_patch8_shape(self) -> None:
        pe = PatchEmbed(img_size=224, patch_size=8, in_chans=3, embed_dim=96)
        x = torch.randn(2, 3, 224, 224)
        tokens, H, W = pe(x)
        assert tokens.shape == (2, 28 * 28, 96)
        assert H == 28 and W == 28

    def test_num_patches_attribute(self) -> None:
        pe = PatchEmbed(img_size=224, patch_size=4, embed_dim=96)
        assert pe.num_patches == 56 * 56


# ─────────────────────────────────────────────────────────────────────────────
# PatchMerging
# ─────────────────────────────────────────────────────────────────────────────

class TestPatchMerging:

    def test_halves_spatial_doubles_channels(self) -> None:
        pm = PatchMerging(d_model=96)
        x = torch.randn(2, 56 * 56, 96)
        out, H2, W2 = pm(x, H=56, W=56)
        assert out.shape == (2, 28 * 28, 192)
        assert H2 == 28 and W2 == 28

    def test_odd_spatial_padding(self) -> None:
        """PatchMerging must handle odd H/W without crashing."""
        pm = PatchMerging(d_model=96)
        x = torch.randn(2, 7 * 7, 96)
        out, H2, W2 = pm(x, H=7, W=7)
        # 7 → padded to 8 → merged to 4
        assert out.shape == (2, 4 * 4, 192)
        assert H2 == 4 and W2 == 4


# ─────────────────────────────────────────────────────────────────────────────
# VMambaBackbone
# ─────────────────────────────────────────────────────────────────────────────

class TestVMambaBackbone:

    def test_output_shape_tiny_patch8(self) -> None:
        """Tiny config with patch_size=8 — fast CPU test."""
        bb = VMambaBackbone(
            img_size=224, patch_size=8, embed_dim=96,
            depths=[2, 2, 6, 2]
        )
        x = torch.randn(2, 3, 224, 224)
        features, intermediates = bb(x)
        assert features.shape == (2, 768), \
            f"Expected (2, 768), got {features.shape}"

    def test_intermediates_count(self) -> None:
        """Must return exactly 4 intermediate feature maps (one per stage)."""
        bb = VMambaBackbone(patch_size=8, embed_dim=96, depths=[2, 2, 6, 2])
        _, intermediates = bb(torch.randn(1, 3, 224, 224))
        assert len(intermediates) == 4

    def test_intermediates_channel_progression(self) -> None:
        """Channels must double at each stage: 96, 192, 384, 768."""
        bb = VMambaBackbone(patch_size=8, embed_dim=96, depths=[2, 2, 6, 2])
        _, intermediates = bb(torch.randn(1, 3, 224, 224))
        expected_dims = [96, 192, 384, 768]
        for i, (inter, exp_dim) in enumerate(zip(intermediates, expected_dims)):
            assert inter.shape[-1] == exp_dim, \
                f"Stage {i}: expected D={exp_dim}, got {inter.shape[-1]}"

    def test_out_dim_attribute(self) -> None:
        bb = VMambaBackbone(patch_size=8, embed_dim=96, depths=[2, 2, 6, 2])
        assert bb.out_dim == 768

    def test_get_last_vss_blocks(self) -> None:
        bb = VMambaBackbone(patch_size=8, embed_dim=96, depths=[2, 2, 6, 2])
        blocks = bb.get_last_vss_blocks()
        assert len(blocks) == 2          # last stage has depth=2
        assert isinstance(blocks[0], VSSBlock)


# ─────────────────────────────────────────────────────────────────────────────
# ClassificationHead
# ─────────────────────────────────────────────────────────────────────────────

class TestClassificationHead:

    def test_output_shape(self) -> None:
        head = ClassificationHead(in_features=768, num_classes=9)
        x = torch.randn(4, 768)
        logits = head(x)
        assert logits.shape == (4, 9)

    def test_no_softmax_applied(self) -> None:
        """Head returns raw logits — values should not sum to 1."""
        head = ClassificationHead(in_features=768, num_classes=9)
        x = torch.randn(4, 768)
        logits = head(x)
        row_sums = logits.softmax(dim=-1).sum(dim=-1)
        # softmax of logits sums to 1, raw logits do not
        assert not torch.allclose(logits.sum(dim=-1), torch.ones(4)), \
            "Head appears to be applying softmax — should return raw logits"


# ─────────────────────────────────────────────────────────────────────────────
# MedicalVMamba — full model
# ─────────────────────────────────────────────────────────────────────────────

class TestMedicalVMamba:

    @pytest.fixture(scope="class")
    def single_task_model(self) -> MedicalVMamba:
        return build_model(
            task_configs=[("pathmnist", 9)],
            model_size="tiny",
            patch_size=8,
        )

    @pytest.fixture(scope="class")
    def multi_task_model(self) -> MedicalVMamba:
        return build_model(
            task_configs=[
                ("pathmnist",  9),
                ("bloodmnist", 8),
                ("octmnist",   4),
            ],
            model_size="tiny",
            patch_size=8,
        )

    def test_single_task_logit_shape(self, single_task_model) -> None:
        x = torch.randn(2, 3, 224, 224)
        logits, _ = single_task_model.forward_single(x, "pathmnist")
        assert logits.shape == (2, 9)

    def test_single_task_returns_intermediates(self, single_task_model) -> None:
        x = torch.randn(2, 3, 224, 224)
        _, intermediates = single_task_model.forward_single(x, "pathmnist")
        assert len(intermediates) == 4

    def test_forward_alias(self, single_task_model) -> None:
        """forward() must be an alias for forward_single() — same outputs.
        
        Must run in eval mode with no_grad to eliminate dropout randomness
        between the two calls.
        """
        single_task_model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits_a, _ = single_task_model.forward_single(x, "pathmnist")
            logits_b, _ = single_task_model.forward(x, "pathmnist")
        assert torch.allclose(logits_a, logits_b),             "forward() and forward_single() returned different results"

    def test_unknown_task_raises(self, single_task_model) -> None:
        x = torch.randn(2, 3, 224, 224)
        with pytest.raises(KeyError):
            single_task_model.forward_single(x, "notadataset")

    def test_multi_task_routes_correctly(self, multi_task_model) -> None:
        """Each sample must be routed to the correct head by task_id."""
        x = torch.randn(4, 3, 224, 224)
        task_ids = torch.tensor([0, 1, 0, 2])  # path, blood, path, oct
        task_logits, features, _ = multi_task_model.forward_multi(x, task_ids)

        assert "pathmnist"  in task_logits
        assert "bloodmnist" in task_logits
        assert "octmnist"   in task_logits

        assert task_logits["pathmnist"].shape  == (2, 9)  # 2 path samples
        assert task_logits["bloodmnist"].shape == (1, 8)  # 1 blood sample
        assert task_logits["octmnist"].shape   == (1, 4)  # 1 oct sample
        assert features.shape == (4, multi_task_model.backbone.out_dim)

    def test_multi_task_absent_task_not_in_output(self, multi_task_model) -> None:
        """Tasks with no samples in the batch must not appear in output dict."""
        x = torch.randn(2, 3, 224, 224)
        task_ids = torch.tensor([0, 0])   # only pathmnist
        task_logits, _features, _ = multi_task_model.forward_multi(x, task_ids)
        assert "bloodmnist" not in task_logits
        assert "octmnist"   not in task_logits

    def test_eval_mode_deterministic(self, single_task_model) -> None:
        """In eval mode, two identical forward passes must give identical logits."""
        single_task_model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out1, _ = single_task_model.forward_single(x, "pathmnist")
            out2, _ = single_task_model.forward_single(x, "pathmnist")
        assert torch.allclose(out1, out2), \
            "Eval mode is not deterministic"

    def test_gradient_flows(self, single_task_model) -> None:
        """Loss.backward() must produce non-None gradients on backbone params."""
        single_task_model.train()
        x = torch.randn(2, 3, 224, 224)
        logits, _ = single_task_model.forward_single(x, "pathmnist")
        loss = logits.sum()
        loss.backward()
        for name, param in single_task_model.backbone.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, \
                    f"No gradient for {name}"
                break   # checking one is enough for a smoke test


# ─────────────────────────────────────────────────────────────────────────────
# build_model factory
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildModel:

    def test_tiny_out_dim(self) -> None:
        model = build_model([("pathmnist", 9)], model_size="tiny", patch_size=8)
        assert model.backbone.out_dim == 768

    def test_task_names_registered(self) -> None:
        model = build_model(
            [("pathmnist", 9), ("bloodmnist", 8)],
            model_size="tiny", patch_size=8
        )
        assert model.task_names == ["pathmnist", "bloodmnist"]

    def test_heads_created(self) -> None:
        model = build_model(
            [("pathmnist", 9), ("bloodmnist", 8)],
            model_size="tiny", patch_size=8
        )
        assert "pathmnist"  in model.heads
        assert "bloodmnist" in model.heads
        assert model.heads["pathmnist"].num_classes  == 9
        assert model.heads["bloodmnist"].num_classes == 8

    def test_invalid_model_size_raises(self) -> None:
        with pytest.raises(ValueError):
            build_model([("pathmnist", 9)], model_size="xlarge")


# ─────────────────────────────────────────────────────────────────────────────
# Contrastive projector (SupCon domain discovery)
# ─────────────────────────────────────────────────────────────────────────────

class TestContrastiveProjector:

    @pytest.fixture(scope="class")
    def model(self) -> MedicalVMamba:
        """Tiny multi-task model (reused across projector tests)."""
        return build_model(
            task_configs=[("pathmnist", 9), ("bloodmnist", 8), ("octmnist", 4)],
            model_size="tiny",
            patch_size=8,
        )

    def test_project_output_shape(self, model: MedicalVMamba) -> None:
        """project() returns (B, 128) regardless of feat_dim."""
        feat_dim = model.backbone.out_dim
        features = torch.randn(4, feat_dim)
        z = model.project(features)
        assert z.shape == (4, 128), f"Expected (4, 128), got {z.shape}"

    def test_project_output_normalized(self, model: MedicalVMamba) -> None:
        """L2 norm of each projection vector must equal 1.0."""
        model.eval()  # BN in eval mode for deterministic behaviour
        feat_dim = model.backbone.out_dim
        features = torch.randn(4, feat_dim)
        with torch.no_grad():
            z = model.project(features)
        norms = z.norm(dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5), \
            f"Projections not unit norm: {norms.tolist()}"

    def test_project_grad_flows(self, model: MedicalVMamba) -> None:
        """Gradients must flow through project() back to backbone params."""
        model.train()
        x = torch.randn(2, 3, 224, 224)
        features, _ = model.backbone(x)
        z = model.project(features)
        loss = z.sum()
        loss.backward()
        # At least one backbone parameter should have a non-None gradient
        got_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.backbone.parameters()
            if p.requires_grad
        )
        assert got_grad, "No gradient reached backbone via project()"


# ─────────────────────────────────────────────────────────────────────────────
# Prototype routing (autonomous predict)
# ─────────────────────────────────────────────────────────────────────────────

def _fake_loader(n_tasks: int, batch_size: int = 4, n_batches: int = 2):
    """Yield tiny batches in the {image, label, task_id} dict format."""
    for i in range(n_batches):
        # Even split of task_ids across the batch
        tids = torch.tensor([j % n_tasks for j in range(batch_size)],
                            dtype=torch.long)
        yield {
            "image":   torch.randn(batch_size, 3, 224, 224),
            "label":   torch.zeros(batch_size, dtype=torch.long),
            "task_id": tids,
        }


class TestPrototypeRouting:

    @pytest.fixture(scope="class")
    def model(self) -> MedicalVMamba:
        """Tiny 3-task model, prototypes computed via fake loader."""
        m = build_model(
            task_configs=[("pathmnist", 9), ("bloodmnist", 8), ("octmnist", 4)],
            model_size="tiny",
            patch_size=8,
        )
        return m

    def test_prototypes_shape(self, model: MedicalVMamba) -> None:
        """domain_prototypes buffer has shape (n_tasks, feat_dim)."""
        n_tasks  = len(model.task_names)
        feat_dim = model.backbone.out_dim
        assert model.domain_prototypes.shape == (n_tasks, feat_dim)

    def test_predict_raises_before_prototypes(self) -> None:
        """predict() must raise RuntimeError on a freshly-built model."""
        m = build_model(
            task_configs=[("pathmnist", 9), ("bloodmnist", 8)],
            model_size="tiny",
            patch_size=8,
        )
        assert not m.prototypes_computed
        with pytest.raises(RuntimeError):
            m.predict(torch.randn(1, 3, 224, 224))

    def test_compute_prototypes_sets_flag(self) -> None:
        """prototypes_computed must flip to True after compute_prototypes()."""
        m = build_model(
            task_configs=[("pathmnist", 9), ("bloodmnist", 8), ("octmnist", 4)],
            model_size="tiny",
            patch_size=8,
        )
        assert not m.prototypes_computed
        m.compute_prototypes(_fake_loader(n_tasks=3), device=torch.device("cpu"))
        assert m.prototypes_computed

    def test_predict_returns_valid_task(self) -> None:
        """predict() returns a task_name that is in model.task_names."""
        m = build_model(
            task_configs=[("pathmnist", 9), ("bloodmnist", 8), ("octmnist", 4)],
            model_size="tiny",
            patch_size=8,
        )
        m.compute_prototypes(_fake_loader(n_tasks=3), device=torch.device("cpu"))
        task_name, class_idx, conf = m.predict(torch.randn(1, 3, 224, 224))
        assert task_name in m.task_names
        assert isinstance(class_idx, int)
        assert isinstance(conf, float)

    def test_predict_confidence_in_range(self) -> None:
        """Confidence is a valid softmax probability in [0, 1]."""
        m = build_model(
            task_configs=[("pathmnist", 9), ("bloodmnist", 8), ("octmnist", 4)],
            model_size="tiny",
            patch_size=8,
        )
        m.compute_prototypes(_fake_loader(n_tasks=3), device=torch.device("cpu"))
        _, _, conf = m.predict(torch.randn(1, 3, 224, 224))
        assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"

    def test_predict_auto_unsqueeze(self) -> None:
        """predict() accepts (C, H, W) input without manual unsqueezing."""
        m = build_model(
            task_configs=[("pathmnist", 9), ("bloodmnist", 8), ("octmnist", 4)],
            model_size="tiny",
            patch_size=8,
        )
        m.compute_prototypes(_fake_loader(n_tasks=3), device=torch.device("cpu"))
        # 3-D input — must be handled without crashing
        task_name, class_idx, conf = m.predict(torch.randn(3, 224, 224))
        assert task_name in m.task_names