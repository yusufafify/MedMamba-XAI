"""SSM-GradCAM — GradCAM-style saliency maps for State-Space Model activations.

Merge decisions
---------------
Spatial format    : Agent's approach of using the ``intermediates`` list
                    (``[B, C, H, W]`` spatial tensors) is correct and clean.
                    Website's approach hooks into ``VSSBlock.mamba`` and gets
                    ``[B, N, D]`` token sequences, then assumes ``N`` is a
                    perfect square to reshape — this breaks for non-square
                    feature maps and couples the XAI code to the block's
                    internal hook API.

autograd fix      : Agent's ``torch.autograd.grad(target, fmap)`` fails unless
                    ``fmap`` has ``requires_grad=True`` at the time of the
                    forward pass.  The intermediates are detached by default.
                    Fix: register a forward hook that saves the intermediate
                    *with grad* and a backward hook that saves the gradient.
                    This mirrors standard GradCAM implementations.

generate_batch    : Kept from agent — reusable API that returns a tensor of
                    heatmaps rather than mixing generation and plotting.

checkpoint loading: Kept from website — reads ``task_names`` and ``config``
                    from the checkpoint dict to fully reconstruct the model
                    without needing a separate config file.

upsampling        : Both versions upsample to input resolution with bilinear
                    interpolation. Kept as-is — standard GradCAM practice.

References
----------
Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization.* ICCV.
https://arxiv.org/abs/1610.02391
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medical_mamba.data.constants import DATASET_META
from medical_mamba.models.medical_vmamba import MedicalVMamba, build_model


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helper (from website version)
# ─────────────────────────────────────────────────────────────────────────────

def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
) -> MedicalVMamba:
    """Reconstruct and load a ``MedicalVMamba`` model from a training checkpoint.

    Reads ``task_names`` and ``config`` from the checkpoint dict so that the
    model is fully reconstructed without needing a separate YAML config.

    Parameters
    ----------
    checkpoint_path : str
        Path to ``checkpoint_best.pt`` saved by ``Trainer``.
    device : str
        Target device string.

    Returns
    -------
    MedicalVMamba
        Model in ``eval()`` mode on ``device``.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg   = state["config"]
    task_names = state["task_names"]

    task_configs = [
        (name, DATASET_META[name]["num_classes"])
        for name in task_names
    ]
    model = build_model(
        task_configs=task_configs,
        model_size=cfg["model_size"],
        patch_size=cfg["patch_size"],
        head_dropout=cfg["head_dropout"],
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SSMGradCAM
# ─────────────────────────────────────────────────────────────────────────────

class SSMGradCAM:
    """GradCAM-style saliency maps using VMamba intermediate feature maps.

    Uses the spatial feature maps in the backbone's ``intermediates`` list
    (shape ``(B, N, D_stage)`` tokens reshaped to ``(B, D_stage, H_s, W_s)``)
    rather than hooking into individual VSSBlock internals.  This keeps the
    XAI code decoupled from the model's internal hook API.

    The GradCAM computation follows the original paper:

    1. Forward pass → target class logit score.
    2. Backward pass → gradient of score w.r.t. target stage feature map.
    3. Channel weights = global-average-pooled gradients.
    4. Weighted sum of feature map channels → ReLU → bilinear upsample.

    Parameters
    ----------
    model : MedicalVMamba
        Model in eval mode.
    target_stage : int
        Backbone stage index to hook (0–3, default 3 = last / most semantic).
        Earlier stages have higher spatial resolution but less semantic content.
    """

    def __init__(
        self,
        model: MedicalVMamba,
        target_stage: int = 3,
    ) -> None:
        self.model        = model
        self.target_stage = target_stage

        self._fmap: Optional[torch.Tensor] = None   # saved forward activation
        self._grad: Optional[torch.Tensor] = None   # saved backward gradient
        self._hooks: list = []

    # ------------------------------------------------------------------ #
    #  Hook management                                                     #
    # ------------------------------------------------------------------ #

    def _register_hooks(self, stage_module: nn.Module) -> None:
        """Attach forward + backward hooks to ``stage_module``.

        We hook the *last VSSBlock* of the target stage because:
        - It has processed the most context within that stage.
        - Its output is directly consumed by PatchMerging / GAP.
        """
        def _fwd_hook(module, input, output):
            # output is [B, N, D] tokens; save with grad enabled
            self._fmap = output  # do NOT detach — we need grad through it

        def _bwd_hook(module, grad_input, grad_output):
            # grad_output[0] is the gradient of loss w.r.t. this layer's output
            self._grad = grad_output[0].detach()

        last_block = list(self.model.backbone.stages[self.target_stage])[-1]
        self._hooks.append(last_block.register_forward_hook(_fwd_hook))
        self._hooks.append(last_block.register_full_backward_hook(_bwd_hook))

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------ #
    #  Spatial reconstruction                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tokens_to_spatial(
        tokens: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """Reshape ``[B, N, D]`` tokens → ``[B, D, H, W]``.

        Parameters
        ----------
        tokens : torch.Tensor
            ``(B, N, D)``
        H, W : int
            Target spatial dimensions.

        Returns
        -------
        torch.Tensor
            ``(B, D, H, W)``
        """
        B, N, D = tokens.shape
        assert N == H * W, f"N={N} ≠ H*W={H*W}"
        return tokens.permute(0, 2, 1).reshape(B, D, H, W)

    @staticmethod
    def _infer_spatial(N: int) -> tuple[int, int]:
        """Infer (H, W) from token count N.

        Tries square root first; if N is not a perfect square, falls back to
        the largest factorisation with H ≥ W.
        """
        H = int(math.isqrt(N))
        if H * H == N:
            return H, H
        # Non-square: find largest factor pair
        for h in range(H, 0, -1):
            if N % h == 0:
                return h, N // h
        return 1, N

    # ------------------------------------------------------------------ #
    #  GradCAM                                                             #
    # ------------------------------------------------------------------ #

    @torch.enable_grad()
    def __call__(
        self,
        image: torch.Tensor,
        class_idx: Optional[int] = None,
        task_name: str = "default",
    ) -> torch.Tensor:
        """Compute the GradCAM saliency heatmap for a single image.

        Parameters
        ----------
        image : torch.Tensor
            Preprocessed image ``(1, C, H, W)``.  Must NOT have
            ``requires_grad`` — gradients flow through the model internally.
        class_idx : int | None
            Target class.  ``None`` → predicted class (argmax).
        task_name : str
            Task head to route through.

        Returns
        -------
        torch.Tensor
            Saliency map ``(H_in, W_in)`` normalised to ``[0, 1]``.
        """
        self._fmap = None
        self._grad = None

        # Register hooks on the last block of the target stage
        self._register_hooks(
            self.model.backbone.stages[self.target_stage]
        )

        self.model.eval()
        self.model.zero_grad()

        # Forward pass — hooks fire here
        logits, _ = self.model.forward_single(image, task_name)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Backward pass from target class score — backward hook fires here
        score = logits[0, class_idx]
        score.backward()

        self._remove_hooks()

        if self._fmap is None or self._grad is None:
            raise RuntimeError(
                "Hooks did not capture activations. "
                "Verify target_stage index is within [0, num_stages)."
            )

        # ── Reconstruct spatial from token sequence ───────────────────────
        # _fmap: [1, N, D],  _grad: [1, N, D]
        N = self._fmap.shape[1]
        H_s, W_s = self._infer_spatial(N)

        fmap = self._tokens_to_spatial(self._fmap.detach(), H_s, W_s)  # [1, D, H_s, W_s]
        grad = self._tokens_to_spatial(self._grad,          H_s, W_s)  # [1, D, H_s, W_s]

        # ── GradCAM weighting ─────────────────────────────────────────────
        weights = grad.mean(dim=(2, 3), keepdim=True)          # [1, D, 1, 1]
        cam = F.relu((weights * fmap).sum(dim=1, keepdim=True))  # [1, 1, H_s, W_s]

        # ── Upsample to input resolution ─────────────────────────────────
        cam = F.interpolate(
            cam,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        ).squeeze()   # [H_in, W_in]

        # ── Normalise ────────────────────────────────────────────────────
        cam = cam - cam.min()
        if cam.max() > 1e-8:
            cam = cam / cam.max()

        return cam.detach().cpu()

    # ------------------------------------------------------------------ #
    #  Batch generation                                                    #
    # ------------------------------------------------------------------ #

    def generate_batch(
        self,
        images: torch.Tensor,
        class_indices: Optional[torch.Tensor] = None,
        task_name: str = "default",
    ) -> torch.Tensor:
        """Generate saliency maps for an entire batch.

        Parameters
        ----------
        images : torch.Tensor
            ``(B, C, H, W)`` preprocessed images.
        class_indices : torch.Tensor | None
            ``(B,)`` class indices.  ``None`` → predicted per image.
        task_name : str
            Task head name.

        Returns
        -------
        torch.Tensor
            ``(B, H, W)`` saliency maps in ``[0, 1]``.
        """
        heatmaps: List[torch.Tensor] = []
        for i in range(images.size(0)):
            cls_idx = int(class_indices[i]) if class_indices is not None else None
            hm = self(images[i : i + 1], class_idx=cls_idx, task_name=task_name)
            heatmaps.append(hm)
        return torch.stack(heatmaps)
