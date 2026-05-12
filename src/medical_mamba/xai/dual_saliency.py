"""Dual-pathway saliency — the architecture's distinguishing XAI contribution.

Standard GradCAM produces ONE saliency map per input: "which pixels drove
the predicted class." This module produces TWO complementary maps for the
same input:

  1. **Class saliency**   — gradient of  logits[predicted_class]
                            (which pixels made it look like class Y)
  2. **Domain saliency**  — gradient of  cos(features, prototype[predicted_domain])
                            (which pixels made it look like modality D)

When the architecture's class/domain disentanglement claim is real, these
two maps highlight DIFFERENT pixels: domain saliency emphasises global
texture / colour / acquisition signatures, while class saliency emphasises
local lesion- or class-specific structure. The empirical observation that
they differ is the scientific finding.

Implementation reuses the same forward+backward hook pattern as
``SSMGradCAM``: hook the last VSSBlock of the target stage, capture the
``[B, N, D]`` SSM activations on forward, capture the gradient of the
target scalar on backward, then weighted-sum and bilinear-upsample to
input resolution.

Reference
---------
Selvaraju et al. (2017). *Grad-CAM*. ICCV. arXiv:1610.02391.
For the domain-saliency direction (gradient of similarity to a
representation prototype), the closest prior work is Score-CAM and
Eigen-CAM but applied to a learned prototype rather than a class logit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from medical_mamba.models.medical_vmamba import MedicalVMamba


@dataclass
class DualSaliencyResult:
    """Single-image dual-pathway saliency output."""

    class_map:        torch.Tensor   # (H_in, W_in) ∈ [0, 1]
    domain_map:       torch.Tensor   # (H_in, W_in) ∈ [0, 1]
    predicted_task:   str
    predicted_class:  int
    class_confidence: float
    domain_similarity: float


class DualPathwaySaliency:
    """GradCAM-style saliency for both class and domain decisions.

    Parameters
    ----------
    model : MedicalVMamba
        Trained model with ``domain_prototypes`` populated.
    target_stage : int
        Backbone stage to hook (default 3 = last). Earlier stages give
        higher spatial resolution but less semantic content.
    """

    def __init__(self, model: MedicalVMamba, target_stage: int = 3) -> None:
        self.model        = model
        self.target_stage = target_stage

        if not getattr(model, "prototypes_computed", False):
            raise RuntimeError(
                "Model has prototypes_computed=False — domain saliency requires "
                "domain prototypes. Run scripts/recompute_prototypes_per_dataset.py "
                "first."
            )

        # Hook state — written by forward/backward hooks, consumed below
        self._fmap: Optional[torch.Tensor] = None
        self._grad: Optional[torch.Tensor] = None
        self._hooks: list = []

    # ------------------------------------------------------------------ #
    #  Hook management (same pattern as SSMGradCAM)                       #
    # ------------------------------------------------------------------ #

    def _register_hooks(self) -> None:
        last_block = list(self.model.backbone.stages[self.target_stage])[-1]

        def _fwd(module, inp, out):
            # out is [B, N, D] — keep grad enabled (do NOT detach)
            self._fmap = out

        def _bwd(module, grad_in, grad_out):
            self._grad = grad_out[0].detach()

        self._hooks.append(last_block.register_forward_hook(_fwd))
        self._hooks.append(last_block.register_full_backward_hook(_bwd))

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------ #
    #  Spatial reconstruction (mirrors SSMGradCAM)                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _infer_spatial(N: int) -> Tuple[int, int]:
        H = int(math.isqrt(N))
        if H * H == N:
            return H, H
        for h in range(H, 0, -1):
            if N % h == 0:
                return h, N // h
        return 1, N

    @staticmethod
    def _tokens_to_map(fmap: torch.Tensor, grad: torch.Tensor,
                       target_h: int, target_w: int) -> torch.Tensor:
        """GradCAM-style weighted sum + ReLU + upsample.

        Parameters
        ----------
        fmap, grad : torch.Tensor
            ``(1, N, D)``  forward activations and their gradients.
        target_h, target_w : int
            Output spatial size (matches input image).

        Returns
        -------
        torch.Tensor
            ``(target_h, target_w)`` saliency map normalised to [0, 1].
        """
        N = fmap.shape[1]
        H_s, W_s = DualPathwaySaliency._infer_spatial(N)

        fmap_sp = fmap.permute(0, 2, 1).reshape(1, -1, H_s, W_s)   # (1, D, H_s, W_s)
        grad_sp = grad.permute(0, 2, 1).reshape(1, -1, H_s, W_s)

        # Channel weights = global-average gradient
        weights = grad_sp.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * fmap_sp).sum(dim=1, keepdim=True))   # (1, 1, H_s, W_s)

        cam = F.interpolate(cam, size=(target_h, target_w),
                            mode="bilinear", align_corners=False).squeeze()
        cam = cam - cam.min()
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        return cam.detach().cpu()

    # ------------------------------------------------------------------ #
    #  Per-pathway saliency                                                #
    # ------------------------------------------------------------------ #

    @torch.enable_grad()
    def class_saliency(
        self,
        image: torch.Tensor,
        task_name: str,
        class_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, float]:
        """Saliency for the class decision.

        Returns
        -------
        cam : torch.Tensor
            ``(H_in, W_in)`` ∈ [0, 1].
        class_idx : int
            The class actually used (predicted if ``class_idx=None`` was passed).
        confidence : float
            Softmax probability of the chosen class.
        """
        self._fmap = self._grad = None
        self._register_hooks()
        self.model.eval()
        self.model.zero_grad()

        logits, _ = self.model.forward_single(image, task_name)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        confidence = float(logits.softmax(dim=-1)[0, class_idx].item())

        score = logits[0, class_idx]
        score.backward()
        self._remove_hooks()

        if self._fmap is None or self._grad is None:
            raise RuntimeError("class_saliency: hooks did not capture activations")

        cam = self._tokens_to_map(self._fmap.detach(), self._grad,
                                   image.shape[2], image.shape[3])
        return cam, class_idx, confidence

    @torch.enable_grad()
    def domain_saliency(
        self,
        image: torch.Tensor,
        domain_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, float]:
        """Saliency for the domain (modality) decision.

        Computes the gradient of cosine similarity between the pooled
        backbone features and the chosen domain prototype, then propagates
        that signal back to the last VSSBlock activations.

        Returns
        -------
        cam : torch.Tensor
            ``(H_in, W_in)`` ∈ [0, 1].
        domain_idx : int
            The domain actually used (predicted if ``None`` was passed).
        similarity : float
            Cosine similarity to the chosen prototype.
        """
        self._fmap = self._grad = None
        self._register_hooks()
        self.model.eval()
        self.model.zero_grad()

        # Forward through the backbone with grad enabled
        features, _ = self.model.backbone(image)              # (1, feat_dim)
        # Routing decision
        sim_all = F.cosine_similarity(
            features, self.model.domain_prototypes, dim=-1,
        )                                                     # (n_tasks,)
        if domain_idx is None:
            domain_idx = int(sim_all.argmax().item())
        similarity = float(sim_all[domain_idx].item())

        # Backprop the chosen prototype's similarity
        score = F.cosine_similarity(
            features,
            self.model.domain_prototypes[domain_idx:domain_idx + 1],
            dim=-1,
        ).squeeze()                                           # scalar
        score.backward()
        self._remove_hooks()

        if self._fmap is None or self._grad is None:
            raise RuntimeError("domain_saliency: hooks did not capture activations")

        cam = self._tokens_to_map(self._fmap.detach(), self._grad,
                                   image.shape[2], image.shape[3])
        return cam, domain_idx, similarity

    # ------------------------------------------------------------------ #
    #  Combined: both maps for the same input                             #
    # ------------------------------------------------------------------ #

    def both(
        self,
        image: torch.Tensor,
        task_name: Optional[str] = None,
        class_idx: Optional[int]  = None,
        domain_idx: Optional[int] = None,
    ) -> DualSaliencyResult:
        """Compute both class and domain saliency for the same image.

        If ``task_name`` is None, the predicted domain is used and that
        domain's task is used for class saliency.
        """
        # Auto-pick the target domain (and therefore the task head) if not given
        if task_name is None:
            with torch.no_grad():
                features, _ = self.model.backbone(image)
                sim = F.cosine_similarity(features, self.model.domain_prototypes, dim=-1)
                if domain_idx is None:
                    domain_idx = int(sim.argmax().item())
                task_name = self.model.task_names[domain_idx]

        # Resolve the predicted class first (we need it before domain saliency
        # would overwrite the hook state).
        cls_map, cls_idx, cls_conf = self.class_saliency(image, task_name, class_idx)
        dom_map, dom_idx, dom_sim  = self.domain_saliency(image, domain_idx)

        return DualSaliencyResult(
            class_map=cls_map,
            domain_map=dom_map,
            predicted_task=self.model.task_names[dom_idx],
            predicted_class=cls_idx,
            class_confidence=cls_conf,
            domain_similarity=dom_sim,
        )
