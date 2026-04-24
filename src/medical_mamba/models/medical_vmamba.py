"""MedicalVMamba — full model: backbone + GAP + per-task heads + XAI.

Design decisions vs the two source versions
--------------------------------------------
Forward signatures : model.py had ``forward(images, task_ids)`` for
                     multi-task and ``forward_single_task(images, task_name)``
                     for single-task.  The agent had ``forward(x, task=str)``
                     and ``forward_multitask(x, task_ids, task_names)``.

                     The merge keeps *both* signatures under clearer names:
                       • ``forward_single(x, task_name)``   — clean single-task
                       • ``forward_multi(x, task_ids)``     — multi-task routing
                     and makes ``forward`` an alias for ``forward_single`` so
                     the model works as a drop-in in standard training loops.

Intermediates      : The backbone now returns ``(features, intermediates)``.
                     Both forward methods surface the intermediates so XAI and
                     future dense heads can consume them.

XAI               : GradCAM-style saliency from model.py, operating on the
                     ``[B, N, D]`` SSM activations of the last VSSBlock.

build_model factory: From model.py, with tiny / small / base presets and VRAM
                     estimates.  The agent had no factory.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from medical_mamba.models.backbone import VMambaBackbone
from medical_mamba.models.blocks import VSSBlock
from medical_mamba.models.heads import ClassificationHead


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class MedicalVMamba(nn.Module):
    """Universal Medical Encoder with shared VMamba backbone and per-task heads.

    Parameters
    ----------
    task_configs : List[Tuple[str, int]]
        Ordered list of ``(task_name, num_classes)`` pairs.
        Position in the list = ``task_id`` value from the DataLoader.
    backbone_cfg : dict, optional
        Keyword arguments forwarded to ``VMambaBackbone.__init__``.
        If ``None``, the backbone uses its own defaults (Tiny config).
    head_dropout : float
        Dropout probability inside each ``ClassificationHead``.

    Examples
    --------
    Single-dataset mode::

        model = MedicalVMamba(task_configs=[("pathmnist", 9)])
        logits = model.forward_single(images, "pathmnist")

    Multi-task mode::

        model = MedicalVMamba(
            task_configs=[
                ("pathmnist",  9),
                ("bloodmnist", 8),
                ("dermamnist", 7),
                ("octmnist",   4),
            ]
        )
        task_logits, features, _ = model.forward_multi(images, task_ids)
    """

    def __init__(
        self,
        task_configs: List[Tuple[str, int]],
        backbone_cfg: Optional[Dict] = None,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ── Backbone ─────────────────────────────────────────────────────
        cfg = backbone_cfg or {}
        self.backbone = VMambaBackbone(**cfg)
        feat_dim = self.backbone.out_dim

        # ── Task registry ────────────────────────────────────────────────
        self.task_names: List[str] = [name for name, _ in task_configs]

        # ── Per-task classification heads ────────────────────────────────
        self.heads = nn.ModuleDict({
            name: ClassificationHead(feat_dim, n_cls, head_dropout)
            for name, n_cls in task_configs
        })

        # ── Contrastive domain projector (training only) ──────────────────
        # MLP 768→512→128 used with SupCon for domain-discriminative
        # representations. L2-normalisation is applied in project().
        # Params fall into the "head" optimizer group via the trainer's
        # naming filter; no separate group needed.
        self.domain_projector = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
        )

        # ── Domain prototypes (inference routing) ─────────────────────────
        # Mean backbone features per task_id, computed post-training by
        # compute_prototypes(). Registered as buffer so they are saved
        # inside state_dict and moved with .to(device).
        self.register_buffer(
            "domain_prototypes",
            torch.zeros(len(task_configs), feat_dim),
        )
        self.prototypes_computed: bool = False

        # ── XAI state ────────────────────────────────────────────────────
        self._xai_enabled: bool = False

    # ------------------------------------------------------------------ #
    #  Forward — single task (primary interface for training)             #
    # ------------------------------------------------------------------ #

    def forward_single(
        self,
        x: torch.Tensor,
        task_name: str,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the model for a single task.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, C, H, W)``
        task_name : str
            Must be a key in ``self.heads``.

        Returns
        -------
        logits : torch.Tensor
            ``(B, num_classes)``
        intermediates : List[torch.Tensor]
            Per-stage feature maps from the backbone.
        """
        if task_name not in self.heads:
            raise KeyError(
                f"Unknown task '{task_name}'. "
                f"Registered tasks: {self.task_names}"
            )
        features, intermediates = self.backbone(x)
        logits = self.heads[task_name](features)
        return logits, intermediates

    def forward(
        self,
        x: torch.Tensor,
        task_name: str,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Alias for ``forward_single`` — keeps the model a drop-in for
        standard ``output = model(batch, task)`` training loops."""
        return self.forward_single(x, task_name)

    # ------------------------------------------------------------------ #
    #  Forward — multi-task                                               #
    # ------------------------------------------------------------------ #

    def forward_multi(
        self,
        x: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        """Route each sample in the batch to its task-specific head.

        All samples share one backbone forward pass (efficient), then logits
        are split by ``task_id`` mask. Pooled backbone features are also
        returned so downstream contrastive projection heads can consume them
        without a second backbone pass.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, C, H, W)``
        task_ids : torch.Tensor
            Integer tensor ``(B,)`` with values in ``[0, n_tasks)``.

        Returns
        -------
        task_logits : Dict[str, torch.Tensor]
            Maps task name → logits for the samples belonging to that task.
            Tasks absent from the batch are not included in the dict.
        features : torch.Tensor
            Pooled backbone features ``(B, feat_dim)`` — the full batch
            (not masked), so the caller can feed them straight to
            ``project()`` / contrastive loss.
        intermediates : List[torch.Tensor]
            Per-stage feature maps from the backbone.
        """
        features, intermediates = self.backbone(x)

        task_logits: Dict[str, torch.Tensor] = {}
        for tid, name in enumerate(self.task_names):
            mask = task_ids == tid
            if mask.any():
                task_logits[name] = self.heads[name](features[mask])

        return task_logits, features, intermediates

    # ------------------------------------------------------------------ #
    #  Contrastive projection + prototype routing                         #
    # ------------------------------------------------------------------ #

    def project(self, features: torch.Tensor) -> torch.Tensor:
        """L2-normalised ``(B, 128)`` projections for SupCon loss.

        Used during training only — the contrastive head is a throwaway
        auxiliary branch. At inference, features go straight to prototype
        matching without passing through this projector.

        Parameters
        ----------
        features : torch.Tensor
            Pooled backbone features ``(B, feat_dim)``.

        Returns
        -------
        torch.Tensor
            ``(B, 128)`` L2-normalised projections (unit norm along dim=1).
        """
        z = self.domain_projector(features)
        return F.normalize(z, dim=1, p=2)

    @torch.no_grad()
    def compute_prototypes(self, dataloader, device) -> None:
        """Compute per-task mean backbone features and store them.

        Iterates the dataloader in eval mode (no grad), accumulates
        running sums of backbone features bucketed by ``task_id``, then
        divides by counts to obtain mean prototypes. Sets
        ``self.prototypes_computed = True`` on completion.

        Must be called after training for autonomous ``predict()`` to work.

        Parameters
        ----------
        dataloader : Iterable[Dict]
            Yields batches of ``{"image", "label", "task_id"}`` — typically
            the training dataloader used during ``Trainer.fit()``.
        device : torch.device or str
            Device to run the backbone on.
        """
        was_training = self.training
        self.eval()
        n_tasks = len(self.task_names)
        feat_dim = self.domain_prototypes.size(1)
        sums   = torch.zeros(n_tasks, feat_dim, device=device)
        counts = torch.zeros(n_tasks, device=device)

        for batch in tqdm(dataloader, desc="Computing prototypes", leave=False):
            images   = batch["image"].to(device, non_blocking=True)
            task_ids = batch["task_id"].to(device, non_blocking=True)
            features, _ = self.backbone(images)     # (B, feat_dim)
            for tid in range(n_tasks):
                mask = task_ids == tid
                if mask.any():
                    sums[tid]   += features[mask].sum(dim=0)
                    counts[tid] += mask.sum()

        counts = counts.clamp_min(1.0)               # avoid /0 for empty tasks
        self.domain_prototypes.copy_(sums / counts.unsqueeze(1))
        self.prototypes_computed = True
        self.train(was_training)

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> Tuple[str, int, float]:
        """Fully autonomous prediction — no ``task_id`` required.

        Runs the backbone, routes to the best-matching domain by cosine
        similarity against the learned prototypes, then runs that task's
        classification head. This is the 1-NN classifier in representation
        space (Snell et al. 2017, Prototypical Networks) paired with the
        per-task heads.

        Parameters
        ----------
        image : torch.Tensor
            ``(1, C, H, W)`` or ``(C, H, W)`` — auto-unsqueezed to 4-D.

        Returns
        -------
        task_name : str
            Predicted dataset name (e.g. ``"dermamnist"``).
        class_idx : int
            Predicted class within that dataset.
        confidence : float
            Softmax confidence of the class prediction in ``[0, 1]``.

        Raises
        ------
        RuntimeError
            If ``compute_prototypes()`` has not been called yet.
        """
        if not self.prototypes_computed:
            raise RuntimeError(
                "Domain prototypes have not been computed. "
                "Call compute_prototypes(dataloader, device) after training "
                "before using predict()."
            )

        if image.dim() == 3:
            image = image.unsqueeze(0)

        was_training = self.training
        self.eval()
        features, _ = self.backbone(image)                            # (1, feat_dim)
        sim = F.cosine_similarity(features, self.domain_prototypes)   # (n_tasks,)
        domain_idx = int(sim.argmax().item())
        task_name  = self.task_names[domain_idx]

        logits = self.heads[task_name](features)
        probs  = logits.softmax(dim=-1)
        class_idx  = int(logits.argmax(dim=-1).item())
        confidence = float(probs.max().item())
        self.train(was_training)

        return task_name, class_idx, confidence

    # ------------------------------------------------------------------ #
    #  XAI                                                                #
    # ------------------------------------------------------------------ #

    def enable_xai(self) -> None:
        """Register forward + backward hooks on the last-stage VSSBlocks.

        Must be called *before* the forward pass you want to explain.
        Hooks are cumulative — avoid calling this more than once per block.
        """
        self._xai_enabled = True
        for block in self.backbone.get_last_vss_blocks():
            block.register_xai_hooks()

    @torch.enable_grad()
    def get_saliency_map(
        self,
        image: torch.Tensor,
        task_name: str,
        class_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute a GradCAM-style saliency map using SSM activations.

        Adapted from standard GradCAM to work on the ``[B, N, D]`` token
        representation of the last VSSBlock's Mamba output rather than CNN
        feature maps.

        Parameters
        ----------
        image : torch.Tensor
            Single image ``(1, C, H, W)`` — must have ``requires_grad=False``.
        task_name : str
            Task head to use for the score.
        class_idx : int, optional
            Target class.  Defaults to ``argmax`` of the predicted logits.

        Returns
        -------
        torch.Tensor
            ``(224, 224)`` float tensor in ``[0, 1]``.
        """
        if image.dim() != 4 or image.size(0) != 1:
            raise ValueError("image must have shape (1, C, H, W)")

        self.enable_xai()
        last_block: VSSBlock = self.backbone.get_last_vss_blocks()[-1]

        image = image.clone().requires_grad_(True)
        logits, _ = self.forward_single(image, task_name)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.zero_grad()
        logits[0, class_idx].backward()

        acts  = last_block.activations   # (1, N, D)
        grads = last_block.gradients     # (1, N, D)

        if acts is None or grads is None:
            raise RuntimeError(
                "XAI hooks did not capture activations. "
                "Call enable_xai() before the forward pass."
            )

        # GradCAM: weight each channel by the gradient's global average
        weights = grads.mean(dim=1, keepdim=True)    # (1, 1, D)
        cam = F.relu((weights * acts).sum(dim=-1))   # (1, N)

        # Reshape to spatial grid
        N    = cam.size(1)
        H_s  = int(math.isqrt(N))
        W_s  = N // H_s
        if H_s * W_s != N:
            raise RuntimeError(f"Cannot reshape {N} tokens into a 2-D grid.")

        cam = cam.view(1, 1, H_s, W_s)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze()                           # (224, 224)

        # Normalise to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 1e-8:
            cam = cam / cam.max()

        return cam.detach().cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

# Preset configurations matching published VMamba variants
_MODEL_CONFIGS: Dict[str, Dict] = {
    #              embed_dim  depths          VRAM @ bs=16, 224², A100-40GB
    "tiny":  {"embed_dim": 96,  "depths": [2, 2,  6, 2]},   # ~8 GB
    "small": {"embed_dim": 96,  "depths": [2, 2, 18, 2]},   # ~14 GB
    "base":  {"embed_dim": 128, "depths": [2, 2, 12, 2]},   # ~22 GB
}


def build_model(
    task_configs: List[Tuple[str, int]],
    model_size:   str   = "tiny",
    patch_size:   int   = 4,
    head_dropout: float = 0.1,
    **backbone_kwargs,
) -> MedicalVMamba:
    """Construct a ``MedicalVMamba`` from a named size preset.

    Parameters
    ----------
    task_configs : List[Tuple[str, int]]
        ``[(task_name, num_classes), ...]``
    model_size : str
        One of ``"tiny"``, ``"small"``, ``"base"``.
    patch_size : int
        ``4`` (default, high-res) or ``8`` (faster, ~4× less VRAM).
    head_dropout : float
        Head dropout probability.
    **backbone_kwargs
        Any extra kwargs forwarded to ``VMambaBackbone`` (e.g.
        ``drop_path_rate``, ``d_state``).

    Returns
    -------
    MedicalVMamba

    Examples
    --------
    ::

        # Single dataset, VRAM-constrained GPU
        model = build_model([("pathmnist", 9)], model_size="tiny", patch_size=8)

        # Full multi-task
        model = build_model(
            task_configs=[("pathmnist", 9), ("bloodmnist", 8),
                          ("dermamnist", 7), ("octmnist", 4)],
            model_size="tiny",
        )
    """
    if model_size not in _MODEL_CONFIGS:
        raise ValueError(
            f"model_size must be one of {list(_MODEL_CONFIGS.keys())}, "
            f"got '{model_size}'"
        )

    backbone_cfg = {
        **_MODEL_CONFIGS[model_size],
        "patch_size": patch_size,
        **backbone_kwargs,
    }

    return MedicalVMamba(
        task_configs=task_configs,
        backbone_cfg=backbone_cfg,
        head_dropout=head_dropout,
    )
