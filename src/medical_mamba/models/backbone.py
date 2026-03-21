"""VMamba backbone: PatchEmbed → staged VSSBlocks → PatchMerging → GAP.

Design decisions vs the two source versions
--------------------------------------------
Token convention  : ``[B, N, D]`` with explicit ``H, W`` threading through
                    each stage (from model.py).  The agent version kept the
                    spatial ``[B, C, H, W]`` layout and permuted at every
                    block boundary — correct but adds overhead.

out_dim attribute : Backbone exposes ``self.out_dim`` so the model assembly
                    layer (``medical_vmamba.py``) can build heads without
                    hardcoding channel counts.

Intermediates     : ``forward`` returns ``(features, intermediates)`` as in
                    the agent version.  This enables future XAI and
                    segmentation / detection heads that consume multi-scale
                    features.  Single-task callers can simply ignore the
                    second return value.

PatchMerging      : Odd-H/W padding guard from model.py is retained.  The
                    agent version silently failed on non-power-of-two inputs.

Weight init       : ``trunc_normal_(std=0.02)`` from model.py, matching
                    official Swin / VMamba checkpoints.

Stochastic depth  : Linear ``dpr`` schedule from the agent version, wired
                    into each ``VSSBlock``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from medical_mamba.models.blocks import VSSBlock


# ─────────────────────────────────────────────────────────────────────────────
# PatchEmbed
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Convert ``(B, C, H, W)`` images into ``(B, N, D)`` patch tokens.

    Default: ``patch_size=4`` on 224×224 → N = 3 136 tokens, D = 96.

    Memory / speed trade-off::

        patch_size=4  → 3 136 tokens, high detail, ~4× more compute than p=8
        patch_size=8  → 784 tokens,   sufficient for tissue / cell textures
        patch_size=16 → 196 tokens,   ViT-style, fastest

    Parameters
    ----------
    img_size : int
        Assumed square input size (used only to pre-compute ``num_patches``).
    patch_size : int
        Convolution kernel stride.
    in_chans : int
        Input channels (3 for RGB, 1 for grayscale — convert to RGB first).
    embed_dim : int
        Output channel dimension D.
    """

    def __init__(
        self,
        img_size:   int = 224,
        patch_size: int = 4,
        in_chans:   int = 3,
        embed_dim:  int = 96,
    ) -> None:
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size  = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Embed patches.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, C, H, W)``

        Returns
        -------
        tokens : torch.Tensor
            ``(B, N, D)``
        H, W : int
            Spatial dimensions of the token grid.
        """
        x = self.proj(x)                              # (B, D, H', W')
        B, D, H, W = x.shape
        x = rearrange(x, "b d h w -> b (h w) d")     # (B, N, D)
        x = self.norm(x)
        return x, H, W


# ─────────────────────────────────────────────────────────────────────────────
# PatchMerging
# ─────────────────────────────────────────────────────────────────────────────

class PatchMerging(nn.Module):
    """Halve spatial resolution and double channel width.

    Stage i: ``(B, H*W, D)`` → ``(B, H/2 * W/2, 2D)``.

    Parameters
    ----------
    d_model : int
        Input channel dimension D.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm      = nn.LayerNorm(4 * d_model)
        self.reduction = nn.Linear(4 * d_model, 2 * d_model, bias=False)

    def forward(
        self, x: torch.Tensor, H: int, W: int
    ) -> Tuple[torch.Tensor, int, int]:
        """Downsample.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, H*W, D)``
        H, W : int
            Current spatial dimensions.

        Returns
        -------
        x : torch.Tensor
            ``(B, H/2 * W/2, 2D)``
        H2, W2 : int
            New spatial dimensions.
        """
        x = rearrange(x, "b (h w) d -> b h w d", h=H, w=W)

        # Pad to even dimensions before strided slicing
        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]   # top-left
        x1 = x[:, 1::2, 0::2, :]   # bottom-left
        x2 = x[:, 0::2, 1::2, :]   # top-right
        x3 = x[:, 1::2, 1::2, :]   # bottom-right

        x = torch.cat([x0, x1, x2, x3], dim=-1)   # (B, H/2, W/2, 4D)
        H2, W2 = x.shape[1], x.shape[2]

        x = rearrange(x, "b h w d -> b (h w) d")
        x = self.norm(x)
        x = self.reduction(x)                       # (B, H/2*W/2, 2D)
        return x, H2, W2


# ─────────────────────────────────────────────────────────────────────────────
# VMambaBackbone
# ─────────────────────────────────────────────────────────────────────────────

class VMambaBackbone(nn.Module):
    """Hierarchical VMamba feature extractor.

    Stage layout for the default Tiny config (``patch_size=4``, ``img=224``)::

        Stage 0 : 56×56, D=96    (2 VSSBlocks)
        Stage 1 : 28×28, D=192   (2 VSSBlocks + PatchMerging)
        Stage 2 : 14×14, D=384   (6 VSSBlocks + PatchMerging)
        Stage 3 :  7×7,  D=768   (2 VSSBlocks + PatchMerging)
        GAP     : [B, 768]

    Parameters
    ----------
    img_size : int
        Input image size (square).
    patch_size : int
        PatchEmbed stride.
    in_chans : int
        Input channels.
    embed_dim : int
        Stage-0 channel width (doubles after each PatchMerging).
    depths : List[int]
        Number of VSSBlocks per stage (must have length 4).
    d_state : int
        SSM state dimension.
    d_conv : int
        Mamba inner depthwise conv kernel.
    expand : int
        Mamba inner-dim expansion.
    mlp_ratio : float
        MLP hidden-dim multiplier inside VSSBlock.
    drop_path_rate : float
        Maximum stochastic depth rate (linearly scheduled across all blocks).
    drop_rate : float
        Positional-embedding dropout after PatchEmbed.
    """

    def __init__(
        self,
        img_size:       int         = 224,
        patch_size:     int         = 4,
        in_chans:       int         = 3,
        embed_dim:      int         = 96,
        depths:         List[int]   = (2, 2, 6, 2),
        d_state:        int         = 16,
        d_conv:         int         = 4,
        expand:         int         = 2,
        mlp_ratio:      float       = 4.0,
        drop_path_rate: float       = 0.2,
        drop_rate:      float       = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim  = embed_dim
        self.num_stages = len(depths)

        # ── Patch embedding ──────────────────────────────────────────────
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_drop    = nn.Dropout(p=drop_rate)

        # ── Stochastic depth schedule (linear) ──────────────────────────
        total_blocks = sum(depths)
        dpr: List[float] = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, total_blocks)
        ]

        # ── Stages + downsamplers ────────────────────────────────────────
        self.stages:       nn.ModuleList = nn.ModuleList()
        self.downsamplers: nn.ModuleList = nn.ModuleList()
        cur_dim   = embed_dim
        block_idx = 0

        for i, depth in enumerate(depths):
            stage = nn.ModuleList([
                VSSBlock(
                    d_model=cur_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[block_idx + j],
                )
                for j in range(depth)
            ])
            self.stages.append(stage)
            block_idx += depth

            if i < self.num_stages - 1:
                self.downsamplers.append(PatchMerging(cur_dim))
                cur_dim *= 2
            else:
                self.downsamplers.append(None)  # type: ignore[arg-type]

        self.out_dim = cur_dim                   # 768 for Tiny; 1536 for Base
        self.norm    = nn.LayerNorm(cur_dim)

        self._init_weights()

    # ------------------------------------------------------------------ #
    #  Weight initialisation                                               #
    # ------------------------------------------------------------------ #

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Extract hierarchical features.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, C, H, W)``

        Returns
        -------
        features : torch.Tensor
            Global-average-pooled feature vector ``(B, out_dim)``.
        intermediates : List[torch.Tensor]
            Per-stage token tensors ``(B, N_i, D_i)`` before downsampling.
            Useful for multi-scale XAI or dense-prediction heads.
        """
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        intermediates: List[torch.Tensor] = []

        for i in range(self.num_stages):
            for block in self.stages[i]:
                x = block(x, H, W)

            intermediates.append(x)               # save before downsampling

            ds = self.downsamplers[i]
            if ds is not None:
                x, H, W = ds(x, H, W)

        x = self.norm(x)                          # (B, N_final, out_dim)
        features = x.mean(dim=1)                  # Global Average Pool → (B, D)
        return features, intermediates

    # ------------------------------------------------------------------ #
    #  XAI helper                                                          #
    # ------------------------------------------------------------------ #

    def get_last_vss_blocks(self) -> List[VSSBlock]:
        """Return the ``VSSBlock`` list from the final stage (for hook registration)."""
        return list(self.stages[-1])
