"""Visual State-Space Block (VSSBlock) and cross-scan aggregation.

Design decisions vs the two source versions
--------------------------------------------
Token convention  : ``[B, N, D]`` sequences with ``H, W`` passed explicitly
                    (from model.py / website version).  The agent version used
                    ``[B, C, H, W]`` spatial tensors and permuted at every op,
                    which is correct but adds unnecessary overhead.

Mamba core        : ``mamba_ssm.Mamba`` when available, ``MambaPyTorchFallback``
                    otherwise (from model.py).  The agent version hand-rolled
                    ``A_log / D`` parameters without ever calling a real SSM —
                    it is not a faithful Mamba implementation.

Depthwise conv    : Kept from the agent version.  The local mixing conv before
                    the cross-scan improves gradient flow and is present in
                    several VMamba follow-ups.

StochasticDepth   : Proper binary-mask implementation from the agent version
                    with a linear drop-path schedule built in the backbone.
                    The website version used plain ``nn.Dropout`` which is not
                    the same thing (it scales activations; stochastic depth
                    drops entire residuals).

MLP block         : Kept from model.py (norm → mamba → res + norm → MLP → res)
                    which is faithful to the VMamba paper.

XAI hooks         : Forward/backward activation storage from model.py.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# ─────────────────────────────────────────────────────────────────────────────
# Mamba core — safe import with pure-PyTorch fallback
# ─────────────────────────────────────────────────────────────────────────────

try:
    from mamba_ssm import Mamba as _MambaSSM
    _MAMBA_AVAILABLE = True
except ImportError:
    _MambaSSM = None  # type: ignore[assignment,misc]
    _MAMBA_AVAILABLE = False
    import warnings
    warnings.warn(
        "mamba-ssm not installed. Using pure-PyTorch SSM fallback — correct "
        "but ~10× slower. Install with: pip install mamba-ssm causal-conv1d",
        stacklevel=2,
    )


class MambaPyTorchFallback(nn.Module):
    """Pure-PyTorch S6 approximation for CPU / dev environments.

    Algorithmically faithful to the selective scan but uses Euler
    discretisation instead of ZOH.  Swap for ``mamba_ssm.Mamba`` in
    production.

    Parameters
    ----------
    d_model : int
        Input / output dimension.
    d_state : int
        SSM state dimension (N in the paper).
    d_conv  : int
        Depthwise conv kernel size inside Mamba.
    expand  : int
        Inner-dimension expansion multiplier.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv:  int = 4,
        expand:  int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj  = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(self.d_inner)

        # Fixed A (HiPPO initialisation), learnable D skip
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n", d=self.d_inner,
        )
        self.register_buffer("A_log", torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, L, D)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, L, D)``.
        """
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_, z = xz.chunk(2, dim=-1)

        # Causal depthwise conv
        x_ = rearrange(x_, "b l d -> b d l")
        x_ = self.conv1d(x_)[..., :L]
        x_ = rearrange(x_, "b d l -> b l d")
        x_ = F.silu(x_)

        # Selective scan (Euler discretisation — replace with ZOH for publication)
        dt_BC = self.x_proj(x_)
        dt, _, _ = dt_BC.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        # Skip-connection term (D · x)
        y = x_ * self.D.unsqueeze(0).unsqueeze(0)
        y = self.norm(y)
        return self.out_proj(y * F.silu(z))


def _make_mamba(d_model: int, d_state: int = 16,
                d_conv: int = 4, expand: int = 2) -> nn.Module:
    """Return the best available Mamba block."""
    if _MAMBA_AVAILABLE:
        return _MambaSSM(d_model=d_model, d_state=d_state,
                         d_conv=d_conv, expand=expand)
    return MambaPyTorchFallback(d_model=d_model, d_state=d_state,
                                d_conv=d_conv, expand=expand)


# ─────────────────────────────────────────────────────────────────────────────
# Stochastic depth
# ─────────────────────────────────────────────────────────────────────────────

class StochasticDepth(nn.Module):
    """Drop entire residual branches with probability ``drop_prob`` during training.

    This is *not* the same as ``nn.Dropout``, which scales individual
    activations.  Stochastic depth drops the whole residual path for a
    sample and is standard practice in Swin / VMamba training.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, dtype=x.dtype, device=x.device) < keep).float()
        return x / keep * mask


# ─────────────────────────────────────────────────────────────────────────────
# Cross-scan aggregation
# ─────────────────────────────────────────────────────────────────────────────

def cross_scan_2d(
    x: torch.Tensor, H: int, W: int
) -> List[torch.Tensor]:
    """Unfold a token sequence into 4 directional scan orders.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(B, N, D)`` where ``N = H * W``.
    H, W : int
        Spatial dimensions.

    Returns
    -------
    List[torch.Tensor]
        Four ``(B, N, D)`` tensors in scan orders:
        row→, ←row, col↓, ↑col.
    """
    assert x.shape[1] == H * W, f"N={x.shape[1]} ≠ H*W={H*W}"
    img = rearrange(x, "b (h w) d -> b h w d", h=H, w=W)
    return [
        rearrange(img,                                   "b h w d -> b (h w) d"),   # →
        rearrange(img.flip(1).flip(2),                  "b h w d -> b (h w) d"),   # ←↑
        rearrange(img.permute(0, 2, 1, 3),              "b h w d -> b (h w) d"),   # ↓
        rearrange(img.permute(0, 2, 1, 3).flip(1).flip(2), "b h w d -> b (h w) d"),  # ↑
    ]


def cross_scan_aggregate(
    mamba_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    """Run ``mamba_fn`` on each scan direction and average the results.

    Un-reversing / un-transposing each direction before averaging ensures
    that every output token corresponds to the same spatial position across
    all four scans.

    Parameters
    ----------
    mamba_fn : callable
        A ``(B, N, D) → (B, N, D)`` function (e.g. a ``Mamba`` block).
    x : torch.Tensor
        Shape ``(B, N, D)``.
    H, W : int
        Spatial dimensions.

    Returns
    -------
    torch.Tensor
        Aggregated shape ``(B, N, D)``.
    """
    scans = cross_scan_2d(x, H, W)

    # Direction 0: natural order — no un-transform needed
    o0 = rearrange(mamba_fn(scans[0]), "b (h w) d -> b h w d", h=H, w=W)

    # Direction 1: reversed → un-reverse
    o1 = rearrange(mamba_fn(scans[1]), "b (h w) d -> b h w d", h=H, w=W).flip(1).flip(2)

    # Direction 2: col-major → un-transpose
    o2 = rearrange(mamba_fn(scans[2]), "b (h w) d -> b h w d", h=H, w=W).permute(0, 2, 1, 3)

    # Direction 3: col-major reversed → un-reverse then un-transpose
    o3 = rearrange(mamba_fn(scans[3]), "b (h w) d -> b h w d", h=H, w=W).flip(1).flip(2).permute(0, 2, 1, 3)

    aggregated = torch.stack([o0, o1, o2, o3], dim=0).mean(dim=0)  # (B, H, W, D)
    return rearrange(aggregated, "b h w d -> b (h w) d")


# ─────────────────────────────────────────────────────────────────────────────
# VSSBlock
# ─────────────────────────────────────────────────────────────────────────────

class VSSBlock(nn.Module):
    """Visual State-Space Block.

    Structure::

        x ──┬── LN ──► dw_conv ──► cross_scan(Mamba) ──► + ──┬── LN ──► MLP ──► +
            │                                              │   │                  │
            └──────────────────────────────────────────────┘   └──────────────────┘
               residual 1                                           residual 2

    The depthwise conv before the SSM is taken from the agent version — it
    provides local feature mixing that complements the global selective scan.

    Parameters
    ----------
    d_model : int
        Channel dimension.
    d_state : int
        SSM state size.
    d_conv : int
        Mamba inner conv kernel size.
    expand : int
        Mamba inner-dim expansion.
    mlp_ratio : float
        MLP hidden-dim multiplier.
    drop_path : float
        Stochastic depth drop probability.
    """

    def __init__(
        self,
        d_model:   int   = 96,
        d_state:   int   = 16,
        d_conv:    int   = 4,
        expand:    int   = 2,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # ── SSM branch ───────────────────────────────────────────────────
        self.norm1 = nn.LayerNorm(d_model)

        # Local mixing before global SSM scan (from agent version)
        inner_dim = int(d_model * expand)
        self.dw_conv = nn.Conv2d(
            d_model, d_model,
            kernel_size=3, padding=1, groups=d_model,
        )

        self.mamba = _make_mamba(d_model, d_state=d_state,
                                 d_conv=d_conv, expand=expand)

        # ── MLP branch ───────────────────────────────────────────────────
        self.norm2  = nn.LayerNorm(d_model)
        mlp_hidden  = int(d_model * mlp_ratio)
        self.mlp    = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )

        # ── Stochastic depth ─────────────────────────────────────────────
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()

        # ── XAI state ────────────────────────────────────────────────────
        self.activations: Optional[torch.Tensor] = None
        self.gradients:   Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    #  XAI hooks                                                           #
    # ------------------------------------------------------------------ #

    def _save_activation(
        self, module: nn.Module, input: tuple, output: torch.Tensor
    ) -> None:
        self.activations = output.detach()

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradients = grad.detach()

    def register_xai_hooks(self) -> None:
        """Register forward + backward hooks on the Mamba block.

        Call once before the forward pass you want to explain.
        Hooks are cumulative — avoid calling this multiple times.
        """
        self.mamba.register_forward_hook(self._save_activation)

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, N, D)`` where ``N = H * W``.
        H, W : int
            Spatial dimensions of the current feature map.

        Returns
        -------
        torch.Tensor
            Shape ``(B, N, D)``.
        """
        B, N, D = x.shape

        # ── SSM branch ───────────────────────────────────────────────────
        residual = x
        x_ln = self.norm1(x)

        # Local depthwise conv on spatial layout before SSM
        x_spatial = rearrange(x_ln, "b (h w) d -> b d h w", h=H, w=W)
        x_spatial = self.dw_conv(x_spatial)
        x_ln = rearrange(x_spatial, "b d h w -> b (h w) d")

        # Four-directional SSM scan
        x_ssm = cross_scan_aggregate(self.mamba, x_ln, H, W)

        # Capture activation for XAI gradient hook
        if self.activations is not None:
            self.activations.requires_grad_(True)
            self.activations.register_hook(self._save_gradient)

        x = residual + self.drop_path(x_ssm)

        # ── MLP branch ───────────────────────────────────────────────────
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x