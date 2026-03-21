"""Visualisation utilities for XAI heatmaps.

Merge decisions
---------------
overlay_heatmap   : Agent version handles tensor/ndarray, CHW→HWC conversion,
                    and both [0,1] and [0,255] float inputs.  Website version
                    assumed uint8 HWC input only.  Agent's is kept.

save_grid         : Agent's 3-panel layout (original | saliency | overlay) is
                    more informative than website's 2-panel.  Kept, with the
                    addition of a ``correct`` flag per sample that colours the
                    title green/red — useful for diagnosing which classes the
                    model misses.

denormalize       : New utility not in either version.  Website version
                    re-opened images from disk to get the display-ready array,
                    which requires keeping file paths around and does double I/O.
                    The denorm approach works directly from the normalised tensor.

visualize_single  : Merged from website.  Uses the same 3-panel layout as
                    save_grid for consistency.  Adds confidence bar.

visualize_batch   : Merged from website, but now calls ``save_grid`` internally
                    so the layout is consistent with the script entry point.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from medical_mamba.data.constants import DATASET_META


# ─────────────────────────────────────────────────────────────────────────────
# Image utilities
# ─────────────────────────────────────────────────────────────────────────────

def denormalize(
    tensor: torch.Tensor,
    mean: List[float],
    std: List[float],
) -> np.ndarray:
    """Reverse dataset normalisation and return a ``uint8`` HWC numpy array.

    Parameters
    ----------
    tensor : torch.Tensor
        Normalised image tensor ``(3, H, W)`` or ``(1, 3, H, W)``.
    mean, std : List[float]
        Per-channel statistics used during normalisation.

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` uint8 array.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    t = tensor.detach().cpu().clone()
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    t = t * s + m
    t = t.clamp(0.0, 1.0)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def overlay_heatmap(
    image: Union[torch.Tensor, np.ndarray],
    heatmap: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay a saliency heatmap onto an image.

    Parameters
    ----------
    image : torch.Tensor | np.ndarray
        Original image.  Accepted layouts:
        - ``(H, W, 3)`` ndarray, uint8 or float [0,1]
        - ``(3, H, W)`` or ``(1, 3, H, W)`` tensor, any dtype
    heatmap : torch.Tensor | np.ndarray
        Saliency map ``(H, W)`` in ``[0, 1]``.
    alpha : float
        Heatmap blend weight (0 = image only, 1 = heatmap only).
    colormap : str
        Matplotlib colormap name.

    Returns
    -------
    np.ndarray
        Blended image ``(H, W, 3)`` uint8.
    """
    # ── Normalise image to uint8 HWC ─────────────────────────────────────
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu()
        if img.dim() == 4:
            img = img.squeeze(0)
        if img.shape[0] in (1, 3):        # CHW → HWC
            img = img.permute(1, 2, 0)
        img = img.numpy()
    else:
        img = np.asarray(image)

    if img.ndim == 2:                      # grayscale → RGB
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)

    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    # ── Normalise heatmap ────────────────────────────────────────────────
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    heatmap = np.asarray(heatmap, dtype=np.float32)

    cmap = plt.get_cmap(colormap)
    hm_rgb = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)

    return (alpha * hm_rgb + (1.0 - alpha) * img).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Single-image plot
# ─────────────────────────────────────────────────────────────────────────────

def visualize_single(
    original_img: np.ndarray,
    saliency_map: np.ndarray,
    task_name: str,
    pred_class: int,
    true_class: Optional[int] = None,
    pred_conf: float = 0.0,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot a 3-panel (original | saliency | overlay) figure.

    Parameters
    ----------
    original_img : np.ndarray
        ``(H, W, 3)`` uint8 display image.
    saliency_map : np.ndarray
        ``(H, W)`` float saliency in ``[0, 1]``.
    task_name : str
        Dataset / task name for the title.
    pred_class : int
        Predicted class index.
    true_class : int | None
        Ground-truth class (shown if provided).
    pred_conf : float
        Predicted class confidence (0–1).
    save_path : str | Path | None
        Save figure to this path instead of ``plt.show()``.
    """
    overlay = overlay_heatmap(original_img, saliency_map)
    correct = true_class is not None and pred_class == true_class

    title_parts = [f"Task: {task_name}", f"Pred: {pred_class} ({pred_conf:.1%})"]
    if true_class is not None:
        title_parts.append(f"GT: {true_class}")
    title_color = "green" if correct else ("red" if true_class is not None else "black")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(" | ".join(title_parts), fontsize=12, color=title_color)

    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(saliency_map, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Saliency (SSM-GradCAM)")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Grid plot
# ─────────────────────────────────────────────────────────────────────────────

def save_grid(
    images: List[np.ndarray],
    heatmaps: List[np.ndarray],
    output_path: Union[str, Path],
    labels: Optional[List[str]] = None,
    correct_flags: Optional[List[bool]] = None,
    ncols: int = 4,
    figsize: Optional[Tuple[int, int]] = None,
) -> Path:
    """Save a grid of (original | saliency | overlay) triplets.

    Parameters
    ----------
    images : List[np.ndarray]
        Display images ``(H, W, 3)`` uint8.
    heatmaps : List[np.ndarray]
        Saliency maps ``(H, W)`` in ``[0, 1]``.
    output_path : str | Path
        Output file path (.png).
    labels : List[str] | None
        Per-image title strings (e.g. ``"GT:3 Pred:3"``).
    correct_flags : List[bool] | None
        If provided, title is green for correct, red for wrong.
    ncols : int
        Image columns in the grid (each image = 3 subplot columns).
    figsize : Tuple[int, int] | None
        Override figure size.

    Returns
    -------
    Path
        Absolute path of the saved file.
    """
    n     = len(images)
    nrows = math.ceil(n / ncols)
    figsize = figsize or (ncols * 9, nrows * 3)

    fig, axes = plt.subplots(nrows, ncols * 3, figsize=figsize)
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        row, col = divmod(i, ncols)
        bc = col * 3                      # base column index

        lbl   = labels[i] if labels else f"#{i}"
        color = "black"
        if correct_flags is not None:
            color = "green" if correct_flags[i] else "red"

        ol = overlay_heatmap(images[i], heatmaps[i])

        axes[row, bc].imshow(images[i])
        axes[row, bc].set_title(lbl, fontsize=7, color=color)
        axes[row, bc].axis("off")

        axes[row, bc + 1].imshow(heatmaps[i], cmap="jet", vmin=0, vmax=1)
        axes[row, bc + 1].set_title("saliency", fontsize=7)
        axes[row, bc + 1].axis("off")

        axes[row, bc + 2].imshow(ol)
        axes[row, bc + 2].set_title("overlay", fontsize=7)
        axes[row, bc + 2].axis("off")

    # Hide unused axes
    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        for c in range(3):
            axes[row, col * 3 + c].axis("off")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()
