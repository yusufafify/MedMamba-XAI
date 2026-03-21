"""XAI (Explainability) module for MedicalVMamba.

Public API
----------
SSMGradCAM
    GradCAM-style saliency using VMamba intermediate feature maps.
load_model_from_checkpoint
    Reconstruct a MedicalVMamba from a training checkpoint.
denormalize
    Reverse dataset normalisation for display.
overlay_heatmap
    Blend a saliency heatmap onto an image.
save_grid
    Save a grid of (original | saliency | overlay) triplets.
visualize_single
    Plot a 3-panel figure for one image.
"""

from medical_mamba.xai.gradcam import SSMGradCAM, load_model_from_checkpoint
from medical_mamba.xai.visualize import (
    denormalize,
    overlay_heatmap,
    save_grid,
    visualize_single,
)

__all__ = [
    "SSMGradCAM",
    "load_model_from_checkpoint",
    "denormalize",
    "overlay_heatmap",
    "save_grid",
    "visualize_single",
]
