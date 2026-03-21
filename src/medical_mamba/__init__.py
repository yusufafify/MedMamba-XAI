"""Medical Mamba — Interpretable VMamba Models for Medical Image Classification.

This package provides:
- VMamba backbone adapted for 2D medical imaging
- Multi-task learning with per-dataset classification heads
- SSM-GradCAM explainability utilities
- Training, evaluation, and checkpoint management helpers
"""

from medical_mamba.models.medical_vmamba import MedicalVMamba
from medical_mamba.training.trainer import Trainer

__version__ = "0.1.0"
__all__ = ["MedicalVMamba", "Trainer", "__version__"]
