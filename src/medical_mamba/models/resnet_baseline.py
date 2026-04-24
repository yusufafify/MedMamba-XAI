"""ResNet50 baseline — fine-tuned on MedMNIST datasets.

Drop-in replacement for MedicalVMamba in the training pipeline.
Same input/output interface: forward_single(x, task_name) → (logits, [])
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Optional, Tuple


class ResNetBaseline(nn.Module):
    def __init__(
        self,
        task_configs: List[Tuple[str, int]],
        pretrained: bool = True,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Load pretrained ResNet50, remove final FC
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # remove FC
        self.feat_dim = 2048  # ResNet50 output dim

        self.task_names = [name for name, _ in task_configs]
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(self.feat_dim),
                nn.Dropout(p=head_dropout),
                nn.Linear(self.feat_dim, n_cls),
            )
            for name, n_cls in task_configs
        })

    def forward_single(
        self, x: torch.Tensor, task_name: str
    ) -> Tuple[torch.Tensor, List]:
        features = self.backbone(x).flatten(1)   # [B, 2048]
        logits = self.heads[task_name](features)
        return logits, []                         # empty intermediates — no XAI needed

    def forward(self, x, task_name):
        return self.forward_single(x, task_name)

    def forward_multi(self, x, task_ids):
        features = self.backbone(x).flatten(1)
        task_logits = {}
        for tid, name in enumerate(self.task_names):
            mask = task_ids == tid
            if mask.any():
                task_logits[name] = self.heads[name](features[mask])
        return task_logits, features, []