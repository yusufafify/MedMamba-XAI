import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import timm

class ViTBaseline(nn.Module):
    def __init__(
        self,
        task_configs: List[Tuple[str, int]],
        model_size: str = "tiny", # "tiny", "base_16", "base_32"
        pretrained: bool = True,
        head_dropout: float = 0.1,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        
        # Select model name based on size
        if model_size == "tiny":
            model_name = "vit_tiny_patch16_224"
            self.feat_dim = 192
        elif model_size == "base_16":
            model_name = "vit_base_patch16_224"
            self.feat_dim = 768
        elif model_size == "base_32":
            model_name = "vit_base_patch32_224"
            self.feat_dim = 768
        else:
            raise ValueError(f"Unsupported model_size: {model_size}")
            
        # Initialize timm model without head
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0 # removes the classification head
        )

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
        # timm model with num_classes=0 returns the pooled features
        features = self.backbone(x)
        logits = self.heads[task_name](features)
        return logits, [] # empty intermediates

    def forward(self, x, task_name):
        return self.forward_single(x, task_name)

    def forward_multi(self, x, task_ids):
        features = self.backbone(x)
        task_logits = {}
        for tid, name in enumerate(self.task_names):
            mask = task_ids == tid
            if mask.any():
                task_logits[name] = self.heads[name](features[mask])
        return task_logits, []
