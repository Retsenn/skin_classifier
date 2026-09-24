import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

class AcanthosisEfficientNetB0(nn.Module):
    def __init__(self, pretrained: bool = True, drop_rate: float = 0.3):
        super(AcanthosisEfficientNetB0, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Get in_features from default classifier head
        in_features = self.backbone.classifier[1].in_features
        
        # Replace default classifier with custom medical classification head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=drop_rate * 0.67),
            nn.Linear(256, 1) # Single output logit for binary classification
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
        
    def freeze_backbone(self):
        """Freezes feature extractor layers for initial head warmup."""
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
                
    def unfreeze_top_layers(self, num_blocks: int = 2):
        """Unfreezes the top MBConv blocks for stage 2 fine-tuning."""
        # EfficientNet features consist of 8 main block stages
        for name, param in self.backbone.features.named_parameters():
            # Unfreeze top feature stages (stages 6 and 7)
            if any(f"features.{stage}." in name for stage in range(8 - num_blocks, 8)):
                param.requires_grad = True
            else:
                param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
            
    def unfreeze_all(self):
        """Unfreezes all layers for full network fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

def build_efficientnet_b0(pretrained: bool = True) -> AcanthosisEfficientNetB0:
    return AcanthosisEfficientNetB0(pretrained=pretrained)
