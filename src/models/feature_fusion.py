import torch
import torch.nn as nn
from torchvision.models import (efficientnet_b0, vgg16, EfficientNet_B0_Weights,
                                VGG16_Weights)


class EfficientNetVGGFusion(nn.Module):
    """Concatenate globally pooled 1280-D EfficientNet and 512-D VGG features."""
    def __init__(self, num_classes, pretrained=True, dropout=0.4, freeze=False):
        super().__init__()
        self.efficientnet = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        self.efficientnet.classifier = nn.Identity()
        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT if pretrained else None)
        self.vgg.classifier = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        if freeze:
            for module in (self.efficientnet.features, self.vgg.features):
                for parameter in module.parameters():
                    parameter.requires_grad = False
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280 + 512, 512),
                                  nn.ReLU(inplace=True), nn.Dropout(dropout),
                                  nn.Linear(512, num_classes))

    def forward(self, x):
        a = self.pool(self.efficientnet.features(x)).flatten(1)
        b = self.pool(self.vgg.features(x)).flatten(1)
        return self.head(torch.cat((a, b), dim=1))
