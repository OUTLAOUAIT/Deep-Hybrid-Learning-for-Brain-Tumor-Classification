import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_efficientnet(num_classes, pretrained=True, dropout=0.3, freeze=False):
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    if freeze:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))
    return model
