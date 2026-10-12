import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


def build_vit(num_classes, pretrained=True, dropout=0.1, freeze=False):
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)
    if freeze:
        for name, parameter in model.named_parameters():
            if not name.startswith("heads"):
                parameter.requires_grad = False
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model
