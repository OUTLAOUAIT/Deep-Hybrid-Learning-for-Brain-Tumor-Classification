"""ImageNet-pretrained CNN backbones used in the comparison study."""
import torch.nn as nn
from torchvision.models import (EfficientNet_B3_Weights, MobileNet_V2_Weights,
                                VGG16_Weights, efficientnet_b3, mobilenet_v2, vgg16)


def _freeze_backbone(module, head_prefix):
    for name, parameter in module.named_parameters():
        if not name.startswith(head_prefix):
            parameter.requires_grad = False


def build_vgg16(num_classes, pretrained=True, dropout=0.3, freeze=False):
    model = vgg16(weights=VGG16_Weights.DEFAULT if pretrained else None)
    if freeze:
        _freeze_backbone(model, "classifier.6")
    for layer in model.classifier:
        if isinstance(layer, nn.Dropout):
            layer.p = dropout
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model


def build_mobilenet_v2(num_classes, pretrained=True, dropout=0.2, freeze=False):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)
    if freeze:
        _freeze_backbone(model, "classifier.1")
    model.classifier[0].p = dropout
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet_b3(num_classes, pretrained=True, dropout=0.3, freeze=False):
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT if pretrained else None)
    if freeze:
        _freeze_backbone(model, "classifier.1")
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(dropout),
                                     nn.Linear(in_features, num_classes))
    return model
