from .efficientnet import build_efficientnet
from .feature_fusion import EfficientNetVGGFusion
from .vit import build_vit


def build_model(name, num_classes, pretrained=True, dropout=0.3, freeze=False):
    if name == "efficientnet_b0":
        return build_efficientnet(num_classes, pretrained, dropout, freeze)
    if name == "efficientnet_vgg16_fusion":
        return EfficientNetVGGFusion(num_classes, pretrained, dropout, freeze)
    if name == "vit_b_16":
        return build_vit(num_classes, pretrained, dropout, freeze)
    raise ValueError("Unknown model: " + name)
