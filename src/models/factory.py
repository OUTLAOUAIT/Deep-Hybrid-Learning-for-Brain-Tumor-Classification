from .efficientnet import build_efficientnet
from .feature_fusion import EfficientNetVGGFusion
from .vit import build_vit


def build_model(name, num_classes, pretrained=True, dropout=0.3, freeze=False,
                image_size=224):
    if name == "efficientnet_b0":
        return build_efficientnet(num_classes, pretrained, dropout, freeze)
    if name == "efficientnet_vgg16_fusion":
        return EfficientNetVGGFusion(num_classes, pretrained, dropout, freeze)
    if name in ("vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"):
        return build_vit(num_classes, pretrained, dropout, freeze, name)
    raise ValueError("Unknown model: " + name)
