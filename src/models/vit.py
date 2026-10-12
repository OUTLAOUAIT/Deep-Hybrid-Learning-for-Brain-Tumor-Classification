import torch.nn as nn
import timm


VIT_BACKBONES = {
    # Original ViT-B/16, B/32, L/16 and L/32 encoder sizes/patch layouts.
    # The released timm weights use ImageNet-21k pretraining and ImageNet-1k
    # fine-tuning where available; L/32 is available as the original 21k model.
    "vit_b_16": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "vit_b_32": "vit_base_patch32_224.augreg_in21k_ft_in1k",
    "vit_l_16": "vit_large_patch16_224.augreg_in21k_ft_in1k",
    "vit_l_32": "vit_large_patch32_224.orig_in21k",
}


def build_vit(num_classes, pretrained=True, dropout=0.1, freeze=False,
              variant="vit_b_16"):
    if variant not in VIT_BACKBONES:
        raise ValueError("Unknown ViT variant: " + variant)
    model = timm.create_model(VIT_BACKBONES[variant], pretrained=pretrained,
                              num_classes=num_classes, dynamic_img_size=True)
    model.head = nn.Sequential(nn.Dropout(dropout), model.head)
    if freeze:
        for name, parameter in model.named_parameters():
            if not name.startswith("head"):
                parameter.requires_grad = False
    return model
