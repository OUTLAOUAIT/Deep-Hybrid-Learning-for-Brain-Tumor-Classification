"""Shared deterministic evaluation and train-only augmentation transforms."""
from torchvision import transforms


def build_transforms(image_size=224, training=False, normalization="imagenet"):
    ops = [transforms.Resize((image_size, image_size))]
    if training:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    ops.append(transforms.ToTensor())
    if normalization == "vit_keras":
        # The paper's vit-keras preprocessing maps RGB values to [-1, 1].
        ops.append(transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                        std=(0.5, 0.5, 0.5)))
    elif normalization == "imagenet":
        ops.append(transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                        std=(0.229, 0.224, 0.225)))
    else:
        raise ValueError("Unknown normalization: " + normalization)
    return transforms.Compose(ops)
