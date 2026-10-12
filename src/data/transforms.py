"""Shared deterministic evaluation and train-only augmentation transforms."""
from torchvision import transforms


def build_transforms(image_size=224, training=False):
    ops = [transforms.Resize((image_size, image_size))]
    if training:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    ops += [transforms.ToTensor(), transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
    return transforms.Compose(ops)
