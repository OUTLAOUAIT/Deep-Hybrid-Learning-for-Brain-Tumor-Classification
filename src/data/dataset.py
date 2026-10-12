"""Manifest-backed datasets for processed image files."""
import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


def load_manifest(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"path", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Manifest must contain path,label columns and at least one row")
    return rows


class BrainTumorDataset(Dataset):
    def __init__(self, rows, root=None, transform=None, class_to_idx=None):
        self.rows = rows
        self.root = Path(root or ".")
        self.transform = transform
        labels = sorted({str(row["label"]) for row in rows})
        self.class_to_idx = class_to_idx or {name: i for i, name in enumerate(labels)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row["path"])
        if not path.is_absolute():
            path = self.root / path
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[str(row["label"])]
