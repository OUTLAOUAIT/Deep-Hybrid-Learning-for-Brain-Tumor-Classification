import argparse
import csv
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import BrainTumorDataset, load_manifest
from src.data.transforms import build_transforms
from src.models import build_model
from src.training import fit
from src.utils.reproducibility import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(config.get("seed", 42))
    root = Path(config["data_root"])
    train_rows = load_manifest(root / "splits/train.csv")
    val_rows = load_manifest(root / "splits/val.csv")
    classes = sorted({row["label"] for row in train_rows})
    mapping = {name: i for i, name in enumerate(classes)}
    ds = lambda rows, training: BrainTumorDataset(rows, root,
                      build_transforms(config["image_size"], training), mapping)
    loader_args = {"batch_size": config["batch_size"], "num_workers": config.get("workers", 0)}
    train_loader = DataLoader(ds(train_rows, True), shuffle=True, **loader_args)
    val_loader = DataLoader(ds(val_rows, False), shuffle=False, **loader_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"], len(classes), config.get("pretrained", True),
                        config.get("dropout", 0.3), config.get("freeze_backbone", False))
    fit(model, train_loader, val_loader, config,
        config.get("checkpoint", "outputs/checkpoints/model.pt"), device)


if __name__ == "__main__":
    main()
