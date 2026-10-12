import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import BrainTumorDataset, load_manifest
from src.data.transforms import build_transforms
from src.evaluation.evaluator import evaluate
from src.evaluation.visualization import plot_confusion_matrix
from src.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/reports/evaluation.json")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(config["data_root"])
    rows = load_manifest(root / "splits/test.csv")
    classes = sorted({row["label"] for row in rows})
    mapping = {name: i for i, name in enumerate(classes)}
    ds = BrainTumorDataset(rows, root, build_transforms(config["image_size"]), mapping)
    loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=False)
    # Checkpoint contains the complete backbone state, so evaluation needs no weight download.
    model = build_model(config["model"], len(classes), False,
                        config.get("dropout", 0.3), config.get("freeze_backbone", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = evaluate(model, loader, args.checkpoint, device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_confusion_matrix(result["confusion_matrix"], classes,
                          Path("outputs/figures") / (Path(args.output).stem + "_confusion.png"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
