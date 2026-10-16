"""Evaluate equal-weight soft-voting ensembles of trained checkpoints."""
import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import BrainTumorDataset, load_manifest
from src.data.transforms import build_transforms
from src.evaluation.metrics import classification_metrics
from src.evaluation.visualization import plot_confusion_matrix
from src.models import build_model


def _read_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def ensemble_probabilities(configs, checkpoints, loader, classes, device):
    """Average per-model softmax probabilities, as in Tummala et al. (2022)."""
    total_probabilities = None
    actual = []
    inference_seconds = 0.0
    parameter_count = 0

    for config, checkpoint in zip(configs, checkpoints):
        collect_actual = total_probabilities is None
        model = build_model(config["model"], len(classes), False,
                            config.get("dropout", 0.3),
                            config.get("freeze_backbone", False),
                            config["image_size"])
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])
        del state
        model.to(device).eval()
        parameter_count += sum(parameter.numel() for parameter in model.parameters())
        model_probabilities = []

        with torch.no_grad():
            for images, labels in loader:
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                probabilities = model(images.to(device)).softmax(dim=1)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize(device)
                inference_seconds += time.perf_counter() - started
                model_probabilities.append(probabilities.cpu())
                if collect_actual:
                    actual.extend(labels.tolist())

        current = torch.cat(model_probabilities, dim=0)
        total_probabilities = current if total_probabilities is None else total_probabilities + current
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    mean_probabilities = total_probabilities / len(configs)
    result = classification_metrics(actual, mean_probabilities.argmax(dim=1).tolist(),
                                    mean_probabilities.tolist())
    result["selection_score"] = (result["accuracy"] + result["precision_macro"] +
                                 result["recall_macro"]) / 3.0
    result["classes"] = classes
    result["per_class"] = {name: result["report"].get(str(index), {})
                           for index, name in enumerate(classes)}
    result["ensemble_method"] = "equal_weight_soft_voting"
    result["members"] = [config["model"] for config in configs]
    result["member_count"] = len(configs)
    result["image_size"] = configs[0]["image_size"]
    result["inference_seconds"] = inference_seconds
    result["latency_ms_per_image"] = inference_seconds * 1000.0 / max(1, len(actual))
    result["total_parameters"] = parameter_count
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an equal-weight soft-voting ensemble on the test split.")
    parser.add_argument("--configs", nargs="+", required=True,
                        help="Member YAML configs, one per --checkpoints entry")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Trained checkpoint paths in the same order as --configs")
    parser.add_argument("--output", default="outputs/reports/ensemble.json")
    parser.add_argument("--image-size", type=int, choices=(224, 384), default=None,
                        help="Override the shared input resolution (224 or 384)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override the ensemble inference batch size (lower for Colab GPU RAM)")
    args = parser.parse_args()

    if len(args.configs) != len(args.checkpoints):
        parser.error("Provide exactly one checkpoint for each config")
    configs = [_read_config(path) for path in args.configs]
    if args.image_size is not None:
        for config in configs:
            config["image_size"] = args.image_size
    image_sizes = {config["image_size"] for config in configs}
    data_roots = {str(Path(config["data_root"]).resolve()) for config in configs}
    if len(image_sizes) != 1:
        parser.error("All ensemble members must use the same image_size")
    if len(data_roots) != 1:
        parser.error("All ensemble members must use the same data_root")
    normalizations = {config.get("normalization", "imagenet") for config in configs}
    if len(normalizations) != 1:
        parser.error("All ensemble members must use the same input normalization")

    root = Path(configs[0]["data_root"])
    val_rows = load_manifest(root / "splits/val.csv")
    test_rows = load_manifest(root / "splits/test.csv")
    classes = sorted({row["label"] for row in test_rows})
    mapping = {name: index for index, name in enumerate(classes)}
    transform = build_transforms(configs[0]["image_size"],
                                 normalization=configs[0].get("normalization", "imagenet"))
    val_dataset = BrainTumorDataset(val_rows, root, transform, mapping)
    test_dataset = BrainTumorDataset(test_rows, root, transform, mapping)
    batch_size = args.batch_size or min(config.get("batch_size", 32) for config in configs)
    workers = min(config.get("workers", 0) for config in configs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation = ensemble_probabilities(configs, args.checkpoints, val_loader, classes, device)
    result = ensemble_probabilities(configs, args.checkpoints, test_loader, classes, device)
    result["validation"] = {key: validation[key] for key in (
        "accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
        "f1_macro", "selection_score", "confusion_matrix", "per_class")}
    result["checkpoint_paths"] = [str(Path(path)) for path in args.checkpoints]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_confusion_matrix(result["confusion_matrix"], classes,
                          Path("outputs/figures") / (output.stem + "_confusion.png"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
