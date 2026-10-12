"""Convert Figshare cjdata MAT files to PNG plus a reproducible split manifest."""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from pymatreader import read_mat
import yaml

from src.data.splitting import create_splits

LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    source = Path(args.source or config.get("source", "data/raw"))
    output = Path(args.output or config.get("output", "data/processed"))
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for mat_path in sorted(source.rglob("*.mat")):
        item = read_mat(str(mat_path))["cjdata"]
        label_id = int(np.asarray(item["label"]).squeeze())
        if label_id not in LABELS:
            raise ValueError("Unexpected class ID {} in {}".format(label_id, mat_path))
        array = np.asarray(item["image"], dtype=np.float32)
        low, high = float(np.nanmin(array)), float(np.nanmax(array))
        if not np.isfinite(array).all() or high <= low:
            raise ValueError("Invalid or constant image: {}".format(mat_path))
        array = np.clip((array - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
        rel = Path(LABELS[label_id]) / (mat_path.stem + ".png")
        dest = output / "images" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(dest)
        pid = item.get("PID")
        group = str(np.asarray(pid).squeeze()) if pid is not None else ""
        rows.append({"path": str(Path("images") / rel), "label": LABELS[label_id],
                     "group": group, "source": str(mat_path.relative_to(source))})
    if not rows:
        raise FileNotFoundError("No .mat files found under {}".format(source))
    with open(output / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    create_splits(rows, output, output / "splits", seed=seed)
    print("Prepared {} samples; split manifests: {}".format(len(rows), output / "splits"))


if __name__ == "__main__":
    main()
