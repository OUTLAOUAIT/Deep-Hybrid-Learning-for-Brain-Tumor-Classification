"""Summarize already computed evaluation JSON files; never trains or invents scores."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default=None,
                        help="Convenience: read reports here and write comparison.csv here")
    args = parser.parse_args()
    reports_dir = Path(args.reports_dir or args.output_dir or "outputs/reports")
    output = Path(args.output or (Path(args.output_dir) / "comparison.csv"
                                  if args.output_dir else reports_dir / "comparison.csv"))
    reports = []
    for path in sorted(reports_dir.glob("*.json")):
        if path.name == "comparison.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        history_path = Path("outputs/checkpoints") / (path.stem + ".history.json")
        if not history_path.exists():
            raise FileNotFoundError("Missing validation history for {}: {}".format(path.stem, history_path))
        history_file = json.loads(history_path.read_text(encoding="utf-8"))
        history = history_file["history"]
        val_macro_f1 = max(row["val"]["macro_f1"] for row in history)
        reports.append({"model": path.stem, "best_val_macro_f1": val_macro_f1,
                        "training_seconds": history_file.get("training_seconds"),
                        "test_accuracy": data["accuracy"],
                        "test_precision_macro": data["precision_macro"],
                        "test_recall_macro": data["recall_macro"], "test_f1_macro": data["f1_macro"],
                        "test_f1_weighted": data["f1_weighted"],
                        "inference_ms_per_image": data.get("latency_ms_per_image"),
                        "total_parameters": data.get("total_parameters"),
                        "trainable_parameters": data.get("trainable_parameters")})
    if not reports:
        raise FileNotFoundError("No evaluation JSON reports found; evaluate checkpoints first")
    reports.sort(key=lambda row: row["best_val_macro_f1"], reverse=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    print("Wrote {} (ranked by best validation macro F1)".format(output))


if __name__ == "__main__":
    main()
