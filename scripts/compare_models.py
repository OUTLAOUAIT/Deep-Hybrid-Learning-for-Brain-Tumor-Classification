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
        history_file = {}
        best_row = None
        if history_path.exists():
            history_file = json.loads(history_path.read_text(encoding="utf-8"))
            history = history_file["history"]
            best_row = max(history, key=lambda row: row["val"].get(
                "selection_score", row["val"].get("macro_f1", 0.0)))
            val = best_row["val"]
        elif "validation" in data:
            # Fixed soft-voting ensemble: score on validation, test remains held out.
            val = data["validation"]
        else:
            continue
        per_class = data.get("per_class", {})
        per_class_pr = {name: {"precision": scores.get("precision"),
                               "recall": scores.get("recall")}
                        for name, scores in per_class.items()}
        reports.append({"model": path.stem,
                        "selected_epoch": best_row["epoch"] if best_row else "",
                        "best_val_selection_score": val.get("selection_score", val.get("macro_f1")),
                        "val_accuracy": val.get("accuracy"),
                        "val_precision_macro": val.get("precision_macro"),
                        "val_recall_macro": val.get("recall_macro"),
                        "val_f1_macro": val.get("macro_f1"),
                        "training_seconds": history_file.get("training_seconds"),
                        "test_accuracy": data["accuracy"],
                        "test_balanced_accuracy": data.get("balanced_accuracy"),
                        "test_precision_macro": data["precision_macro"],
                        "test_recall_macro": data["recall_macro"], "test_f1_macro": data["f1_macro"],
                        "test_precision_weighted": data.get("precision_weighted"),
                        "test_recall_weighted": data.get("recall_weighted"),
                        "test_per_class_precision_recall": json.dumps(per_class_pr, sort_keys=True),
                        "test_f1_weighted": data["f1_weighted"],
                        "inference_ms_per_image": data.get("latency_ms_per_image"),
                        "total_parameters": data.get("total_parameters"),
                        "trainable_parameters": data.get("trainable_parameters")})
    if not reports:
        raise FileNotFoundError("No evaluation JSON reports found; evaluate checkpoints first")
    reports.sort(key=lambda row: row["best_val_selection_score"] or 0.0, reverse=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    print("Wrote {} (ranked by validation accuracy/macro-precision/macro-recall score)".format(output))


if __name__ == "__main__":
    main()
