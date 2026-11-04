"""Generate provided-model diagnostic metrics and the unlabeled test submission."""
import argparse
from pathlib import Path
from time import perf_counter

from .common import (ImageRecords, load_records, load_provided_model, provenance,
                     setup, write_json, write_predictions)
from .evaluation import make_loader, predict_clean
from .metrics import classification_metrics
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--output", default="runs/baseline")
    parser.add_argument("--submission", default="submissions/submission_baseline.csv")
    args = parser.parse_args()
    config, device = setup(args.config)
    data = Path(config["data_dir"])
    val_rows = load_records(Path(config["split_dir"]) / "validation.csv", labeled=True)
    test_rows = load_records(data / "test.csv", labeled=False)
    template = load_records(data / "sample_submission.csv", labeled=False)
    if [r["id"] for r in test_rows] != [r["id"] for r in template]:
        raise ValueError("test.csv IDs must match the sample submission order")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    Path(args.submission).parent.mkdir(parents=True, exist_ok=True)
    model = load_provided_model(data, device)
    report = provenance(config)
    for name, rows, folder, labeled in [("validation", val_rows, "train_images", True),
                                        ("test", test_rows, "test_images", False)]:
        start = perf_counter()
        dataset = ImageRecords(rows, data / folder, labeled)
        load_seconds = perf_counter() - start
        prediction, logits, timing = predict_clean(model, make_loader(dataset, config), device)
        entry = {"records": len(dataset), "image_loading_seconds": load_seconds, "inference": timing,
                 "predicted_class_counts": np.bincount(prediction, minlength=10).tolist()}
        if labeled:
            entry["diagnostic_metrics"] = classification_metrics(dataset.labels, prediction)
        report[name] = entry
        write_predictions(out / f"{name}_predictions.csv", dataset.ids, prediction)
        np.save(out / f"{name}_logits.npy", logits)
        if name == "test":
            write_predictions(args.submission, dataset.ids, prediction)
        print(f"{name}: {len(dataset)} images, inference {timing['seconds']:.3f}s", flush=True)
    report["kaggle_public_score"] = None
    report["kaggle_status"] = "Not submitted by this command; obtain the actual score from Kaggle"
    write_json(out / "metrics.json", report)
    print(f"Diagnostic Macro-F1: {report['validation']['diagnostic_metrics']['macro_f1']:.6f}")
    print(f"Submission: {args.submission}")


if __name__ == "__main__":
    main()
