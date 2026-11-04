"""Validate CSV metadata once, then persist a deterministic per-class split.

Image integrity was checked during import (docs/dataset3_validation.json).
This entry point reuses that audit and its SHA-256 manifest; it does not rescan
images on every run. Training should consume the saved splits directly.
Run from the repository root: python scripts/prepare_dataset.py
"""
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main():
    config = json.loads(Path("configs/experiment.json").read_text())
    data = Path(config["data_dir"])
    train = read_csv(data / "train.csv")
    test = read_csv(data / "test.csv")
    sample = read_csv(data / "sample_submission.csv")
    classes = read_csv(data / "class_names.csv")
    audit = json.loads(Path("docs/dataset3_validation.json").read_text())
    manifest = read_csv(Path("docs/dataset3_manifest.csv"))
    train_ids = [row["id"] for row in train]
    test_ids = [row["id"] for row in test]
    labels = np.array([int(row["label"]) for row in train])
    # All external metadata checks live here, before split generation.
    if not (0 < config["validation_fraction"] < 1):
        raise ValueError("validation_fraction must lie between 0 and 1")
    if len(set(train_ids)) != len(train) or len(set(test_ids)) != len(test):
        raise ValueError("Duplicate IDs within a CSV")
    if [row["id"] for row in sample] != test_ids:
        raise ValueError("Sample submission ID order differs from test.csv")
    expected_labels = set(range(config["num_classes"]))
    if set(labels) != expected_labels or {int(r["label"]) for r in classes} != expected_labels:
        raise ValueError("Class labels do not match the configured classes")
    if (len(train), len(test)) != (audit["train"]["records"], audit["test"]["records"]):
        raise ValueError("CSV counts differ from the validated import")
    if not audit["copy_hashes_verified"]:
        raise ValueError("Complete the dataset import integrity audit first")
    hashes = {row["path"]: row["sha256"] for row in manifest}
    train_hashes = [hashes[f"train_images/{i}.png"] for i in train_ids]
    test_hashes = {hashes[f"test_images/{i}.png"] for i in test_ids}
    # A duplicate image would require a group-based split instead.
    if len(set(train_hashes)) != len(train_hashes):
        raise ValueError("Duplicate training images found; inspect before splitting")

    rng = np.random.default_rng(config["seed"])
    validation_indices = []
    distribution = []
    for label in sorted(expected_labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        n_val = round(len(indices) * config["validation_fraction"])
        validation_indices.extend(indices[:n_val].tolist())
        distribution.append({"label": label, "total": len(indices),
                             "train": len(indices) - n_val, "validation": n_val})
    validation_indices = set(validation_indices)
    split_dir = Path(config["split_dir"])
    split_dir.mkdir(exist_ok=True)
    for name, is_validation in [("train", False), ("validation", True)]:
        rows = [row for i, row in enumerate(train) if (i in validation_indices) == is_validation]
        with (split_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "label"])
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "seed": config["seed"], "validation_fraction_requested": config["validation_fraction"],
        "method": "NumPy default_rng; per-class permutation; round(n_class * fraction); original row order",
        "training_records": len(train) - len(validation_indices),
        "validation_records": len(validation_indices), "classes": distribution,
        "duplicate_training_file_hashes": len(train_hashes) - len(set(train_hashes)),
        "train_test_shared_file_hashes": len(set(train_hashes) & test_hashes),
        "integrity_source": "docs/dataset3_validation.json and dataset3_manifest.csv (import audit)",
        "limitation": "The provided model was trained on the supplied training data. This split is not unseen data for that model or its fine-tuned descendants. Train new models only on splits/train.csv for held-out evaluation."
    }
    Path("docs/split_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
