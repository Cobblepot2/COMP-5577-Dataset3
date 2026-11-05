"""Write and validate deterministic Dataset3 test predictions; never submit them."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from .common import (ImageRecords, load_records, provenance, setup, write_json,
                     write_predictions)
from .evaluate_defense import load_defense
from .evaluation import make_loader, predict_clean
from .tta import configure_predictor


def validate_submission(ids, predictions, sample_ids, expected_count=3925):
    predictions = np.asarray(predictions)
    if len(ids) != expected_count or len(sample_ids) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} submission records")
    if len(set(ids)) != len(ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Submission/sample IDs must be unique")
    if list(ids) != list(sample_ids):
        raise ValueError("Submission IDs must match the sample submission in exact order")
    if predictions.shape != (expected_count,) or not np.issubdtype(predictions.dtype, np.integer):
        raise ValueError("Expected one integer label for every test ID")
    if np.any((predictions < 0) | (predictions > 9)):
        raise ValueError("Submission labels must be in 0..9")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--checkpoint", required=True, help="Trusted local best.pt/last.pt")
    parser.add_argument("--output", default="runs/inference")
    parser.add_argument("--horizontal-flip", action="store_true",
                        help="Average identity/horizontal-flip probabilities")
    args = parser.parse_args()
    config, device = setup(args.config)
    data = Path(config["data_dir"])
    test_csv = data / "test.csv"
    sample_csv = data / "sample_submission.csv"
    rows = load_records(test_csv, labeled=False)
    sample = load_records(sample_csv, labeled=False)
    ids = [row["id"] for row in rows]
    sample_ids = [row["id"] for row in sample]
    validate_submission(ids, np.zeros(len(ids), dtype=np.int64), sample_ids)
    dataset = ImageRecords(rows, data / "test_images", labeled=False)
    model, checkpoint = load_defense(config, args.checkpoint, device)
    model, transform = configure_predictor(model, args.horizontal_flip)
    checkpoint["inference_transform"] = transform
    loader = make_loader(dataset, config)
    predictions, logits, timing = predict_clean(model, loader, device)
    repeat, repeat_logits, repeat_timing = predict_clean(model, loader, device)
    if not np.array_equal(predictions, repeat) or not np.array_equal(logits, repeat_logits):
        raise RuntimeError("Repeated full-test inference was not bitwise deterministic")
    validate_submission(dataset.ids, predictions, sample_ids)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    submission = out / "submission.csv"
    write_predictions(submission, dataset.ids, predictions)
    saved = load_records(submission, labeled=True)
    validate_submission([r["id"] for r in saved],
                        np.array([int(r["label"]) for r in saved]), sample_ids)
    report = provenance(config)
    report.update(scope="Unlabeled 3925-image test inference; no test accuracy or Kaggle score is inferred",
                  checkpoint=checkpoint, inference_transform=transform,
                  records=len(dataset), output=str(submission),
                  submission_sha256=hashlib.sha256(submission.read_bytes()).hexdigest(),
                  test_csv_sha256=hashlib.sha256(test_csv.read_bytes()).hexdigest(),
                  sample_submission_sha256=hashlib.sha256(sample_csv.read_bytes()).hexdigest(),
                  label_counts=np.bincount(predictions, minlength=10).tolist(),
                  timing=timing, repeat_timing=repeat_timing,
                  deterministic_repeat={"full_passes": 2, "predictions_equal": True,
                                        "logits_bitwise_equal": True}, submitted_to_kaggle=False)
    write_json(out / "inference_metrics.json", report)
    print(f"Validated {len(dataset)} test predictions: {submission}", flush=True)


if __name__ == "__main__":
    main()
