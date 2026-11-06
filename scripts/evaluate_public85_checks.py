"""Post-freeze diagnostics: stratified-200 adaptive stress or full reserved transfer.

This orchestrates the repository's tested attacks; it does not add an attack,
select a model, train/refit, infer a Kaggle score, or change a final selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np

from dataset3.common import (ImageRecords, load_records, provenance, setup,
                             write_json, write_predictions)
from dataset3.evaluate_defense import (attack_suite, evaluate_attack,
                                       fixed_mixed_mask, load_predictor,
                                       load_transfer_source)
from dataset3.evaluation import make_loader, predict_clean
from dataset3.metrics import classification_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_sha(ids):
    encoded = json.dumps(list(ids), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def select_records(full_rows, mode):
    """Choose solely from committed labels and seed, before loading a predictor."""
    if len(full_rows) != 1894:
        raise ValueError("Expected the unchanged 1894-record validation CSV")
    labels = np.array([int(row["label"]) for row in full_rows])
    if mode == "stress":
        generator = np.random.default_rng(5577)
        selected = []
        for label in range(10):
            candidates = np.flatnonzero(labels == label)
            if len(candidates) < 20:
                raise ValueError(f"Class {label} has fewer than 20 validation records")
            selected.extend(generator.choice(candidates, 20, replace=False).tolist())
        indices = np.array(sorted(selected), dtype=np.int64)
    elif mode == "reserved":
        indices = np.arange(len(full_rows), dtype=np.int64)
    else:
        raise ValueError("Use stress or reserved mode")
    rows = [full_rows[int(index)] for index in indices]
    selected_labels = labels[indices]
    metadata = {
        "population_records": len(full_rows),
        "evaluated_records": len(rows),
        "sampling_method": ("Stratified simple random sample: 20 per class, then restore original CSV order"
                            if mode == "stress" else "Complete fixed validation CSV in original order"),
        "sampling_seed": 5577 if mode == "stress" else None,
        "sampling_depends_on_model_predictions_or_errors": False,
        "per_class_counts": np.bincount(selected_labels, minlength=10).tolist(),
        "selected_ids": [row["id"] for row in rows],
        "selected_ids_sha256": ids_sha(row["id"] for row in rows),
        "selected_ids_sha256_encoding": "UTF-8 compact JSON array of ID strings, ensure_ascii=False",
        "original_validation_indices_zero_based": indices.tolist(),
        "scores_apply_to_full_1894_validation": mode == "reserved",
        "sampled_stress_is_separate_from_full_suite_scores": mode == "stress",
    }
    return rows, metadata


def exposure_scope(pipeline):
    active = [member for member in pipeline["members"] if member["weight"] > 0]
    full_refit = any(member.get("final_refit") is True for member in active)
    if full_refit:
        scope = ("Post-freeze in-sample diagnostic security checks: at least one active deployed member "
                 "was refitted on all 9469 supplied training records, including these validation IDs. "
                 "No held-out performance estimate; not used for post-refit model/epoch selection.")
    else:
        scope = ("Post-freeze local diagnostic security checks on validation used for selection. "
                 "Supplied models/attack teachers were exposed to these validation IDs; "
                 "not an independent held-out performance estimate.")
    return scope, full_refit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ensemble-checkpoint")
    parser.add_argument("--ensemble-weight", type=float, default=.5)
    parser.add_argument("--horizontal-flip", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--transfer-cache", default="runs/public85_final_check_transfer_cache")
    parser.add_argument("--mode", choices=("stress", "reserved"), required=True)
    args = parser.parse_args()
    os.chdir(REPO)
    config, device = setup(args.config)
    if config["seed"] != 5577 or config["input_range"] != [0, 1]:
        raise ValueError("Preserve seed5577 and original [0,1] pixel inputs")
    split = Path(config["split_dir"]) / "validation.csv"
    full_rows = load_records(split, labeled=True)
    # Label-based sampling is fixed before loading any candidate/predictions.
    rows, sampling = select_records(full_rows, args.mode)
    dataset = ImageRecords(rows, Path(config["data_dir"]) / "train_images", labeled=True)
    model, primary, pipeline = load_predictor(
        config, args.checkpoint, device, args.horizontal_flip,
        args.ensemble_checkpoint, args.ensemble_weight)
    clean, _, clean_timing = predict_clean(model, make_loader(dataset, config), device)
    mask = fixed_mixed_mask(len(dataset))
    out = Path(args.output or f"runs/public85_final_checks_{args.mode}")
    out.mkdir(parents=True, exist_ok=True)
    selected_csv = out / "selected_ids.csv"
    with selected_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "label", "original_validation_index_zero_based"])
        writer.writerows((row["id"], row["label"], index) for row, index in
                         zip(rows, sampling["original_validation_indices_zero_based"]))
    write_predictions(out / "clean_predictions.csv", dataset.ids, clean)
    with (out / "mixed_mask.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "attacked"])
        writer.writerows(zip(dataset.ids, mask.astype(int)))
    scope, full_refit = exposure_scope(pipeline)
    report = provenance(config)
    report.update(
        mode=args.mode, scope=scope, frozen_pipeline_security_check_only=True,
        contains_active_full_refit_member=full_refit, checkpoint=primary,
        inference_pipeline=pipeline, inference_transform=pipeline["inference_transform"],
        sampling=dict(sampling, selected_ids_csv=str(selected_csv),
                      selected_ids_csv_sha256=sha(selected_csv),
                      full_validation_csv=str(split), full_validation_csv_sha256=sha(split)),
        script_sha256=sha(__file__),
        implementation_sha256={name: sha(REPO / "dataset3" / name) for name in
                               ("evaluate_defense.py", "attacks.py", "stress_attacks.py", "tta.py")},
        clean=classification_metrics(dataset.labels, clean), clean_timing=clean_timing,
        initially_correct=int((clean == dataset.labels).sum()),
        mixed_attacked_records=int(mask.sum()), mixed_clean_records=int((~mask).sum()),
        mixed_mask_seed=5578, mixed_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
        mixed_mask_scope=("Fixed 100 attacked / 100 clean within the selected 200, separate from the full 947 mask"
                          if args.mode == "stress" else "Original fixed 947 attacked / 947 clean on all 1894 records"),
        attack_epsilon=8 / 255, attacks=[], completed=False, certified_robustness=False,
        kaggle_score_inferred=False,
        attack_selection="Existing per-image failure-retaining CE/margin PGD; no new attack implementation",
    )
    if args.mode == "stress":
        names = {"white_box_pgd20x3", "white_box_pgd50x5", "white_box_margin_pgd50x5"}
        settings_list = [item for item in attack_suite("full", stress=True) if item["name"] in names]
        source, cache_identity, cache_dir = model, None, None
        report.update(
            threat_model="White-box attacks differentiate through the complete actual deployed predictor",
            coverage_limitation="CE/margin50x5 stress uses only a fixed stratified 200; not a full1894 stress result or a robustness certificate",
            stress_comparison="CE20x3 and CE50x5 use the SAME 200, ordering, batch size and per-batch seeds",
        )
    else:
        reserved_path = Path("checkpoints/pgd_random_converged.pt")
        source, source_metadata, source_identity = load_transfer_source(config, reserved_path, device)
        if source_metadata["initialization"] != "random" or source_metadata["epoch"] != 50:
            raise ValueError("Reserved transfer source must be the frozen random50 checkpoint")
        cache_identity = dict(source_identity, validation_csv_sha256=sha(split),
                              input_pixels_sha256=hashlib.sha256(dataset.images.numpy().tobytes()).hexdigest())
        cache_dir = args.transfer_cache
        settings_list = [item for item in attack_suite("full")
                         if item["name"] in {"transfer_fgsm8", "transfer_pgd20x3"}]
        report.update(
            transfer_source=source_metadata, source=cache_identity,
            threat_model="Full1894 frozen unwrapped random50-source transfer; no repeated white-box cases",
            reserved_source_not_used_to_generate_new_candidate_training_attacks=True,
        )
    write_json(out / "metrics.json", report)
    predictions, quantized_predictions = {}, {}
    survives_float = clean == dataset.labels
    survives_quantized = survives_float.copy()
    for settings in settings_list:
        entry, pred, quantized = evaluate_attack(
            model, source, dataset, config, device, settings, clean, mask,
            cache_dir=cache_dir, cache_identity=cache_identity)
        report["attacks"].append(entry)
        name = settings["name"]
        predictions[name], quantized_predictions[name] = pred, quantized
        survives_float &= pred == dataset.labels
        survives_quantized &= quantized == dataset.labels
        write_predictions(out / f"{name}_predictions.csv", dataset.ids, pred)
        write_predictions(out / f"{name}_quantized_predictions.csv", dataset.ids, quantized)
        write_json(out / "metrics.json", report)
        print(f"{args.mode} {name}: {len(dataset)} records, floatF1={entry['float_metrics']['macro_f1']:.6f}", flush=True)
    report.update(
        empirical_accuracy_surviving_clean_and_all_configured_float_attacks=float(survives_float.mean()),
        empirical_accuracy_surviving_clean_and_all_configured_8bit_attacks=float(survives_quantized.mean()),
        lowest_configured_attack_mixed_macro_f1=min(entry["mixed_50pct_metrics"]["macro_f1"] for entry in report["attacks"]),
    )
    if args.mode == "stress":
        ce20 = predictions["white_box_pgd20x3"] == dataset.labels
        ce50 = predictions["white_box_pgd50x5"] == dataset.labels
        margin50 = predictions["white_box_margin_pgd50x5"] == dataset.labels
        recovered = int((~ce20 & ce50).sum())
        report["white_box_stress_check_on_same_stratified_200"] = {
            "records": len(dataset), "ce20_accuracy": float(ce20.mean()),
            "ce50_accuracy": float(ce50.mean()), "margin50_accuracy": float(margin50.mean()),
            "new_ce50_failures_beyond_ce20": int((ce20 & ~ce50).sum()),
            "recovered_ce20_failures_under_ce50": recovered,
            "shared_prefix_failure_retention_passed": recovered == 0,
            "margin_additional_failures_beyond_ce50": int((ce50 & ~margin50).sum()),
            "ce50_additional_failures_beyond_margin": int((margin50 & ~ce50).sum()),
            "accuracy_surviving_clean_ce50_and_margin": float(((clean == dataset.labels) & ce50 & margin50).mean()),
            "interpretation": "Only the sampled200; finite adaptive stress, no certification or full1894 inference",
        }
    report["completed"] = True
    write_json(out / "metrics.json", report)
    print(f"Completed {args.mode}: {len(dataset)} evaluated records; {out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
