"""Archive and verify the fixed ten-epoch clean/adversarial weight comparison."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset3.metrics import classification_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/loss_weight_10")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/loss_weights_10")
    args = parser.parse_args()
    with (ROOT / "splits/validation.csv").open(newline="", encoding="utf-8-sig") as stream:
        records = list(csv.DictReader(stream))
    ids = [row["id"] for row in records]
    labels = np.array([int(row["label"]) for row in records])
    summary = {"date": "2026-11-07", "timezone": "Asia/Shanghai", "seed": 5577,
               "scope": "Post-hoc, single-seed, fixed-epoch local diagnostic; no Kaggle submission or change to deployed models",
               "primary_checkpoint_rule": "Completed epoch 10 of a 10-epoch cosine horizon, independent of validation scores",
               "training_records": 7575, "validation_records": 1894, "runs": []}
    reference_settings = None
    reference_mask = None
    source_predictions = {}
    verified_csvs = 0
    for name, weight in [("clean50", .5), ("clean75", .75)]:
        run = args.runs / name
        evaluation = args.runs / f"eval_{name}"
        train = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        report = json.loads((evaluation / "metrics.json").read_text(encoding="utf-8"))
        settings = dict(train["training_config"])
        assert settings.pop("clean_weight") == weight
        if reference_settings is None:
            reference_settings = settings
        assert settings == reference_settings, "Configurations must differ only in clean_weight"
        assert train["completed_epochs"] == report["checkpoint"]["epoch"] == 10
        assert len(train["history"]) == 10 and len(report["attacks"]) == 6
        assert train["training_records"] == 7575 and train["validation_records"] == len(ids) == 1894
        assert train["history"][-1]["validation"]["clean"] == report["clean"]
        with (evaluation / "mixed_mask.csv").open(newline="", encoding="utf-8") as stream:
            mask_rows = list(csv.DictReader(stream))
        assert [row["id"] for row in mask_rows] == ids
        mask = np.array([int(row["attacked"]) for row in mask_rows], dtype=bool)
        assert mask.sum() == 947
        if reference_mask is None:
            reference_mask = mask
        assert np.array_equal(mask, reference_mask)

        predictions = {}
        hashes = {}
        for path in sorted(evaluation.glob("*_predictions.csv")):
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            assert [row["id"] for row in rows] == ids
            predictions[path.stem] = np.array([int(row["label"]) for row in rows])
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            verified_csvs += 1
        clean = predictions["clean_predictions"]
        assert classification_metrics(labels, clean) == report["clean"]
        attacks = {}
        for attack in report["attacks"]:
            attack_name = attack["settings"]["name"]
            pred = predictions[f"{attack_name}_predictions"]
            quantized = predictions[f"{attack_name}_quantized_predictions"]
            for recorded, actual in [
                (attack["float_metrics"], classification_metrics(labels, pred)),
                (attack["quantized_8bit_metrics"], classification_metrics(labels, quantized)),
                (attack["mixed_50pct_metrics"], classification_metrics(labels, np.where(mask, pred, clean))),
                (attack["mixed_50pct_quantized_8bit_metrics"], classification_metrics(labels, np.where(mask, quantized, clean))),
            ]:
                assert recorded == actual, (name, attack_name)
            assert attack["observed_linf"] <= 8 / 255 + 1e-6
            assert attack["observed_quantized_linf"] <= 8 / 255 + 1e-6
            assert 0 <= attack["observed_pixel_range"][0] <= attack["observed_pixel_range"][1] <= 1
            if attack_name.startswith("transfer_"):
                if name == "clean50":
                    source_predictions[attack_name] = attack["cache"]["path"]
                else:
                    assert attack["cache"]["reused"]
                    assert attack["cache"]["path"] == source_predictions[attack_name]
            attacks[attack_name] = {key: attack[key]["macro_f1"] for key in
                                  ["float_metrics", "mixed_50pct_metrics", "quantized_8bit_metrics", "mixed_50pct_quantized_8bit_metrics"]}
        weak = predictions["white_box_pgd10_predictions"] == labels
        strong = predictions["white_box_pgd20x3_predictions"] == labels
        assert not np.any(~weak & strong)
        assert report["white_box_pgd_strength_check"]["recovered_under_pgd20x3"] == 0

        destination = args.output / name
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run / "metrics.json", destination / "training.json")
        shutil.copy2(evaluation / "metrics.json", destination / "evaluation.json")
        for path in sorted(evaluation.glob("*.csv")):
            shutil.copy2(path, destination / path.name)
        summary["runs"].append({"name": name, "clean_weight": weight, "adversarial_weight": 1 - weight,
                                "final_clean_macro_f1": report["clean"]["macro_f1"],
                                "final_epoch_training_validation": train["history"][-1]["validation"],
                                "training_selected_best": train["best_selection"],
                                "attacks": attacks, "training_config": train["training_config"],
                                "provenance": train["provenance"],
                                "final_checkpoint_sha256": report["checkpoint"]["checkpoint_sha256"],
                                "csv_sha256": hashes,
                                "total_epoch_seconds": sum(entry["seconds"] for entry in train["history"]),
                                "peak_training_allocated_mib": max(entry["peak_allocated_mib"] for entry in train["history"]),
                                "history": [{"epoch": entry["epoch"], "learning_rate": entry["learning_rate"],
                                             "clean_macro_f1": entry["validation"]["clean"]["macro_f1"],
                                             "own_pgd10_macro_f1": entry["validation"]["adversarial"]["macro_f1"],
                                             "own_pgd10_mixed_macro_f1": entry["validation"]["mixed_50pct"]["macro_f1"]}
                                            for entry in train["history"]]})
    summary["verification"] = {"configurations_differ_only_in_clean_weight": True,
                               "fixed_tenth_epoch_both_runs": True, "prediction_csvs_verified": verified_csvs,
                               "all_float_quantized_and_mixed_metrics_recomputed": True,
                               "shared_original_source_attack_cache": True,
                               "all_attack_bounds_passed": True, "pgd_failure_retention_passed": True}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(args.output / "summary.json"), "runs": [
        {"name": run["name"], "clean_f1": run["final_clean_macro_f1"], "attacks": run["attacks"]} for run in summary["runs"]],
        "verification": summary["verification"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
