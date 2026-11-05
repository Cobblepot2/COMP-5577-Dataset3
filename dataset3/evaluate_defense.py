"""Measure a defense on the fixed validation IDs using white-box/transfer attacks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .attacks import fgsm, pgd
from .common import (ImageRecords, load_provided_model, load_records, provenance,
                     setup, write_json, write_predictions)
from .evaluation import make_loader, predict_clean
from .metrics import classification_metrics


def load_defense(config, checkpoint, device):
    """Load the common architecture and strictly replace it with trained weights.

    Training checkpoints contain NumPy/Python RNG state and therefore require
    trusted local checkpoint deserialization, unlike the tensor-only source.
    """
    model = load_provided_model(config["data_dir"], device)
    if checkpoint is None:
        path = Path(config["data_dir"]) / "provided_model.pt"
        return model, {"checkpoint": str(path),
                       "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "initialization": "provided baseline", "epoch": 0}
    path = Path(checkpoint)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model_state_dict" not in state:
        raise ValueError("Expected a training checkpoint with model_state_dict")
    for key in ("num_classes", "image_size", "input_range"):
        if key in state.get("config", {}) and key in config and state["config"][key] != config[key]:
            raise ValueError(f"Checkpoint/config mismatch for {key}")
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    metadata = {key: state[key] for key in
                ("epoch", "training_config", "config", "provenance", "best_selection")
                if key in state}
    metadata.update(checkpoint=str(path),
                    checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return model, metadata


def fixed_mixed_mask(records):
    """The same half of the fixed validation ordering is attacked for every model."""
    mask = np.zeros(records, dtype=bool)
    mask[np.random.default_rng(5578).choice(records, records // 2, replace=False)] = True
    return mask


def attack_metrics(labels, clean, attacked, quantized, mixed_mask):
    correct = clean == labels
    initially_correct = int(correct.sum())
    successes = int((correct & (attacked != labels)).sum())
    quantized_successes = int((correct & (quantized != labels)).sum())
    return {
        "float_metrics": classification_metrics(labels, attacked),
        "quantized_8bit_metrics": classification_metrics(labels, quantized),
        "mixed_50pct_metrics": classification_metrics(labels, np.where(mixed_mask, attacked, clean)),
        "mixed_50pct_quantized_8bit_metrics": classification_metrics(
            labels, np.where(mixed_mask, quantized, clean)),
        "attack_successes_on_initially_correct": successes,
        "attack_success_rate_on_initially_correct": successes / initially_correct if initially_correct else None,
        "quantized_attack_successes_on_initially_correct": quantized_successes,
        "quantized_attack_success_rate_on_initially_correct":
            quantized_successes / initially_correct if initially_correct else None,
    }


def attack_suite(suite):
    settings = []
    methods = [("fgsm8", "fgsm", 1, 1), ("pgd10", "pgd", 10, 1)]
    if suite == "full":
        methods.append(("pgd20x3", "pgd", 20, 3))
    for threat_model in ("white_box", "transfer"):
        for name, method, steps, restarts in methods:
            if suite == "quick" and threat_model == "transfer" and method == "fgsm":
                continue
            entry = {"name": f"{threat_model}_{name}", "threat_model": threat_model,
                     "method": method, "epsilon": 8 / 255}
            if method == "pgd":
                entry.update(step_size=2 / 255, steps=steps, restarts=restarts)
            settings.append(entry)
    return settings


def _generate(source, images, labels, settings, generator):
    if settings["method"] == "fgsm":
        return fgsm(source, images, labels, settings["epsilon"])
    return pgd(source, images, labels, settings["epsilon"], settings["step_size"],
               settings["steps"], settings["restarts"], generator=generator)


def evaluate_attack(model, source, dataset, config, device, settings, clean,
                    mixed_mask, cache_dir=None, cache_identity=None):
    """Attack source, measure target; a transfer cache never depends on target.

    Cached float32 tensors retain the original attack. Quantization is measured
    separately and is never substituted for the attack during cache reuse.
    """
    loader = make_loader(dataset, config)
    cached = None
    cache_path = None
    identity = None
    if cache_dir is not None and settings["threat_model"] == "transfer":
        if not cache_identity:
            raise ValueError("Transfer cache requires source/split/input identity")
        identity = dict(cache_identity, settings=settings, seed=config["seed"],
                        batch_size=config["batch_size"], records=len(dataset),
                        attack_code_sha256=hashlib.sha256(
                            Path(__file__).with_name("attacks.py").read_bytes()).hexdigest(),
                        evaluator_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache_path = Path(cache_dir) / f"{digest}.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            cached = payload["adversarial"]
            if payload["identity"] != identity or len(cached) != len(dataset):
                raise ValueError(f"Invalid transfer cache identity: {cache_path}")
    reused = cached is not None
    warm_images, warm_labels = next(iter(loader))
    warm_images, warm_labels = warm_images.to(device), warm_labels.to(device)
    if reused:
        warm_adv = cached[:len(warm_images)].to(device)
    else:
        warm_adv = _generate(source, warm_images, warm_labels, settings,
                             torch.Generator(device=device).manual_seed(config["seed"]))
    with torch.inference_mode():
        model(warm_adv)
        model((warm_adv * 255).round() / 255)
    del warm_images, warm_labels, warm_adv
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    predictions, quantized_predictions, new_cache = [], [], []
    max_delta, max_quantized_delta, min_pixel, max_pixel = 0., 0., 1., 0.
    offset = 0
    all_labels = []
    for batch_index, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        if reused:
            adv = cached[offset:offset + len(images)].to(device)
        else:
            # Matching per-batch seeds make the first restart of PGD20x3
            # include the same trajectory as PGD10 for a fair strength check.
            generator = torch.Generator(device=device).manual_seed(config["seed"] + batch_index)
            adv = _generate(source, images, labels, settings, generator)
            if cache_path is not None:
                new_cache.append(adv.detach().cpu())
        if adv.shape != images.shape or not torch.isfinite(adv).all():
            raise ValueError("Adversarial tensor shape or finiteness violation")
        with torch.inference_mode():
            quantized = (adv * 255).round() / 255
            predictions.append(model(adv).argmax(1).cpu().numpy())
            quantized_predictions.append(model(quantized).argmax(1).cpu().numpy())
            max_delta = max(max_delta, (adv - images).abs().amax().item())
            max_quantized_delta = max(max_quantized_delta, (quantized - images).abs().amax().item())
            min_pixel = min(min_pixel, adv.amin().item())
            max_pixel = max(max_pixel, adv.amax().item())
        all_labels.append(labels.cpu().numpy())
        offset += len(images)
        if offset == len(dataset) or (batch_index + 1) % 10 == 0:
            print(f"{settings['name']}: {offset}/{len(dataset)}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = perf_counter() - start
    if max_delta > settings["epsilon"] + 1e-6 or min_pixel < 0 or max_pixel > 1:
        raise ValueError("Measured attack violates the pixel-space threat bounds")
    if max_quantized_delta > settings["epsilon"] + 1e-6:
        raise ValueError("Quantized attack violates the 8-bit threat bound")
    if cache_path is not None and not reused:
        temporary = cache_path.with_suffix(".tmp")
        torch.save({"identity": identity, "adversarial": torch.cat(new_cache)}, temporary)
        temporary.replace(cache_path)
    pred, quantized = np.concatenate(predictions), np.concatenate(quantized_predictions)
    report = attack_metrics(np.concatenate(all_labels), clean, pred, quantized, mixed_mask)
    report.update(settings=settings, observed_linf=max_delta,
                  observed_quantized_linf=max_quantized_delta,
                  observed_pixel_range=[min_pixel, max_pixel], seconds=seconds,
                  images_per_second=len(dataset) / seconds,
                  peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
                  peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
                  cache={"reused": reused, "path": str(cache_path) if cache_path else None},
                  timing_scope=("Warm-up excluded; loader/transfers, " +
                                ("cached attack loading to device" if reused else "attack generation") +
                                ", float/8-bit target inference and bound checks included; "
                                "initial image/cache disk loading, cache save, metrics/CSV writing excluded"))
    return report, pred, quantized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--checkpoint", help="Trusted local best.pt/last.pt; omit for provided baseline")
    parser.add_argument("--output", default="runs/defense")
    parser.add_argument("--suite", choices=("quick", "full"), default="full")
    parser.add_argument("--transfer-cache", help="Reuse source float attacks across candidate defenses")
    args = parser.parse_args()
    config, device = setup(args.config)
    if config["seed"] != 5577:
        raise ValueError("The fixed Dataset3 experiment uses seed5577")
    data = Path(config["data_dir"])
    split = Path(config["split_dir"]) / "validation.csv"
    rows = load_records(split, labeled=True)
    if len(rows) != 1894:
        raise ValueError("Expected the fixed 1894-record validation split")
    dataset = ImageRecords(rows, data / "train_images", labeled=True)
    loader = make_loader(dataset, config)
    model, checkpoint = load_defense(config, args.checkpoint, device)
    source = load_provided_model(data, device)
    clean, _, clean_timing = predict_clean(model, loader, device)
    mask = fixed_mixed_mask(len(dataset))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_predictions(out / "clean_predictions.csv", dataset.ids, clean)
    with (out / "mixed_mask.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "attacked"])
        writer.writerows(zip(dataset.ids, mask.astype(int)))
    cache_identity = {"provided_model_sha256": hashlib.sha256((data / "provided_model.pt").read_bytes()).hexdigest(),
                      "validation_csv_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
                      "input_pixels_sha256": hashlib.sha256(dataset.images.numpy().tobytes()).hexdigest(),
                      "source_architecture_sha256": hashlib.sha256((data / "dataset3_model.py").read_bytes()).hexdigest()}
    report = provenance(config)
    report["scope"] = checkpoint.get("provenance", {}).get("scope", report["scope"])
    report.update(checkpoint=checkpoint, suite=args.suite, source=cache_identity,
                  clean=classification_metrics(dataset.labels, clean), clean_timing=clean_timing,
                  initially_correct=int((clean == dataset.labels).sum()),
                  mixed_attacked_records=int(mask.sum()), mixed_mask_seed=5578,
                  mixed_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(), attacks=[],
                  certified_robustness=False,
                  threat_model="Untargeted Linf <=8/255 in [0,1]; transfers use the frozen provided source; white-box attacks use the evaluated defense",
                  pgd_selection="Per-image misclassification first, then maximum CE across clean input, all iterates and restarts; no FGSM injection")
    survives_float = clean == dataset.labels
    survives_quantized = survives_float.copy()
    white_float = survives_float.copy()
    predictions = {}
    for settings in attack_suite(args.suite):
        attack_source = model if settings["threat_model"] == "white_box" else source
        entry, pred, quantized = evaluate_attack(
            model, attack_source, dataset, config, device, settings, clean, mask,
            args.transfer_cache, cache_identity)
        report["attacks"].append(entry)
        survives_float &= pred == dataset.labels
        survives_quantized &= quantized == dataset.labels
        if settings["threat_model"] == "white_box":
            white_float &= pred == dataset.labels
        predictions[settings["name"]] = pred
        write_predictions(out / f"{settings['name']}_predictions.csv", dataset.ids, pred)
        write_predictions(out / f"{settings['name']}_quantized_predictions.csv", dataset.ids, quantized)
        write_json(out / "metrics.json", report)
        print(f"{settings['name']}: F1={entry['float_metrics']['macro_f1']:.6f}; {entry['seconds']:.2f}s", flush=True)
    report["empirical_accuracy_surviving_clean_and_all_configured_float_attacks"] = float(survives_float.mean())
    report["empirical_accuracy_surviving_clean_and_all_configured_8bit_attacks"] = float(survives_quantized.mean())
    report["empirical_accuracy_surviving_clean_and_all_white_box_float_attacks"] = float(white_float.mean())
    if args.suite == "full":
        weak = predictions["white_box_pgd10"] == dataset.labels
        strong = predictions["white_box_pgd20x3"] == dataset.labels
        report["white_box_pgd_strength_check"] = {
            "pgd10_accuracy": float(weak.mean()), "pgd20x3_accuracy": float(strong.mean()),
            "new_failures_under_pgd20x3": int((weak & ~strong).sum()),
            "recovered_under_pgd20x3": int((~weak & strong).sum()),
            "interpretation": "Finite-attack stress test; survival does not certify robustness"}
    write_json(out / "metrics.json", report)


if __name__ == "__main__":
    main()
