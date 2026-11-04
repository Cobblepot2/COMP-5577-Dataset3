"""Run the configured white-box attack sweep on the fixed diagnostic split."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

from .common import (ImageRecords, load_records, load_provided_model, provenance,
                     setup, write_json, write_predictions)
from .attacks import fgsm, pgd
from .evaluation import make_loader, predict_clean
from .metrics import classification_metrics
import numpy as np
from PIL import Image, ImageDraw
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--attacks", default="configs/attacks.json")
    parser.add_argument("--output", default="runs/attacks")
    args = parser.parse_args()
    config, device = setup(args.config)
    experiments = json.loads(Path(args.attacks).read_text())["experiments"]
    # Check the entire attack configuration once at this CLI boundary.
    if not experiments or len({e["name"] for e in experiments}) != len(experiments):
        raise ValueError("Attack names must be nonempty and unique")
    for e in experiments:
        if e["method"] not in {"fgsm", "pgd"} or not 0 <= e["epsilon"] <= 8 / 255:
            raise ValueError("Use FGSM/PGD with epsilon in [0,8/255]")
        if e["method"] == "pgd" and (e["step_size"] <= 0 or e["steps"] < 1 or e["restarts"] < 1):
            raise ValueError("PGD step size, step count and restart count must be positive")
    data = Path(config["data_dir"])
    rows = load_records(Path(config["split_dir"]) / "validation.csv", labeled=True)
    dataset = ImageRecords(rows, data / "train_images", labeled=True)
    loader = make_loader(dataset, config)
    model = load_provided_model(data, device)
    clean, _, clean_timing = predict_clean(model, loader, device)
    correct = clean == dataset.labels
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_predictions(out / "clean_predictions.csv", dataset.ids, clean)
    rng = np.random.default_rng(config["seed"] + 1)
    mixed_mask = np.zeros(len(dataset), dtype=bool)
    mixed_mask[rng.choice(len(dataset), len(dataset) // 2, replace=False)] = True
    with (out / "mixed_mask.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "attacked"])
        writer.writerows(zip(dataset.ids, mixed_mask.astype(int)))
    report = provenance(config)
    report.update(clean=classification_metrics(dataset.labels, clean), clean_timing=clean_timing,
                  initially_correct=int(correct.sum()), mixed_attacked_records=int(mixed_mask.sum()),
                  mixed_mask_seed=config["seed"] + 1, attacks=[])
    survives_all = correct.copy()
    examples = {}
    for experiment in experiments:
        generator = torch.Generator(device=device).manual_seed(config["seed"])
        # Warm up this attack shape outside the timed section, with a separate RNG.
        warmup, y_warm = next(iter(loader))
        warmup, y_warm = warmup.to(device), y_warm.to(device)
        if experiment["method"] == "fgsm":
            fgsm(model, warmup, y_warm, experiment["epsilon"])
        else:
            pgd(model, warmup, y_warm, experiment["epsilon"], experiment["step_size"],
                experiment["steps"], experiment["restarts"],
                generator=torch.Generator(device=device).manual_seed(config["seed"]))
        del warmup, y_warm
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        predictions, quantized_predictions = [], []
        max_delta = torch.tensor(0., device=device)
        max_quantized_delta = torch.tensor(0., device=device)
        min_pixel = torch.tensor(1., device=device)
        max_pixel = torch.tensor(0., device=device)
        offset = 0
        for batch, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            if experiment["method"] == "fgsm":
                adv = fgsm(model, images, labels, experiment["epsilon"])
            else:
                adv = pgd(model, images, labels, experiment["epsilon"], experiment["step_size"],
                          experiment["steps"], experiment["restarts"], generator=generator)
            with torch.no_grad():
                pred = model(adv).argmax(1).cpu().numpy()
                # PNG-style 8-bit quantization is evaluated separately from float attacks.
                quantized = (adv * 255).round() / 255
                quantized_predictions.append(model(quantized).argmax(1).cpu().numpy())
                max_delta = torch.maximum(max_delta, (adv - images).abs().amax())
                max_quantized_delta = torch.maximum(max_quantized_delta, (quantized - images).abs().amax())
                min_pixel = torch.minimum(min_pixel, adv.amin())
                max_pixel = torch.maximum(max_pixel, adv.amax())
            predictions.append(pred)
            if experiment["name"] == experiments[-1]["name"]:
                for j, label in enumerate(labels.cpu().tolist()):
                    if label not in examples and correct[offset + j] and pred[j] != label:
                        examples[label] = {"id": dataset.ids[offset + j], "label": label,
                                           "clean_prediction": int(clean[offset + j]), "adversarial_prediction": int(pred[j]),
                                           "original": images[j].detach().cpu().numpy(),
                                           "adversarial": adv[j].detach().cpu().numpy()}
            offset += len(images)
            if (batch + 1) % 10 == 0 or offset == len(dataset):
                print(f"{experiment['name']}: {offset}/{len(dataset)}", flush=True)
        torch.cuda.synchronize()
        seconds = perf_counter() - start
        pred = np.concatenate(predictions)
        quantized_pred = np.concatenate(quantized_predictions)
        # These are measured attack postconditions, not repeated input validation.
        assert max_delta.item() <= experiment["epsilon"] + 1e-6
        assert min_pixel.item() >= 0 and max_pixel.item() <= 1
        success = correct & (pred != dataset.labels)
        survives_all &= pred == dataset.labels
        entry = {"settings": experiment, "float_metrics": classification_metrics(dataset.labels, pred),
                 "quantized_8bit_metrics": classification_metrics(dataset.labels, quantized_pred),
                 "mixed_50pct_metrics": classification_metrics(dataset.labels, np.where(mixed_mask, pred, clean)),
                 "attack_successes_on_initially_correct": int(success.sum()),
                 "attack_success_rate_on_initially_correct": float(success.sum() / correct.sum()),
                 "observed_linf": max_delta.item(), "observed_pixel_range": [min_pixel.item(), max_pixel.item()],
                 "observed_quantized_linf": max_quantized_delta.item(),
                 "seconds": seconds, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                 "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
        report["attacks"].append(entry)
        write_predictions(out / f"{experiment['name']}_predictions.csv", dataset.ids, pred)
        write_predictions(out / f"{experiment['name']}_quantized_predictions.csv", dataset.ids, quantized_pred)
        write_json(out / "metrics.json", report)
        print(f"{experiment['name']}: F1={entry['float_metrics']['macro_f1']:.6f}; ASR={entry['attack_success_rate_on_initially_correct']:.6f}; {seconds:.2f}s", flush=True)
    report["empirical_accuracy_surviving_clean_and_all_configured_float_attacks"] = float(survives_all.mean())
    report["timing_scope"] = "Per-attack warm-up excluded; includes loader/transfers, attack generation, float and quantized inference, bound statistics and example selection; excludes image cache loading and metric/file writing"
    report["pgd_selection"] = "Best per sample across clean input, all iterates and restarts: misclassification first, then cross entropy. No FGSM candidate inclusion."
    report["certified_robustness"] = False
    write_json(out / "metrics.json", report)
    # One first-found success per true class; selection is descriptive, not a random sample.
    canvas = Image.new("RGB", (720, 45 + 150 * len(examples)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "Original | perturbation (scaled to full contrast) | adversarial", fill="black")
    example_metadata = []
    for row, (label, example) in enumerate(sorted(examples.items())):
        orig, adv = example["original"], example["adversarial"]
        delta = (adv - orig) / experiments[-1]["epsilon"] / 2 + 0.5
        for col, pixels in enumerate([orig, delta, adv]):
            array = np.rint(pixels.transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
            thumb = Image.fromarray(array).resize((128, 128), Image.Resampling.NEAREST)
            canvas.paste(thumb, (10 + col * 145, 40 + row * 150))
        draw.text((450, 70 + row * 150), f"id={example['id']}  true={label}\nclean={example['clean_prediction']}  adv={example['adversarial_prediction']}", fill="black")
        example_metadata.append({k: v for k, v in example.items() if k not in {"original", "adversarial"}})
    canvas.save(out / "attack_examples.png")
    write_json(out / "attack_examples.json", {"attack": experiments[-1], "selection": "First initially-correct attack success for each true class", "examples": example_metadata})


if __name__ == "__main__":
    main()
