"""Explicit TRADES and self/source hybrid training on the fixed Dataset3 split."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .attacks import fgsm, pgd
from .common import ImageRecords, load_provided_model, load_records, provenance, setup, write_json
from .metrics import classification_metrics
from .training import (_evaluation_mode, augment_training, initialize_model, load_checkpoint,
                       save_checkpoint, training_pgd, validate, validate_training_config)


def clean_to_adversarial_kl(clean_logits, adversarial_logits):
    """KL(p_clean || p_adversarial), sum classes then mean over the batch.

    Neither outer branch is detached. The inner attack explicitly detaches its
    separately computed eval-mode reference instead.
    """
    probabilities = F.softmax(clean_logits, dim=1)
    return (probabilities * (F.log_softmax(clean_logits, dim=1)
                             - F.log_softmax(adversarial_logits, dim=1))).sum(1).mean()


def trades_objective(clean_logits, adversarial_logits, labels, beta):
    natural = F.cross_entropy(clean_logits, labels)
    divergence = clean_to_adversarial_kl(clean_logits, adversarial_logits)
    return natural + beta * divergence, natural, divergence


def trades_adversary(model, images, epsilon, step_size, steps, generator=None):
    """One uniform random start and final-iterate KL ascent in pixel space."""
    original = images.detach()
    lower, upper = (original - epsilon).clamp(0, 1), (original + epsilon).clamp(0, 1)
    noise = torch.empty_like(original).uniform_(-epsilon, epsilon, generator=generator)
    current = (original + noise).clamp(0, 1)
    with _evaluation_mode(model), torch.enable_grad():
        with torch.no_grad():
            reference = F.softmax(model(original), dim=1).detach()
            log_reference = reference.clamp_min(torch.finfo(reference.dtype).tiny).log()
        for _ in range(steps):
            current = current.detach().requires_grad_(True)
            log_adversarial = F.log_softmax(model(current), dim=1)
            divergence = (reference * (log_reference - log_adversarial)).sum(1).mean()
            gradient, = torch.autograd.grad(divergence, current)
            current = current.detach() + step_size * gradient.sign()
            current = torch.maximum(torch.minimum(current, upper), lower)
    return current.detach()


def hybrid_adversary(model, source, images, labels, epsilon, step_size, steps):
    """First floor-half uses the current model, remaining samples frozen source."""
    split = len(images) // 2
    pieces = []
    if split:
        pieces.append(training_pgd(model, images[:split], labels[:split], epsilon, step_size, steps))
    pieces.append(training_pgd(source, images[split:], labels[split:], epsilon, step_size, steps))
    return torch.cat(pieces)


def train_one_epoch(model, loader, optimizer, device, settings, source=None):
    model.train()
    if source is not None:
        source.eval().requires_grad_(False)
    totals = {"loss": 0.0, "clean_ce": 0.0, "adversarial_ce": 0.0, "kl": 0.0}
    records, self_records, source_records = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if settings["augmentation"]:
            images = augment_training(images)
        optimizer.zero_grad(set_to_none=True)
        if settings["mode"] == "trades":
            adversarial = trades_adversary(model, images, settings["epsilon"],
                                           settings["step_size"], settings["steps"])
            self_records += len(images)
        else:
            adversarial = hybrid_adversary(model, source, images, labels, settings["epsilon"],
                                           settings["step_size"], settings["steps"])
            self_records += len(images) // 2
            source_records += len(images) - len(images) // 2
        clean_logits, adversarial_logits = model(images), model(adversarial)
        adversarial_ce = F.cross_entropy(adversarial_logits, labels)
        if settings["mode"] == "trades":
            loss, clean_ce, divergence = trades_objective(
                clean_logits, adversarial_logits, labels, settings["beta"])
        else:
            clean_ce = F.cross_entropy(clean_logits, labels)
            divergence = clean_to_adversarial_kl(clean_logits, adversarial_logits)
            weight = settings["clean_weight"]
            loss = weight * clean_ce + (1 - weight) * adversarial_ce
        loss.backward()
        optimizer.step()
        for name, value in [("loss", loss), ("clean_ce", clean_ce),
                            ("adversarial_ce", adversarial_ce), ("kl", divergence)]:
            totals[name] += value.detach().item() * len(images)
        records += len(images)
    return {"records": records, "self_attack_records": self_records,
            "source_attack_records": source_records,
            **{name: value / records for name, value in totals.items()}}


def source_prediction_metrics(labels, clean, adversarial, seed=5577, num_classes=10):
    mask = np.zeros(len(labels), dtype=bool)
    mask[np.random.default_rng(seed + 1).choice(len(labels), len(labels) // 2, replace=False)] = True
    return {"float_metrics": classification_metrics(labels, adversarial, num_classes),
            "mixed_50pct_metrics": classification_metrics(labels, np.where(mask, adversarial, clean), num_classes),
            "mixed_mask_seed": seed + 1, "mixed_attacked_records": int(mask.sum())}


def selection_score(own_validation, source_validation, clean_floor):
    clean = own_validation["clean"]["macro_f1"]
    worst = min(value["mixed_50pct_metrics"]["macro_f1"] for value in source_validation.values())
    eligible = clean >= clean_floor
    return {"eligible": eligible, "clean_floor": clean_floor, "clean_macro_f1": clean,
            "worst_source_mixed_macro_f1": worst,
            "own_pgd10_mixed_macro_f1": own_validation["mixed_50pct"]["macro_f1"],
            "rank": [int(eligible), worst, clean]}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_source_cache(source, dataset, config, device, cache_dir):
    """Generate/reuse source attacks once, with all identity work at this boundary."""
    root, split = Path(config["data_dir"]), Path(config["split_dir"]) / "validation.csv"
    identity = {"provided_model_sha256": _sha(root / "provided_model.pt"),
                "architecture_sha256": _sha(root / "dataset3_model.py"),
                "validation_csv_sha256": _sha(split),
                "input_pixels_sha256": hashlib.sha256(dataset.images.numpy().tobytes()).hexdigest(),
                "attack_code_sha256": _sha(Path(__file__).with_name("attacks.py")),
                "cache_code_sha256": _sha(__file__), "seed": config["seed"],
                "batch_size": config["batch_size"], "records": len(dataset),
                "input_range": [0, 1], "torch": str(torch.__version__)}
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    cached, metadata = {}, {}
    attacks = {"fgsm8": {"method": "fgsm", "epsilon": 8 / 255},
               "pgd20x3": {"method": "pgd", "epsilon": 8 / 255,
                            "step_size": 2 / 255, "steps": 20, "restarts": 3}}
    for name, settings in attacks.items():
        current_identity = dict(identity, settings=settings)
        digest = hashlib.sha256(json.dumps(current_identity, sort_keys=True).encode()).hexdigest()
        path = directory / f"{digest}.pt"
        reused = path.exists()
        if reused:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["identity"] != current_identity:
                raise ValueError("Source cache identity mismatch")
            attacked = payload["adversarial"]
        else:
            # Dedicated loader/attack generators keep cache creation from consuming
            # the global streams used for augmentation, dropout and training PGD.
            loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                                num_workers=config["num_workers"],
                                generator=torch.Generator().manual_seed(config["seed"]))
            pieces = []
            with _evaluation_mode(source):
                for index, (images, labels) in enumerate(loader):
                    images, labels = images.to(device), labels.to(device)
                    if settings["method"] == "fgsm":
                        result = fgsm(source, images, labels, settings["epsilon"])
                    else:
                        result = pgd(source, images, labels, settings["epsilon"], settings["step_size"],
                                     settings["steps"], settings["restarts"],
                                     generator=torch.Generator(device=device).manual_seed(config["seed"] + index))
                    pieces.append(result.cpu())
            attacked = torch.cat(pieces)
        clean = dataset.images.float().div(255)
        if (attacked.shape != clean.shape or not torch.isfinite(attacked).all()
                or attacked.min() < 0 or attacked.max() > 1
                or (attacked - clean).abs().max() > 8 / 255 + 1e-6):
            raise ValueError("Source cache violates shape/pixel/epsilon bounds")
        if not reused:
            temporary = path.with_suffix(".tmp")
            torch.save({"identity": current_identity, "adversarial": attacked}, temporary)
            temporary.replace(path)
        cached[name] = attacked
        metadata[name] = {"path": str(path), "identity": current_identity, "reused": reused}
        print(f"source cache {name}: {'reused' if reused else 'created'}", flush=True)
    return cached, metadata


def evaluate_source_conditions(model, dataset, cached, config, device):
    clean_predictions, predictions = [], {name: [] for name in cached}
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"])
    offset = 0
    with _evaluation_mode(model), torch.inference_mode():
        for images, _ in loader:
            clean_predictions.append(model(images.to(device)).argmax(1).cpu().numpy())
            for name, attacked in cached.items():
                predictions[name].append(model(attacked[offset:offset + len(images)].to(device)).argmax(1).cpu().numpy())
            offset += len(images)
    clean_predictions = np.concatenate(clean_predictions)
    return {name: source_prediction_metrics(dataset.labels, clean_predictions, np.concatenate(values), config["seed"])
            for name, values in predictions.items()}


def load_training_protocol(config, final_refit):
    if final_refit:
        path = Path(config["data_dir"]) / "train.csv"
        return {"training_path": path, "validation_path": None,
                "training_rows": load_records(path, True), "validation_rows": [],
                "scope": "Full-data final fit, no held-out metric; configuration and horizon frozen before refit"}
    train, validation = (Path(config["split_dir"]) / name for name in ["train.csv", "validation.csv"])
    training_rows, validation_rows = load_records(train, True), load_records(validation, True)
    if {row["id"] for row in training_rows} & {row["id"] for row in validation_rows}:
        raise ValueError("Normal training requires disjoint fixed training/validation IDs")
    return {"training_path": train, "validation_path": validation,
            "training_rows": training_rows, "validation_rows": validation_rows,
            "scope": "Supplied-weight descendant: fixed validation is diagnostic and used for model selection"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--training", default="configs/improve_trades_beta1.json")
    parser.add_argument("--output", default="runs/improve_trades_beta1")
    parser.add_argument("--transfer-cache", default="runs/improve_validation_cache")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--final-refit", action="store_true",
                        help="After freezing config/horizon, fit full train.csv without validation or selection")
    args = parser.parse_args()
    settings = json.loads(Path(args.training).read_text(encoding="utf-8"))
    validate_training_config(dict(settings, mode="pgd"))
    if settings["mode"] not in {"trades", "hybrid"} or settings["initialization"] != "provided":
        raise ValueError("Use trades/hybrid with supplied initialization")
    if (not math.isfinite(settings["beta"]) or settings["beta"] <= 0
            or not math.isfinite(settings["clean_floor"]) or not 0 <= settings["clean_floor"] <= 1):
        raise ValueError("Use positive finite beta and clean_floor in [0,1]")
    settings["final_refit"] = args.final_refit
    config, device = setup(args.config)
    if config["seed"] != 5577 or config["input_range"] != [0, 1]:
        raise ValueError("Preserve fixed seed5577 and [0,1] model inputs")
    protocol = load_training_protocol(config, args.final_refit)
    train_data = ImageRecords(protocol["training_rows"], Path(config["data_dir"]) / "train_images", True)
    model = initialize_model(config, settings, device)
    source = load_provided_model(config["data_dir"], device)
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(train_data, batch_size=settings["batch_size"], shuffle=True,
                        generator=generator, num_workers=config["num_workers"], pin_memory=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=settings["learning_rate"],
                                momentum=settings["momentum"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings["epochs"], eta_min=settings.get("min_learning_rate", 0))
    metadata = provenance(config)
    metadata["torch"], metadata["scope"] = str(metadata["torch"]), protocol["scope"]
    metadata["final_refit"] = args.final_refit
    for path in [protocol["training_path"], Path(config["data_dir"]) / "dataset3_model.py",
                 Path(args.config), Path(args.training), Path(__file__), Path(__file__).with_name("training.py"),
                 Path(__file__).with_name("attacks.py")]:
        metadata["sha256"][str(path)] = _sha(path)
    cache_metadata = {}
    if not args.final_refit:
        validation_data = ImageRecords(protocol["validation_rows"], Path(config["data_dir"]) / "train_images", True)
        validation_loader = DataLoader(validation_data, batch_size=config["batch_size"], shuffle=False,
                                       num_workers=config["num_workers"], pin_memory=True)
        cached, cache_metadata = prepare_source_cache(source, validation_data, config, device, args.transfer_cache)
        metadata["source_attack_identities"] = {name: value["identity"] for name, value in cache_metadata.items()}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    history, start_epoch, best = [], 0, None
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, generator, device)
        if state["config"] != config or state["training_config"] != settings:
            raise ValueError("Exact resume requires unchanged config, horizon and final-refit flag")
        if state["provenance"] != metadata:
            raise ValueError("Exact resume requires unchanged source/data/code hashes and protocol")
        history, start_epoch, best = state["history"], state["epoch"], state["best_selection"]
    stop = args.stop_after_epoch if args.stop_after_epoch is not None else settings["epochs"]
    if not 0 < stop <= settings["epochs"] or stop < start_epoch:
        raise ValueError("Stop epoch must be within original horizon and at/after resume epoch")
    report = {"provenance": metadata, "config": config, "training_config": settings,
              "training_records": len(train_data), "validation_records": len(protocol["validation_rows"]),
              "criterion": "Prefer clean>=clean_floor, then maximum minimum source-FGSM8/source-PGD20x3 mixed F1, then clean F1; earliest exact tie",
              "ineligible_policy": "Best ineligible checkpoint is diagnostic only; retain prior eligible model if no eligible candidate",
              "cache": cache_metadata, "selection_performed": not args.final_refit}
    for epoch in range(start_epoch, stop):
        start = perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        torch.cuda.reset_peak_memory_stats(device)
        train = train_one_epoch(model, loader, optimizer, device, settings, source)
        entry = {"epoch": epoch + 1, "learning_rate": learning_rate, "train": train}
        improved = False
        if not args.final_refit:
            own = validate(model, validation_loader, device, settings["validation_attack"], config["seed"])
            transferred = evaluate_source_conditions(model, validation_data, cached, config, device)
            criterion = selection_score(own, transferred, settings["clean_floor"])
            criterion["epoch"] = epoch + 1
            improved = best is None or criterion["rank"] > best["rank"]
            if improved:
                best = criterion
            entry.update(validation=own, source_validation=transferred, criterion=criterion)
        scheduler.step()
        entry.update(seconds=perf_counter() - start,
                     peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20)
        history.append(entry)
        save_checkpoint(out / "last.pt", model, optimizer, scheduler, epoch + 1, generator,
                        config, settings, metadata, history, best)
        if improved:
            save_checkpoint(out / "best.pt", model, optimizer, scheduler, epoch + 1, generator,
                            config, settings, metadata, history, best)
        report.update(history=history, best_selection=best, completed_epochs=epoch + 1)
        write_json(out / "metrics.json", report)
        suffix = (f"clean={criterion['clean_macro_f1']:.6f} "
                  f"worst_source_mixed={criterion['worst_source_mixed_macro_f1']:.6f} eligible={criterion['eligible']}"
                  if not args.final_refit else "final-refit; no validation/selection")
        print(f"epoch {epoch + 1}/{settings['epochs']}: loss={train['loss']:.6f} {suffix} {entry['seconds']:.2f}s", flush=True)
    report.update(history=history, best_selection=best, completed_epochs=stop)
    write_json(out / "metrics.json", report)


if __name__ == "__main__":
    main()
