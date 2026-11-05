"""Fine-tune ProvidedNet with clean, FGSM or PGD training on fixed CSV splits.

Training PGD performs untargeted cross-entropy maximization by sign ascent and
uses its final iterate, with one optional random start. It deliberately differs
from evaluation PGD's misclassification-first retention across all iterates.
Pixels stay in [0,1]; ProvidedNet retains its own internal normalization.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .attacks import fgsm, pgd
from .common import (ImageRecords, load_provided_model, load_records, provenance,
                     setup, write_json)
from .metrics import classification_metrics


@contextmanager
def _evaluation_mode(model):
    """Do not update batch norm or sample dropout during input optimization."""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def training_pgd(model, images, labels, epsilon, step_size, steps,
                 random_start=True, generator=None):
    """Final-iterate, one-start CE-ascent PGD; no parameter gradients accumulated."""
    original = images.detach()
    lower, upper = (original - epsilon).clamp(0, 1), (original + epsilon).clamp(0, 1)
    current = original.clone()
    if random_start:
        noise = torch.empty_like(original).uniform_(-epsilon, epsilon, generator=generator)
        current = (current + noise).clamp(0, 1)
    with _evaluation_mode(model), torch.enable_grad():
        for _ in range(steps):
            current = current.detach().requires_grad_(True)
            loss = F.cross_entropy(model(current), labels)
            gradient, = torch.autograd.grad(loss, current)
            current = current.detach() + step_size * gradient.sign()
            current = torch.maximum(torch.minimum(current, upper), lower)
    return current.detach()


def initialize_model(config, training_config, device):
    if training_config["initialization"] == "provided":
        model = load_provided_model(config["data_dir"], device)
    else:
        spec = importlib.util.spec_from_file_location(
            "provided_training_architecture", Path(config["data_dir"]) / "dataset3_model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.ProvidedNet().to(device)
    # The shared evaluation loader freezes parameters; training must undo that.
    return model.requires_grad_(True)


def augment_training(images):
    """Optional reflection-pad/random-crop/horizontal-flip, before the attack."""
    padded = F.pad(images, (4, 4, 4, 4), mode="reflect")
    offsets = torch.randint(9, (len(images), 2), device=images.device)
    height, width = images.shape[-2:]
    rows = offsets[:, 0, None, None] + torch.arange(height, device=images.device)[None, :, None]
    cols = offsets[:, 1, None, None] + torch.arange(width, device=images.device)[None, None, :]
    crops = padded.permute(0, 2, 3, 1)[torch.arange(len(images), device=images.device)[:, None, None], rows, cols]
    crops = crops.permute(0, 3, 1, 2)
    flip = torch.rand(len(images), device=images.device) < 0.5
    return torch.where(flip[:, None, None, None], crops.flip(-1), crops)


def train_one_epoch(model, loader, optimizer, device, settings):
    model.train()
    totals = {"loss": 0.0, "clean_loss": 0.0, "adversarial_loss": 0.0}
    records = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if settings.get("augmentation", False):
            images = augment_training(images)
        optimizer.zero_grad(set_to_none=True)
        adversarial = None
        if settings["mode"] == "fgsm":
            with _evaluation_mode(model), torch.enable_grad():
                adversarial = fgsm(model, images, labels, settings["epsilon"])
        elif settings["mode"] == "pgd":
            adversarial = training_pgd(model, images, labels, settings["epsilon"],
                                       settings["step_size"], settings["steps"],
                                       settings.get("random_start", True))
        clean_loss = F.cross_entropy(model(images), labels)
        if adversarial is None:
            adversarial_loss = torch.zeros((), device=device)
            loss = clean_loss
        else:
            adversarial_loss = F.cross_entropy(model(adversarial), labels)
            weight = settings["clean_weight"]
            loss = weight * clean_loss + (1 - weight) * adversarial_loss
        loss.backward()
        optimizer.step()
        count = len(images)
        for key, value in [("loss", loss), ("clean_loss", clean_loss),
                           ("adversarial_loss", adversarial_loss)]:
            totals[key] += float(value.detach()) * count
        records += count
    return {"records": records, **{name: value / records for name, value in totals.items()}}


def validate(model, loader, device, attack, seed, num_classes=10):
    """Evaluate every validation record and recompute F1 on fixed mixed predictions."""
    generator = torch.Generator(device=device).manual_seed(seed)
    clean, adversarial, labels_all = [], [], []
    with _evaluation_mode(model):
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.no_grad():
                clean.append(model(images).argmax(1).cpu().numpy())
            attacked = pgd(model, images, labels, attack["epsilon"], attack["step_size"],
                           attack["steps"], attack["restarts"], generator=generator)
            with torch.no_grad():
                adversarial.append(model(attacked).argmax(1).cpu().numpy())
            labels_all.append(labels.cpu().numpy())
    labels_all, clean, adversarial = map(np.concatenate, (labels_all, clean, adversarial))
    mixed_mask = np.zeros(len(labels_all), dtype=bool)
    rng = np.random.default_rng(seed + 1)
    mixed_mask[rng.choice(len(labels_all), len(labels_all) // 2, replace=False)] = True
    mixed = np.where(mixed_mask, adversarial, clean)
    return {"clean": classification_metrics(labels_all, clean, num_classes),
            "adversarial": classification_metrics(labels_all, adversarial, num_classes),
            "mixed_50pct": classification_metrics(labels_all, mixed, num_classes),
            "mixed_mask_seed": seed + 1, "mixed_attacked_records": int(mixed_mask.sum())}


def save_checkpoint(path, model, optimizer, scheduler, epoch, loader_generator,
                    config, training_config, run_provenance, history, best_selection):
    """Persist a completed epoch, including all state required for the next one."""
    state = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
             "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch,
             "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                           "torch": torch.get_rng_state(),
                           "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
             "loader_generator_state": loader_generator.get_state(), "config": config,
             "training_config": training_config, "provenance": run_provenance,
             "history": history, "best_selection": best_selection}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, loader_generator=None,
                    device="cpu", restore_rng=True):
    # These are trusted local training checkpoints: NumPy/Python RNG state needs pickle.
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if loader_generator is not None:
        loader_generator.set_state(state["loader_generator_state"].cpu())
    if restore_rng:
        rng = state["rng_state"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])
    return state


def validate_training_config(settings):
    if settings["mode"] not in {"clean", "fgsm", "pgd"}:
        raise ValueError("Training mode must be clean, fgsm or pgd")
    if settings["initialization"] not in {"provided", "random"} or settings["scheduler"] != "cosine":
        raise ValueError("Use provided/random initialization and cosine scheduling")
    for name in ["epochs", "batch_size", "steps"]:
        if not isinstance(settings[name], int) or settings[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ["learning_rate", "momentum", "weight_decay", "epsilon", "step_size", "clean_weight"]:
        if not math.isfinite(settings[name]):
            raise ValueError(f"{name} must be finite")
    if settings["learning_rate"] <= 0 or settings["momentum"] < 0 or settings["weight_decay"] < 0:
        raise ValueError("Use positive learning rate and nonnegative momentum/weight decay")
    if not 0 <= settings.get("min_learning_rate", 0) <= settings["learning_rate"]:
        raise ValueError("Minimum learning rate must be between zero and initial learning rate")
    if not 0 <= settings["clean_weight"] <= 1 or not 0 <= settings["epsilon"] <= 8 / 255:
        raise ValueError("Use clean_weight in [0,1] and epsilon in [0,8/255]")
    attack = settings["validation_attack"]
    if settings["step_size"] <= 0 or not 0 <= attack["epsilon"] <= 8 / 255:
        raise ValueError("Use positive PGD step size and validation epsilon in [0,8/255]")
    if not math.isfinite(attack["step_size"]) or attack["step_size"] <= 0:
        raise ValueError("Validation PGD step size must be positive and finite")
    if any(not isinstance(attack[name], int) or attack[name] < 1 for name in ["steps", "restarts"]):
        raise ValueError("Validation PGD steps/restarts must be positive integers")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--training", default="configs/training.json")
    parser.add_argument("--output", default="runs/training")
    parser.add_argument("--resume", help="Trusted local last.pt or best.pt checkpoint")
    parser.add_argument("--stop-after-epoch", type=int,
                        help="Stop at this completed epoch without changing the scheduler horizon")
    args = parser.parse_args()
    settings = json.loads(Path(args.training).read_text(encoding="utf-8"))
    validate_training_config(settings)
    config, device = setup(args.config)
    if config["seed"] != 5577 or config["input_range"] != [0, 1]:
        raise ValueError("This continuation preserves seed 5577 and [0,1] pixels")
    train_path, val_path = (Path(config["split_dir"]) / name for name in ["train.csv", "validation.csv"])
    train_rows, val_rows = load_records(train_path, True), load_records(val_path, True)
    if {row["id"] for row in train_rows} & {row["id"] for row in val_rows}:
        raise ValueError("Fixed training and validation IDs overlap")
    train_data = ImageRecords(train_rows, Path(config["data_dir"]) / "train_images", True)
    val_data = ImageRecords(val_rows, Path(config["data_dir"]) / "train_images", True)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(train_data, batch_size=settings["batch_size"], shuffle=True,
                              generator=generator, num_workers=config["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False,
                            num_workers=config["num_workers"], pin_memory=True)
    model = initialize_model(config, settings, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=settings["learning_rate"],
                                momentum=settings["momentum"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings["epochs"], eta_min=settings.get("min_learning_rate", 0))
    metadata = provenance(config)
    metadata["scope"] = (
        "Randomly initialized ProvidedNet trained only on fixed train.csv; validation excluded from "
        "gradient updates but used for checkpoint selection; independent test needed for unbiased final performance"
        if settings["initialization"] == "random" else
        "Fine-tuned supplied checkpoint; upstream pretraining may include validation records, "
        "so local validation is diagnostic rather than unseen-data performance")
    for path in [train_path, Path(config["data_dir"]) / "dataset3_model.py",
                 Path(args.config), Path(args.training)]:
        metadata["sha256"][path.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    history, start_epoch = [], 0
    best = {"mixed_macro_f1": -1.0, "clean_macro_f1": -1.0, "epoch": 0}
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, generator, device)
        if state["config"] != config or state["training_config"] != settings:
            raise ValueError("Exact resume requires unchanged experiment/training configs (including total epochs)")
        if state["provenance"]["sha256"] != metadata["sha256"]:
            raise ValueError("Exact resume requires unchanged data, splits, architecture and config files")
        history, start_epoch, best = state["history"], state["epoch"], state["best_selection"]
        # Keep the original run provenance when continuing an existing run.
        metadata = state["provenance"]
    stop = args.stop_after_epoch if args.stop_after_epoch is not None else settings["epochs"]
    if not 0 < stop <= settings["epochs"] or stop < start_epoch:
        raise ValueError("stop-after-epoch must be within the configured horizon and at/after resume epoch")
    report = {"provenance": metadata, "config": config, "training_config": settings,
              "training_records": len(train_data), "validation_records": len(val_data),
              "selection": "Maximum actual fixed-50%-mixed validation macro-F1, then clean macro-F1; earliest exact tie",
              "training_attack": "Untargeted CE-ascent, final iterate, one start; random start applies to PGD only",
              "validation_attack": settings["validation_attack"],
              "augmentation_scope": "Training only, before attacks; validation always uses original images",
              "checkpoint_epoch_semantics": "Completed 1-based epoch; optimizer and scheduler already stepped"}
    for epoch in range(start_epoch, stop):
        start = perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        torch.cuda.reset_peak_memory_stats(device)
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, settings)
        validation = validate(model, val_loader, device, settings["validation_attack"], config["seed"])
        scheduler.step()
        entry = {"epoch": epoch + 1, "learning_rate": learning_rate, "train": train_metrics,
                 "validation": validation, "seconds": perf_counter() - start,
                 "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20}
        history.append(entry)
        score = (validation["mixed_50pct"]["macro_f1"], validation["clean"]["macro_f1"])
        improved = score > (best["mixed_macro_f1"], best["clean_macro_f1"])
        if improved:
            best = {"mixed_macro_f1": score[0], "clean_macro_f1": score[1], "epoch": epoch + 1}
        save_checkpoint(out / "last.pt", model, optimizer, scheduler, epoch + 1, generator,
                        config, settings, metadata, history, best)
        if improved:
            save_checkpoint(out / "best.pt", model, optimizer, scheduler, epoch + 1, generator,
                            config, settings, metadata, history, best)
        report.update(history=history, best_selection=best, completed_epochs=epoch + 1)
        write_json(out / "metrics.json", report)
        print(f"epoch {epoch + 1}/{settings['epochs']}: loss={train_metrics['loss']:.6f} "
              f"clean_F1={score[1]:.6f} mixed_F1={score[0]:.6f} {entry['seconds']:.2f}s", flush=True)
    report.update(history=history, best_selection=best, completed_epochs=stop)
    write_json(out / "metrics.json", report)


if __name__ == "__main__":
    main()
