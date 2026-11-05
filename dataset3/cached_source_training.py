"""Train a random SmallResNet18 on clean images and cached supplied-source attacks."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .attacks import fgsm, pgd
from .common import ImageRecords, load_provided_model, provenance, setup, write_json
from .improve_training import (evaluate_source_conditions, load_training_protocol,
                               prepare_source_cache, selection_score)
from .metrics import classification_metrics
from .training import augment_training, load_checkpoint, save_checkpoint


class CachedPairs(Dataset):
    def __init__(self, records, attacks):
        self.records, self.attacks = records, attacks

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return (self.records.images[index].float().div(255), self.attacks["fgsm"][index],
                self.attacks["pgd"][index], int(self.records.labels[index]))


def augment_pair(clean, adversarial):
    paired = augment_training(torch.cat([clean, adversarial], dim=1))
    return paired[:, :3], paired[:, 3:]


def train_one_epoch(model, loader, optimizer, device, settings):
    model.train()
    sums = {"loss": 0.0, "clean_ce": 0.0, "adversarial_ce": 0.0}
    records, fgsm_records = 0, 0
    for clean, fgsm_images, pgd_images, labels in loader:
        clean, fgsm_images, pgd_images, labels = (value.to(device, non_blocking=True)
                                                 for value in (clean, fgsm_images, pgd_images, labels))
        choose_fgsm = torch.rand(len(clean), device=device) < settings["fgsm_probability"]
        adversarial = torch.where(choose_fgsm[:, None, None, None], fgsm_images, pgd_images)
        if settings["augmentation"]:
            clean, adversarial = augment_pair(clean, adversarial)
        optimizer.zero_grad(set_to_none=True)
        logits = model(torch.cat([clean, adversarial], dim=0))
        clean_logits, adversarial_logits = logits.split(len(clean))
        clean_ce = F.cross_entropy(clean_logits, labels)
        adversarial_ce = F.cross_entropy(adversarial_logits, labels)
        weight = settings["clean_weight"]
        loss = weight * clean_ce + (1 - weight) * adversarial_ce
        loss.backward()
        optimizer.step()
        for name, value in [("loss", loss), ("clean_ce", clean_ce), ("adversarial_ce", adversarial_ce)]:
            sums[name] += value.detach().item() * len(clean)
        records += len(clean)
        fgsm_records += int(choose_fgsm.sum())
    return {"records": records, "fgsm_records": fgsm_records, "pgd_records": records - fgsm_records,
            "joint_bn_protocol": "One concatenated clean/adversarial forward; equal domain counts",
            **{name: value / records for name, value in sums.items()}}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_training_attacks(source, dataset, config, device, cache_dir, training_csv):
    """Exactly match the previously measured train-only BN-probe cache identity."""
    root = Path(config["data_dir"])
    common = {"training_csv_sha256": _sha(training_csv),
              "training_pixels_sha256": hashlib.sha256(dataset.images.numpy().tobytes()).hexdigest(),
              "source_checkpoint_sha256": _sha(root / "provided_model.pt"),
              "source_architecture_sha256": _sha(root / "dataset3_model.py"),
              "attack_code_sha256": _sha(Path(__file__).with_name("attacks.py")),
              "seed": config["seed"], "batch_size": config["batch_size"],
              "torch": str(torch.__version__), "records": len(dataset)}
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    attacks, metadata = {}, {}
    clean = dataset.images.float().div(255)
    for name, settings in {"fgsm": {"method": "fgsm", "epsilon": 8 / 255},
                           "pgd": {"method": "pgd", "epsilon": 8 / 255,
                                   "step_size": 2 / 255, "steps": 10, "restarts": 1}}.items():
        identity = dict(common, settings=settings)
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = directory / f"train_attack_{digest}.pt"
        reused = path.exists()
        if reused:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["identity"] != identity:
                raise ValueError("Training attack cache identity mismatch")
            attacked = payload["adversarial"]
        else:
            loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                                num_workers=config["num_workers"],
                                generator=torch.Generator().manual_seed(config["seed"]))
            pieces = []
            for index, (images, labels) in enumerate(loader):
                images, labels = images.to(device), labels.to(device)
                if name == "fgsm":
                    result = fgsm(source, images, labels, settings["epsilon"])
                else:
                    result = pgd(source, images, labels, settings["epsilon"], settings["step_size"],
                                 settings["steps"], settings["restarts"],
                                 generator=torch.Generator(device=device).manual_seed(config["seed"] + index))
                pieces.append(result.cpu())
            attacked = torch.cat(pieces)
        if (attacked.shape != clean.shape or not torch.isfinite(attacked).all()
                or attacked.min() < 0 or attacked.max() > 1
                or (attacked - clean).abs().max() > 8 / 255 + 1e-6):
            raise ValueError("Training cache violates image shape/pixel/epsilon constraints")
        if not reused:
            temporary = path.with_suffix(".tmp")
            torch.save({"identity": identity, "adversarial": attacked}, temporary)
            temporary.replace(path)
        attacks[name] = attacked
        metadata[name] = {"path": str(path), "identity": identity, "reused": reused}
        print(f"training source {name}: {'reused' if reused else 'created'}", flush=True)
    return attacks, metadata


def experiment_scope(final_refit):
    if final_refit:
        return "Full-data final fit with frozen supplied teacher attacks, no held-out metric; config/horizon/stop frozen beforehand"
    return ("Student gradient updates exclude validation IDs, but the supplied attack teacher originally saw "
            "all training IDs including validation; local validation is diagnostic and used for selection")


def evaluate_validation(model, dataset, attacks, config, device, floor):
    predictions = []
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"])
    model.eval()
    with torch.inference_mode():
        for images, _ in loader:
            predictions.append(model(images.to(device)).argmax(1).cpu().numpy())
    clean = classification_metrics(dataset.labels, np.concatenate(predictions))
    transferred = evaluate_source_conditions(model, dataset, attacks, config, device)
    criterion = selection_score({"clean": clean, "mixed_50pct": {"macro_f1": None}}, transferred, floor)
    return {"clean": clean, "source_validation": transferred, "criterion": criterion,
            "own_white_box_evaluated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--training", default="configs/public85_residual_source.json")
    parser.add_argument("--output", default="runs/public85_residual_source")
    parser.add_argument("--train-cache", default="runs/public85_bnprobe")
    parser.add_argument("--transfer-cache", default="runs/improve_validation_cache")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--final-refit", action="store_true")
    args = parser.parse_args()
    settings = json.loads(Path(args.training).read_text(encoding="utf-8"))
    if settings["architecture"] != "small_resnet18" or settings["initialization"] != "random":
        raise ValueError("This controlled experiment uses random SmallResNet18 only")
    for name in ["epochs", "batch_size"]:
        if not isinstance(settings[name], int) or settings[name] < 1:
            raise ValueError("Epochs and batch size must be positive integers")
    for name in ["learning_rate", "momentum", "weight_decay", "clean_weight", "fgsm_probability", "clean_floor"]:
        if not math.isfinite(settings[name]):
            raise ValueError(f"{name} must be finite")
    if (settings["learning_rate"] <= 0 or settings["momentum"] < 0 or settings["weight_decay"] < 0
            or settings["scheduler"] != "cosine"
            or any(not 0 <= settings[key] <= 1 for key in ["clean_weight", "fgsm_probability", "clean_floor"])):
        raise ValueError("Invalid optimizer, mixture or eligibility setting")
    settings["final_refit"] = args.final_refit
    config, device = setup(args.config)
    if config["seed"] != 5577 or config["input_range"] != [0, 1]:
        raise ValueError("Keep fixed seed5577 and [0,1] input pixels")
    protocol = load_training_protocol(config, args.final_refit)
    images = ImageRecords(protocol["training_rows"], Path(config["data_dir"]) / "train_images", True)
    source = load_provided_model(config["data_dir"], device)
    attacks, training_cache = prepare_training_attacks(
        source, images, config, device, args.train_cache, protocol["training_path"])
    from .residual import SmallResNet18
    model = SmallResNet18(num_classes=config["num_classes"]).to(device)
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(CachedPairs(images, attacks), batch_size=settings["batch_size"], shuffle=True,
                        generator=generator, num_workers=config["num_workers"], pin_memory=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=settings["learning_rate"], momentum=settings["momentum"],
                                weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"])
    metadata = provenance(config)
    metadata.update(torch=str(metadata["torch"]), architecture="small_resnet18", architecture_width=model.base_channels,
                    scope=experiment_scope(args.final_refit), final_refit=args.final_refit,
                    training_attack_identities={name: item["identity"] for name, item in training_cache.items()})
    for path in [protocol["training_path"], Path(args.config), Path(args.training), Path(__file__),
                 Path(__file__).with_name("residual.py"), Path(__file__).with_name("attacks.py"),
                 Path(__file__).with_name("training.py"), Path(__file__).with_name("improve_training.py"),
                 Path(config["data_dir"]) / "dataset3_model.py"]:
        metadata["sha256"][str(path)] = _sha(path)
    validation_cache = {}
    if not args.final_refit:
        validation = ImageRecords(protocol["validation_rows"], Path(config["data_dir"]) / "train_images", True)
        validation_attacks, validation_cache = prepare_source_cache(source, validation, config, device, args.transfer_cache)
        metadata["validation_attack_identities"] = {name: item["identity"] for name, item in validation_cache.items()}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    history, start_epoch, best = [], 0, None
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, generator, device)
        if state["config"] != config or state["training_config"] != settings or state["provenance"] != metadata:
            raise ValueError("Exact resume requires identical configs/horizon/protocol/source/data/code hashes")
        history, start_epoch, best = state["history"], state["epoch"], state["best_selection"]
    stop = args.stop_after_epoch if args.stop_after_epoch is not None else settings["epochs"]
    if not 0 < stop <= settings["epochs"] or stop < start_epoch:
        raise ValueError("Stop epoch must be within original horizon and at/after resume epoch")
    report = {"provenance": metadata, "config": config, "training_config": settings,
              "training_records": len(images), "validation_records": len(protocol["validation_rows"]),
              "training_cache": training_cache, "validation_cache": validation_cache,
              "selection_performed": not args.final_refit, "own_white_box_evaluated_each_epoch": False,
              "criterion": "Prefer clean>=.857639 then maximize minimum source-FGSM8/source-PGD20x3 fixed-half mixed F1, clean tie-break",
              "ineligible_policy": "Diagnostic only; retain previous eligible model if this experiment never qualifies"}
    for epoch in range(start_epoch, stop):
        started = perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        torch.cuda.reset_peak_memory_stats(device)
        training = train_one_epoch(model, loader, optimizer, device, settings)
        entry = {"epoch": epoch + 1, "learning_rate": learning_rate, "train": training}
        improved = False
        if not args.final_refit:
            measured = evaluate_validation(model, validation, validation_attacks, config, device, settings["clean_floor"])
            criterion = dict(measured["criterion"], epoch=epoch + 1)
            improved = best is None or criterion["rank"] > best["rank"]
            if improved:
                best = criterion
            entry.update(validation=measured, criterion=criterion)
        scheduler.step()
        entry.update(seconds=perf_counter() - started,
                     peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20)
        history.append(entry)
        save_checkpoint(out / "last.pt", model, optimizer, scheduler, epoch + 1, generator,
                        config, settings, metadata, history, best)
        if improved:
            save_checkpoint(out / "best.pt", model, optimizer, scheduler, epoch + 1, generator,
                            config, settings, metadata, history, best)
        report.update(history=history, best_selection=best, completed_epochs=epoch + 1)
        write_json(out / "metrics.json", report)
        suffix = (f"clean={criterion['clean_macro_f1']:.6f} worst_source_mixed={criterion['worst_source_mixed_macro_f1']:.6f} eligible={criterion['eligible']}"
                  if not args.final_refit else "full-data final refit, no validation")
        print(f"epoch {epoch + 1}/{settings['epochs']}: loss={training['loss']:.6f} {suffix} {entry['seconds']:.2f}s", flush=True)
    report.update(history=history, best_selection=best, completed_epochs=stop)
    write_json(out / "metrics.json", report)


if __name__ == "__main__":
    main()
