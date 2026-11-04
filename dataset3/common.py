"""Shared I/O, reproducibility and model setup used by the experiment entry points."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def load_records(path, labeled):
    """Validate CSV input once at the boundary, not in __getitem__ or batches."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = {"id", "label"} if labeled else {"id"}
        if not fields.issubset(reader.fieldnames):
            raise ValueError(f"Missing CSV fields in {path}: {fields}")
        rows = list(reader)
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"Empty CSV or duplicate IDs: {path}")
    if labeled and any(not 0 <= int(r["label"]) < 10 for r in rows):
        raise ValueError(f"Labels must be in 0..9: {path}")
    return rows


class ImageRecords(Dataset):
    """Cache the small supplied image set as uint8 once for repeated attack sweeps."""
    def __init__(self, records, image_dir, labeled):
        self.ids = [r["id"] for r in records]
        self.labels = np.array([int(r["label"]) for r in records]) if labeled else np.full(len(records), -1)
        images = []
        for image_id in self.ids:
            with Image.open(Path(image_dir) / f"{image_id}.png") as image:
                images.append(np.asarray(image).copy())
        self.images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.images[index].float().div(255), int(self.labels[index])


def setup(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config["batch_size"] < 1 or config["num_workers"] < 0 or config["num_classes"] != 10:
        raise ValueError("Expected positive batch_size, nonnegative workers and 10 classes")
    if not torch.cuda.is_available():
        raise RuntimeError("These measured experiment entry points require a CUDA GPU")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    return config, torch.device("cuda")


def load_provided_model(data_dir, device):
    spec = importlib.util.spec_from_file_location("provided_dataset3", Path(data_dir) / "dataset3_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.ProvidedNet().to(device)
    model.load_state_dict(torch.load(Path(data_dir) / "provided_model.pt", map_location=device, weights_only=True))
    model.eval()
    # Input gradients are still available; evaluation does not need parameter gradients.
    model.requires_grad_(False)
    return model


def provenance(config):
    paths = [Path(config["data_dir"]) / "provided_model.pt",
             Path(config["split_dir"]) / "validation.csv"]
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "config": config,
            "sha256": {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "scope": "Local diagnostic on provided-model training data; not unseen-data or Kaggle performance"}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_predictions(path, ids, predictions):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "label"])
        writer.writerows(zip(ids, map(int, predictions)))
