"""Load the supplied weights and test one GPU forward/backward pass, without training."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import platform
import random

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
from PIL import Image
import PIL
import torch


def main():
    config = json.loads(Path("configs/experiment.json").read_text())
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this project environment check")
    device = torch.device("cuda")
    data = Path(config["data_dir"])
    spec = importlib.util.spec_from_file_location("provided_dataset3", data / "dataset3_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.ProvidedNet().to(device)
    model.load_state_dict(torch.load(data / "provided_model.pt", map_location=device, weights_only=True))
    model.eval()
    with Path(config["split_dir"], "train.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))[:config["batch_size"]]
    images = []
    for row in rows:
        with Image.open(data / "train_images" / f'{row["id"]}.png') as image:
            images.append(np.asarray(image).copy())
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().to(device) / 255
    x.requires_grad_(True)
    y = torch.tensor([int(r["label"]) for r in rows], device=device)
    torch.cuda.reset_peak_memory_stats()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    torch.cuda.synchronize()
    assert logits.shape == (len(rows), config["num_classes"])
    assert torch.isfinite(logits).all() and torch.isfinite(x.grad).all()
    assert x.grad.abs().max() > 0
    # No optimizer step: the supplied checkpoint remains unchanged.
    report = {
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "pillow": PIL.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "batch_size_checked": len(rows), "input_shape": list(x.shape),
        "input_range": [x.min().item(), x.max().item()], "logit_shape": list(logits.shape),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
        "finite_input_gradients": True, "nonzero_input_gradients": True,
        "deterministic_algorithms": True, "seed": config["seed"],
        "normalization": "Performed inside ProvidedNet; inputs divided by 255 only",
        "scope": "Single-batch forward/backward smoke check; no optimizer step, no accuracy evaluation"
    }
    Path("docs/environment_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
