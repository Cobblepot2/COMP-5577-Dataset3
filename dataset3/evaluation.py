"""Shared timed clean inference. Loading/decoding and warm-up are excluded."""
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader


def make_loader(dataset, config):
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                      num_workers=config["num_workers"], pin_memory=True)


def predict_clean(model, loader, device):
    warmup, _ = next(iter(loader))
    with torch.inference_mode():
        for _ in range(2):
            model(warmup.to(device))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    outputs = []
    with torch.inference_mode():
        for images, _ in loader:
            logits = model(images.to(device, non_blocking=True))
            outputs.append(logits.cpu().numpy())
    torch.cuda.synchronize()
    seconds = perf_counter() - start
    logits = np.concatenate(outputs)
    return logits.argmax(1), logits, {
        "seconds": seconds, "images_per_second": len(logits) / seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "scope": "Two warm-up batches excluded; includes loader iteration, normalization to [0,1], host/device copies and prediction; excludes initial image decode and metric/file writing"
    }
