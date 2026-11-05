"""Deterministic inference transforms with gradients through the full predictor."""
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class HorizontalFlipAverage(nn.Module):
    """Return log mean probabilities over identity and horizontal-flip views.

    Log-space averaging avoids underflow while preserving both input-gradient
    branches. These normalized log probabilities can be passed to CE attacks.
    Model weights and their requires_grad flags are shared with the base model.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        identity = F.log_softmax(self.model(images), dim=1)
        flipped = F.log_softmax(self.model(images.flip(-1)), dim=1)
        return torch.logaddexp(identity, flipped) - math.log(2.)


def configure_predictor(model, horizontal_flip=False):
    """Leave the original predictor untouched unless the transform is requested."""
    if not horizontal_flip:
        return model, {"name": "identity", "views": ["identity"],
                       "aggregation": "original_model_logits"}
    return HorizontalFlipAverage(model).eval(), {
        "name": "identity_horizontal_flip_probability_average",
        "views": ["identity", "horizontal_flip"],
        "aggregation": "arithmetic_mean_softmax_probabilities",
        "attack_output": "log_mean_probabilities",
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
