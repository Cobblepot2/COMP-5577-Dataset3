import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import TensorDataset


def source_fixture(directory):
    path = Path(directory)
    (path / "dataset3_model.py").write_text(
        "import torch\nfrom torch import nn\n"
        "class ProvidedNet(nn.Module):\n"
        " def __init__(self):\n"
        "  super().__init__(); self.weight=nn.Parameter(torch.zeros(10,1)); self.bias=nn.Parameter(torch.zeros(10))\n"
        " def forward(self,x): return x.flatten(1).mean(1,keepdim=True) @ self.weight.T + self.bias\n",
        encoding="utf-8")
    weight = torch.zeros(10, 1)
    weight[1] = 1
    bias = torch.full((10,), -10.)
    bias[0], bias[1] = 0, -.5
    torch.save({"weight": weight, "bias": bias}, path / "provided_model.pt")
    weight[1] = -1
    bias[1] = .5
    checkpoint = path / "random50.pt"
    torch.save({"model_state_dict": {"weight": weight, "bias": bias}, "epoch": 50,
                "training_config": {"initialization": "random"},
                "config": {"num_classes": 10, "input_range": [0., 1.]}}, checkpoint)
    return {"data_dir": str(path), "num_classes": 10, "input_range": [0., 1.],
            "seed": 5577, "batch_size": 2, "num_workers": 0}, checkpoint


class TransferSourceTests(unittest.TestCase):
    def test_default_source_keeps_provided_weights_and_existing_cache_identity(self):
        from dataset3.evaluate_defense import load_transfer_source
        from dataset3.common import load_provided_model
        with tempfile.TemporaryDirectory() as directory:
            config, _ = source_fixture(directory)
            source, metadata, identity = load_transfer_source(config, None, torch.device("cpu"))
            original = load_provided_model(directory, torch.device("cpu"))
            x = torch.tensor([[[[128/255]]], [[[127/255]]]])
            torch.testing.assert_close(source(x), original(x), rtol=0, atol=0)
            self.assertEqual(set(identity), {"provided_model_sha256", "source_architecture_sha256"})
            self.assertEqual(identity["provided_model_sha256"], hashlib.sha256(
                (Path(directory) / "provided_model.pt").read_bytes()).hexdigest())
            self.assertEqual(metadata["source_kind"], "provided")
            self.assertEqual(metadata["epoch"], 0)
            self.assertFalse(source.training)
            self.assertFalse(any(p.requires_grad for p in source.parameters()))

    def test_custom_source_changes_real_attacks_and_uses_its_own_cache(self):
        from dataset3.evaluate_defense import load_transfer_source, load_defense, evaluate_attack
        with tempfile.TemporaryDirectory() as directory:
            config, checkpoint = source_fixture(directory)
            provided, _, provided_identity = load_transfer_source(config, None, torch.device("cpu"))
            custom, metadata, custom_identity = load_transfer_source(config, checkpoint, torch.device("cpu"))
            target, _ = load_defense(config, None, torch.device("cpu"))
            dataset = TensorDataset(torch.tensor([[[[128/255]]], [[[127/255]]]]),
                                    torch.tensor([1, 0]))
            settings = {"name": "transfer_fgsm8", "method": "fgsm", "epsilon": 8/255,
                        "threat_model": "transfer"}
            kwargs = dict(model=target, dataset=dataset, config=config, device=torch.device("cpu"),
                          settings=settings, clean=np.array([1, 0]), mixed_mask=np.array([True, False]),
                          cache_dir=Path(directory) / "cache")
            first, a, _ = evaluate_attack(source=provided, cache_identity=provided_identity, **kwargs)
            second, b, _ = evaluate_attack(source=custom, cache_identity=custom_identity, **kwargs)
            reused, repeat, _ = evaluate_attack(source=provided, cache_identity=provided_identity, **kwargs)
            np.testing.assert_array_equal(a, [0, 1])
            np.testing.assert_array_equal(b, [1, 0])
            np.testing.assert_array_equal(repeat, a)
            self.assertNotEqual(first["cache"]["path"], second["cache"]["path"])
            self.assertTrue(reused["cache"]["reused"])
            self.assertNotIn("provided_model_sha256", custom_identity)
            self.assertEqual(custom_identity["source_checkpoint_sha256"],
                             hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(metadata["checkpoint"], str(checkpoint))
            self.assertEqual(metadata["checkpoint_sha256"], custom_identity["source_checkpoint_sha256"])
            self.assertEqual(metadata["initialization"], "random")
            self.assertEqual(metadata["epoch"], 50)
            self.assertFalse(custom.training)
            self.assertFalse(any(p.requires_grad for p in custom.parameters()))
            self.assertTrue(all(p.grad is None for p in custom.parameters()))


if __name__ == "__main__":
    unittest.main()
