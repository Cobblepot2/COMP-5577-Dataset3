import hashlib
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn


def provided_fixture(directory):
    root = Path(directory)
    (root / "dataset3_model.py").write_text(
        "import torch\nfrom torch import nn\n"
        "class ProvidedNet(nn.Module):\n"
        " def __init__(self):\n"
        "  super().__init__(); self.weight=nn.Parameter(torch.arange(30.).view(10,3)/100)\n"
        " def forward(self,x): return x.mean((2,3)) @ self.weight.T\n", encoding="utf-8")
    torch.save({"weight": torch.arange(30.).view(10, 3)/100}, root / "provided_model.pt")
    return {"data_dir": str(root), "num_classes": 10, "image_size": 64, "input_range": [0, 1]}


class ResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_identity_and_projection_shortcuts_reach_output(self):
        from dataset3.residual import BasicBlock
        identity = BasicBlock(3, 3).eval()
        with torch.no_grad():
            identity.conv2.weight.zero_()
        x = torch.linspace(-1, 1, 2 * 3 * 8 * 8).view(2, 3, 8, 8)
        torch.testing.assert_close(identity(x), x.relu(), rtol=0, atol=0)
        projection = BasicBlock(3, 5, stride=2).eval()
        with torch.no_grad():
            projection.conv2.weight.zero_()
            projection.shortcut[0].weight.zero_()
            for channel in range(5):
                projection.shortcut[0].weight[channel, channel % 3, 0, 0] = 1
        expected = (x[:, [0, 1, 2, 0, 1], ::2, ::2] /
                    (1 + projection.shortcut[1].eps) ** .5).relu()
        torch.testing.assert_close(projection(x), expected)
        self.assertEqual(tuple(projection(x).shape), (2, 5, 4, 4))

    def test_network_normalizes_once_and_keeps_expected_stage_shapes(self):
        from dataset3.residual import SmallResNet18
        model = SmallResNet18(num_classes=10).eval().requires_grad_(False)
        observed = {}
        handle = model.stem.register_forward_pre_hook(
            lambda module, args: observed.update(normalized=args[0].detach().clone()))
        shapes = []
        stage_handles = [stage.register_forward_hook(
            lambda module, args, output: shapes.append(tuple(output.shape[1:]))) for stage in model.stages]
        x = torch.tensor([.7438, .4580, .1289]).view(1, 3, 1, 1).expand(1, 3, 64, 64).clone().requires_grad_(True)
        output = model(x)
        handle.remove()
        for stage_handle in stage_handles:
            stage_handle.remove()
        torch.testing.assert_close(observed["normalized"],
                                   torch.tensor([1., 0., -1.]).view(1, 3, 1, 1).expand_as(x), atol=1e-6, rtol=0)
        self.assertEqual(shapes, [(32, 64, 64), (64, 32, 32), (128, 16, 16), (256, 8, 8)])
        self.assertEqual(tuple(output.shape), (1, 10))
        self.assertFalse(any(isinstance(module, nn.Dropout) for module in model.modules()))
        gradient, = torch.autograd.grad(output[:, 0].sum(), x)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().max().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_new_checkpoint_factory_is_strict_and_identifies_actual_source_architecture(self):
        from dataset3.residual import SmallResNet18
        from dataset3.evaluate_defense import load_defense, load_transfer_source
        import dataset3.residual as architecture_module
        model = SmallResNet18()
        with tempfile.TemporaryDirectory() as directory:
            config = provided_fixture(directory)
            path = Path(directory) / "residual.pt"
            state = {"model_state_dict": model.state_dict(), "epoch": 3,
                     "training_config": {"architecture": "small_resnet18", "initialization": "random"},
                     "config": config}
            torch.save(state, path)
            loaded, metadata = load_defense(config, path, torch.device("cpu"))
            self.assertIsInstance(loaded, SmallResNet18)
            self.assertFalse(loaded.training)
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
            expected_hash = hashlib.sha256(Path(architecture_module.__file__).read_bytes()).hexdigest()
            self.assertEqual(metadata["architecture"], "small_resnet18")
            self.assertEqual(metadata["architecture_width"], 32)
            self.assertEqual(metadata["architecture_code_sha256"], expected_hash)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
            source, source_metadata, identity = load_transfer_source(config, path, torch.device("cpu"))
            self.assertIsInstance(source, SmallResNet18)
            self.assertEqual(identity["source_architecture_sha256"], expected_hash)
            self.assertEqual(source_metadata["architecture"], "small_resnet18")
            self.assertNotIn("provided_model_sha256", identity)
            state["training_config"].pop("architecture")
            state["provenance"] = {"architecture": "small_resnet18"}
            torch.save(state, path)
            fallback, _ = load_defense(config, path, torch.device("cpu"))
            self.assertIsInstance(fallback, SmallResNet18)
            state["model_state_dict"]["classifier.weight"] = torch.zeros(9, 256)
            torch.save(state, path)
            with self.assertRaises(RuntimeError):
                load_defense(config, path, torch.device("cpu"))

    def test_legacy_checkpoint_and_provided_baseline_keep_exact_outputs(self):
        from dataset3.common import load_provided_model
        from dataset3.evaluate_defense import load_defense, load_transfer_source
        with tempfile.TemporaryDirectory() as directory:
            config = provided_fixture(directory)
            original = load_provided_model(directory, torch.device("cpu"))
            x = torch.rand(2, 3, 64, 64)
            baseline, metadata = load_defense(config, None, torch.device("cpu"))
            torch.testing.assert_close(baseline(x), original(x), rtol=0, atol=0)
            self.assertEqual(metadata["architecture"], "provided")
            path = Path(directory) / "legacy.pt"
            torch.save({"model_state_dict": original.state_dict(), "epoch": 2}, path)
            legacy, metadata = load_defense(config, path, torch.device("cpu"))
            torch.testing.assert_close(legacy(x), original(x), rtol=0, atol=0)
            expected_hash = hashlib.sha256((Path(directory) / "dataset3_model.py").read_bytes()).hexdigest()
            self.assertEqual(metadata["architecture_code_sha256"], expected_hash)
            _, _, identity = load_transfer_source(config, None, torch.device("cpu"))
            self.assertEqual(set(identity), {"provided_model_sha256", "source_architecture_sha256"})
            self.assertEqual(identity["source_architecture_sha256"], expected_hash)

    def test_deployed_ensemble_loads_both_architectures_and_records_both_identities(self):
        from dataset3.residual import SmallResNet18
        from dataset3.evaluate_defense import load_predictor
        with tempfile.TemporaryDirectory() as directory:
            config = provided_fixture(directory)
            first_path, second_path = Path(directory) / "residual.pt", Path(directory) / "hybrid.pt"
            first = SmallResNet18().eval()
            with torch.no_grad():
                first.classifier.weight.zero_()
                first.classifier.bias.fill_(-100.)
                first.classifier.bias[0], first.classifier.bias[1] = torch.log(torch.tensor(9.)), 0.
            torch.save({"model_state_dict": first.state_dict(), "config": config,
                        "training_config": {"architecture": "small_resnet18", "initialization": "random"}}, first_path)
            second_state = {"model_state_dict": {"weight": torch.zeros(10, 3)}, "config": dict(config),
                            "training_config": {"initialization": "provided"}, "epoch": 37}
            torch.save(second_state, second_path)
            predictor, primary, pipeline = load_predictor(
                config, first_path, torch.device("cpu"), horizontal_flip=True,
                ensemble_checkpoint=second_path, ensemble_weight=.5)
            result = predictor(torch.rand(1, 3, 64, 64)).exp()
            torch.testing.assert_close(result, torch.tensor([[.5, .1] + [.05] * 8]), atol=1e-6, rtol=0)
            self.assertFalse(any(p.requires_grad for p in predictor.parameters()))
            members = pipeline["members"]
            self.assertEqual([m["architecture"] for m in members], ["small_resnet18", "provided"])
            self.assertEqual([m["weight"] for m in members], [.5, .5])
            self.assertEqual(members[1]["epoch"], 37)
            self.assertEqual(members[1]["checkpoint_sha256"], hashlib.sha256(second_path.read_bytes()).hexdigest())
            self.assertEqual(members[0]["checkpoint_sha256"], hashlib.sha256(first_path.read_bytes()).hexdigest())
            self.assertNotEqual(pipeline["pipeline_sha256"], primary["checkpoint_sha256"])
            self.assertEqual(pipeline["inference_transform"]["views"], ["identity", "horizontal_flip"])
            single, _, one_member = load_predictor(config, second_path, torch.device("cpu"))
            torch.testing.assert_close(single(torch.rand(1, 3, 64, 64)), torch.zeros(1, 10), rtol=0, atol=0)
            self.assertEqual(len(one_member["members"]), 1)
            self.assertEqual(one_member["members"][0]["weight"], 1.)
            second_state["config"]["input_range"] = [-1, 1]
            torch.save(second_state, second_path)
            with self.assertRaises(ValueError):
                load_predictor(config, first_path, torch.device("cpu"), ensemble_checkpoint=second_path)


if __name__ == "__main__":
    unittest.main()
