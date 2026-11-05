"""Training contracts: CE ascent, train/eval separation and exact epoch resume."""
import copy
import importlib
import random
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LogisticExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.9, -1.1]))

    def forward(self, images):
        z = images.flatten(1) @ self.weight + 0.1
        return torch.stack([torch.zeros_like(z), z], dim=1)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout(0.25)
        self.linear = nn.Linear(4, 2)
        self.forward_modes = []

    def forward(self, images):
        self.forward_modes.append(self.training)
        return self.linear(self.dropout(self.bn(images).flatten(1)))


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Missing training functionality produces an ordinary, descriptive failure.
        try:
            cls.training = importlib.import_module("dataset3.training")
        except ModuleNotFoundError as error:
            if error.name != "dataset3.training":
                raise
            cls.training = None

    def setUp(self):
        self.assertIsNotNone(self.training, "dataset3.training must implement Step 4 training")
        torch.set_num_threads(1)
        torch.manual_seed(5577)
        np.random.seed(5577)
        random.seed(5577)
        self.settings = {"mode": "pgd", "epochs": 3, "batch_size": 4,
                         "learning_rate": 0.02, "momentum": 0.9, "weight_decay": 0.001,
                         "epsilon": 8 / 255, "step_size": 2 / 255, "steps": 2,
                         "random_start": True, "clean_weight": 0.5, "augmentation": False,
                         "initialization": "provided", "scheduler": "cosine",
                         "validation_attack": {"epsilon": 8 / 255, "step_size": 2 / 255,
                                               "steps": 10, "restarts": 1}}

    def loader(self, generator):
        x = torch.arange(32, dtype=torch.float32).reshape(8, 1, 2, 2) / 32
        y = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1])
        return DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True, generator=generator)

    def optimizer(self, model):
        return torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=0.001)

    def test_pgd_ce_ascent_matches_analytic_gradient_and_constraints(self):
        model = LogisticExample().train()
        images = torch.tensor([[[[0.8, 0.3]]], [[[0., 1.]]]])
        labels = torch.tensor([1, 1])
        actual = self.training.training_pgd(model, images, labels, 8 / 255, 2 / 255, 3,
                                            random_start=False)
        expected = torch.tensor([[[[0.8 - 6 / 255, 0.3 + 6 / 255]]], [[[0., 1.]]]])
        torch.testing.assert_close(actual, expected)
        self.assertLessEqual((actual - images).abs().max().item(), 8 / 255 + 1e-6)
        self.assertTrue(model.training)
        self.assertIsNone(model.weight.grad)
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(images, torch.tensor([[[[0.8, 0.3]]], [[[0., 1.]]]]))
        random_attack = self.training.training_pgd(model, images, labels, 8 / 255, 0.2, 4)
        self.assertLessEqual((random_attack - images).abs().max().item(), 8 / 255 + 1e-6)
        self.assertGreaterEqual(random_attack.min().item(), 0)
        self.assertLessEqual(random_attack.max().item(), 1)

    def test_training_pgd_returns_final_iterate_even_if_earlier_ce_was_higher(self):
        class CurvedLogits(nn.Module):
            def forward(self, x):
                z = 1 - 100 * (x.flatten(1)[:, 0] - 0.5).square()
                return torch.stack([torch.zeros_like(z), z], dim=1)

        x = torch.tensor([[[[0.485]]]])
        result = self.training.training_pgd(CurvedLogits(), x, torch.tensor([0]),
                                            8 / 255, 0.02, 2, random_start=False)
        torch.testing.assert_close(result, x, rtol=0, atol=1e-7)

    def test_initialization_loads_provided_weights_and_unfreezes_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "dataset3_model.py").write_text(
                "from torch import nn\nclass ProvidedNet(nn.Linear):\n"
                "    def __init__(self):\n        super().__init__(4, 2)\n", encoding="utf-8")
            expected = {"weight": torch.full((2, 4), 0.75), "bias": torch.full((2,), -0.25)}
            torch.save(expected, data / "provided_model.pt")
            supplied = self.training.initialize_model({"data_dir": str(data)}, self.settings,
                                                       torch.device("cpu"))
            torch.testing.assert_close(supplied.weight, expected["weight"])
            self.assertTrue(all(p.requires_grad for p in supplied.parameters()))
            fresh = self.training.initialize_model({"data_dir": str(data)},
                                                    dict(self.settings, initialization="random"),
                                                    torch.device("cpu"))
            self.assertFalse(torch.equal(fresh.weight, expected["weight"]))
            self.assertTrue(all(p.requires_grad for p in fresh.parameters()))

    def test_optional_training_augmentation_is_bounded_and_seeded(self):
        images = torch.arange(2 * 3 * 8 * 8).float().reshape(2, 3, 8, 8) / (2 * 3 * 8 * 8)
        first = self.training.augment_training(images)
        torch.manual_seed(5577)
        second = self.training.augment_training(images)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(first.shape, images.shape)
        self.assertFalse(torch.equal(first, images))
        self.assertGreaterEqual(first.min().item(), 0)
        self.assertLessEqual(first.max().item(), 1)

    def test_training_configuration_rejects_out_of_budget_attacks(self):
        for changes in [{"epsilon": 9 / 255}, {"mode": "unknown"},
                        {"clean_weight": 1.1}, {"steps": 0}]:
            with self.assertRaises(ValueError):
                self.training.validate_training_config(dict(self.settings, **changes))
        invalid = copy.deepcopy(self.settings)
        invalid["validation_attack"]["epsilon"] = 9 / 255
        with self.assertRaises(ValueError):
            self.training.validate_training_config(invalid)

    def test_outer_training_updates_weights_without_attack_batchnorm_contamination(self):
        model = TinyNet()
        before = model.linear.weight.detach().clone()
        result = self.training.train_one_epoch(model, self.loader(torch.Generator().manual_seed(1)),
                                               self.optimizer(model), torch.device("cpu"), self.settings)
        self.assertFalse(torch.equal(before, model.linear.weight))
        self.assertEqual(result["records"], 8)
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertEqual(model.bn.num_batches_tracked.item(), 4)
        self.assertTrue(model.training)
        self.assertIn(False, model.forward_modes)
        self.assertEqual(sum(model.forward_modes), 4)

    def test_clean_training_uses_clean_ce_and_fgsm_preserves_outer_mode(self):
        for mode in ["clean", "fgsm"]:
            model = TinyNet()
            before = model.linear.weight.detach().clone()
            settings = dict(self.settings, mode=mode)
            self.training.train_one_epoch(model, self.loader(torch.Generator().manual_seed(1)),
                                           self.optimizer(model), torch.device("cpu"), settings)
            self.assertFalse(torch.equal(before, model.linear.weight))
            self.assertEqual(model.bn.num_batches_tracked.item(), 2 if mode == "clean" else 4)
            self.assertTrue(model.training)

    def test_checkpoint_resume_matches_uninterrupted_epoch_with_rng_and_optimizer(self):
        model = TinyNet()
        generator = torch.Generator().manual_seed(5577)
        loader = self.loader(generator)
        optimizer = self.optimizer(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
        self.training.train_one_epoch(model, loader, optimizer, torch.device("cpu"), self.settings)
        scheduler.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            self.training.save_checkpoint(path, model, optimizer, scheduler, 1, generator,
                                           {"seed": 5577}, self.settings, {"scope": "test"},
                                           [{"epoch": 1}],
                                           {"mixed_macro_f1": 0.3, "clean_macro_f1": 0.4, "epoch": 1})
            expected_draws = (random.random(), np.random.random(), torch.rand(3))
            self.training.train_one_epoch(model, loader, optimizer, torch.device("cpu"), self.settings)
            scheduler.step()
            expected_state = copy.deepcopy(model.state_dict())
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            expected_loader = generator.get_state().clone()

            resumed = TinyNet()
            resumed_optimizer = self.optimizer(resumed)
            resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(resumed_optimizer, T_max=3)
            resumed_generator = torch.Generator().manual_seed(123)
            checkpoint = self.training.load_checkpoint(path, resumed, resumed_optimizer,
                                                        resumed_scheduler, resumed_generator,
                                                        torch.device("cpu"))
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(random.random(), expected_draws[0])
            self.assertEqual(np.random.random(), expected_draws[1])
            torch.testing.assert_close(torch.rand(3), expected_draws[2], rtol=0, atol=0)
            self.training.train_one_epoch(resumed, self.loader(resumed_generator), resumed_optimizer,
                                           torch.device("cpu"), self.settings)
            resumed_scheduler.step()
            for name, expected in expected_state.items():
                torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)
            torch.testing.assert_close(resumed_generator.get_state(), expected_loader, rtol=0, atol=0)
            self.assertEqual(resumed_scheduler.state_dict(), scheduler.state_dict())
            actual_optimizer = resumed_optimizer.state_dict()
            self.assertEqual(actual_optimizer["param_groups"], expected_optimizer["param_groups"])
            for index, state in expected_optimizer["state"].items():
                for name, expected in state.items():
                    torch.testing.assert_close(actual_optimizer["state"][index][name], expected, rtol=0, atol=0)

    def test_full_validation_uses_actual_fixed_half_mixed_predictions(self):
        # Misclassifying a different subset each call would invalidate selection.
        class BoundaryModel(nn.Module):
            def forward(self, images):
                z = 10 * (images.flatten(1)[:, 0] - 0.5)
                return torch.stack([torch.zeros_like(z), z], dim=1)

        model = BoundaryModel().train()
        x = torch.tensor([[[[0.51]]], [[[0.1]]], [[[0.9]]], [[[0.49]]]])
        y = torch.tensor([1, 0, 1, 0])
        loader = DataLoader(TensorDataset(x, y), batch_size=2)
        attack = {"epsilon": 8 / 255, "step_size": 2 / 255, "steps": 10, "restarts": 1}
        first = self.training.validate(model, loader, torch.device("cpu"), attack, 5577, 2)
        second = self.training.validate(model, loader, torch.device("cpu"), attack, 5577, 2)
        self.assertEqual(first, second)
        self.assertEqual(first["clean"]["records"], 4)
        self.assertEqual(first["adversarial"]["records"], 4)
        self.assertEqual(first["mixed_50pct"]["records"], 4)
        self.assertEqual(first["mixed_attacked_records"], 2)
        self.assertEqual(first["clean"]["macro_f1"], 1)
        self.assertEqual(first["adversarial"]["macro_f1"], 0.5)
        # Seed 5578 attacks positions 3 and 2: predictions [1,0,1,1].
        self.assertAlmostEqual(first["mixed_50pct"]["macro_f1"], 11 / 15)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
