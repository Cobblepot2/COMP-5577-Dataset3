"""Behavioral checks for explicit TRADES/hybrid objectives and run selection."""
import copy
import importlib
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset3.training import load_checkpoint, save_checkpoint
from dataset3.common import ImageRecords, load_records


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(4, 2))

    def forward(self, images):
        return self.head(self.bn(images))


class ImprovementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.improve = importlib.import_module("dataset3.improve_training")
        except ModuleNotFoundError as error:
            if error.name != "dataset3.improve_training":
                raise
            cls.improve = None

    def setUp(self):
        self.assertIsNotNone(self.improve, "Explicit improvement training module is missing")
        torch.set_num_threads(1)
        torch.manual_seed(5577)
        self.settings = {"mode": "trades", "beta": 1.0, "epsilon": 8 / 255,
                         "step_size": 2 / 255, "steps": 3, "augmentation": False,
                         "clean_weight": 0.75}

    def loader(self, generator):
        x = torch.arange(36, dtype=torch.float32).reshape(9, 1, 2, 2) / 36
        y = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1, 0])
        return DataLoader(TensorDataset(x, y), batch_size=5, shuffle=True, generator=generator)

    def test_kl_direction_batchmean_and_both_outer_gradient_branches(self):
        p = torch.tensor([[0.75, 0.25], [0.2, 0.8]], dtype=torch.float64)
        q = torch.tensor([[0.5, 0.5], [0.4, 0.6]], dtype=torch.float64)
        clean, adv = p.log().requires_grad_(), q.log().requires_grad_()
        kl = self.improve.clean_to_adversarial_kl(clean, adv)
        expected = (0.75 * math.log(1.5) + 0.25 * math.log(0.5)
                    + 0.2 * math.log(0.5) + 0.8 * math.log(4 / 3)) / 2
        self.assertAlmostEqual(kl.item(), expected, places=12)
        dc, da = torch.autograd.grad(kl, (clean, adv))
        self.assertGreater(dc.abs().sum().item(), 0)
        torch.testing.assert_close(da, (q - p) / 2, rtol=0, atol=1e-15)
        expected_dc = p * ((p / q).log() - (p * (p / q).log()).sum(1, keepdim=True)) / 2
        torch.testing.assert_close(dc, expected_dc, rtol=0, atol=1e-15)
        total, ce, outer_kl = self.improve.trades_objective(clean, adv, torch.tensor([0, 1]), 3)
        expected_ce = -(math.log(0.75) + math.log(0.8)) / 2
        self.assertAlmostEqual(total.item(), expected_ce + 3 * expected, places=12)
        self.assertAlmostEqual(ce.item(), expected_ce, places=12)
        self.assertAlmostEqual(outer_kl.item(), expected, places=12)

    def test_kl_inner_attack_is_seeded_bounded_ascent_without_bn_updates(self):
        model = TinyNet().train()
        x = torch.tensor([[[[0., 0.3], [0.8, 1.]]], [[[0.1, 0.4], [0.6, 0.9]]]])
        original_state = copy.deepcopy(model.state_dict())
        initial = self.improve.trades_adversary(model, x, 8 / 255, 2 / 255, 0,
                                                 generator=torch.Generator().manual_seed(4))
        final = self.improve.trades_adversary(model, x, 8 / 255, 2 / 255, 3,
                                               generator=torch.Generator().manual_seed(4))
        again = self.improve.trades_adversary(model, x, 8 / 255, 2 / 255, 3,
                                               generator=torch.Generator().manual_seed(4))
        torch.testing.assert_close(final, again, rtol=0, atol=0)
        self.assertLessEqual((final - x).abs().max().item(), 8 / 255 + 1e-6)
        self.assertGreaterEqual(final.min().item(), 0)
        self.assertLessEqual(final.max().item(), 1)
        self.assertFalse(final.requires_grad)
        self.assertTrue(model.training)
        for name, expected in original_state.items():
            torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        model.eval()
        with torch.no_grad():
            reference = model(x)
            before = self.improve.clean_to_adversarial_kl(reference, model(initial))
            after = self.improve.clean_to_adversarial_kl(reference, model(final))
        self.assertGreater(after.item(), before.item())

    def test_hybrid_uses_self_then_frozen_source_for_odd_batch(self):
        class LinearSign(nn.Module):
            def __init__(self, sign):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(float(sign)))

            def forward(self, x):
                z = self.weight * x.flatten(1)[:, 0]
                return torch.stack([torch.zeros_like(z), z], 1)

        own, source = LinearSign(1).train(), LinearSign(-1).eval().requires_grad_(False)
        images, labels = torch.full((5, 1, 1, 1), 0.5), torch.ones(5, dtype=torch.long)
        # Large step projects to the analytic bound regardless of random start.
        result = self.improve.hybrid_adversary(own, source, images, labels, 8 / 255, 1, 1)
        expected = torch.tensor([0.5 - 8 / 255] * 2 + [0.5 + 8 / 255] * 3).reshape_as(images)
        torch.testing.assert_close(result, expected)
        self.assertTrue(own.training)
        self.assertFalse(source.training)
        self.assertFalse(source.weight.requires_grad)
        self.assertIsNone(source.weight.grad)
        self.assertEqual(source.weight.item(), -1)

    def test_actual_parameter_updates_and_source_immutability(self):
        for mode in ["trades", "hybrid"]:
            model = TinyNet()
            source = TinyNet().eval().requires_grad_(False)
            source_before = copy.deepcopy(source.state_dict())
            before = model.head[-1].weight.detach().clone()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
            result = self.improve.train_one_epoch(model, self.loader(torch.Generator().manual_seed(1)),
                                                   optimizer, torch.device("cpu"),
                                                   dict(self.settings, mode=mode), source)
            self.assertFalse(torch.equal(before, model.head[-1].weight))
            self.assertEqual(model.bn.num_batches_tracked.item(), 4)
            self.assertTrue(model.training)
            self.assertEqual(result["records"], 9)
            self.assertTrue(np.isfinite(result["loss"]))
            if mode == "hybrid":
                self.assertEqual(result["self_attack_records"], 4)
                self.assertEqual(result["source_attack_records"], 5)
            for name, expected in source_before.items():
                torch.testing.assert_close(source.state_dict()[name], expected, rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in source.parameters()))

    def test_selection_prefers_clean_eligibility_then_worst_source_then_clean(self):
        def validation(clean, fgsm, pgd):
            own = {"clean": {"macro_f1": clean}, "mixed_50pct": {"macro_f1": 0.2}}
            source = {"fgsm8": {"mixed_50pct_metrics": {"macro_f1": fgsm}},
                      "pgd20x3": {"mixed_50pct_metrics": {"macro_f1": pgd}}}
            return self.improve.selection_score(own, source, 0.857639)

        eligible = validation(0.86, 0.8, 0.79)
        ineligible = validation(0.85, 0.99, 0.99)
        better_worst = validation(0.859, 0.81, 0.8)
        clean_tie = validation(0.87, 0.8, 0.79)
        self.assertTrue(eligible["eligible"])
        self.assertFalse(ineligible["eligible"])
        self.assertGreater(eligible["rank"], ineligible["rank"])
        self.assertGreater(better_worst["rank"], eligible["rank"])
        self.assertGreater(clean_tie["rank"], eligible["rank"])
        self.assertEqual(eligible["worst_source_mixed_macro_f1"], 0.79)

    def test_source_mixed_metrics_are_recomputed_from_predictions(self):
        result = self.improve.source_prediction_metrics(np.array([1, 0, 1, 0]),
                                                         np.array([1, 0, 1, 0]),
                                                         np.array([0, 0, 1, 1]), 5577, 2)
        self.assertEqual(result["float_metrics"]["macro_f1"], 0.5)
        self.assertAlmostEqual(result["mixed_50pct_metrics"]["macro_f1"], 11 / 15)

    def test_final_refit_uses_full_data_without_validation_and_normal_split_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, split = root / "data", root / "splits"
            data.mkdir()
            split.mkdir()
            (data / "train.csv").write_text("id,label\n1,0\n2,1\n3,0\n", encoding="utf-8")
            (split / "train.csv").write_text("id,label\n1,0\n2,1\n", encoding="utf-8")
            (split / "validation.csv").write_text("id,label\n3,0\n", encoding="utf-8")
            config = {"data_dir": str(data), "split_dir": str(split)}
            normal = self.improve.load_training_protocol(config, False)
            refit = self.improve.load_training_protocol(config, True)
            self.assertEqual(len(normal["training_rows"]), 2)
            self.assertEqual(len(normal["validation_rows"]), 1)
            self.assertEqual(len(refit["training_rows"]), 3)
            self.assertEqual(refit["validation_rows"], [])
            self.assertIsNone(refit["validation_path"])
            self.assertIn("no held-out metric", refit["scope"])
            (split / "validation.csv").write_text("id,label\n2,1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.improve.load_training_protocol(config, False)

    def test_source_cache_reuses_exact_identity_and_changes_for_new_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, split, cache = root / "data", root / "split", root / "cache"
            data.mkdir()
            split.mkdir()
            image_dir = data / "train_images"
            image_dir.mkdir()
            for index in range(2):
                Image.fromarray(np.full((2, 2, 3), 64 + 64 * index, dtype=np.uint8)).save(image_dir / f"{index}.png")
            (split / "validation.csv").write_text("id,label\n0,0\n1,1\n", encoding="utf-8")
            (data / "dataset3_model.py").write_text("# test architecture identity\n", encoding="utf-8")
            source = nn.Sequential(nn.Flatten(), nn.Linear(12, 2)).eval().requires_grad_(False)
            torch.save(source.state_dict(), data / "provided_model.pt")
            dataset = ImageRecords(load_records(split / "validation.csv", True), image_dir, True)
            config = {"data_dir": str(data), "split_dir": str(split), "seed": 5577,
                      "batch_size": 2, "num_workers": 0}
            first, first_meta = self.improve.prepare_source_cache(source, dataset, config, "cpu", cache)
            second, second_meta = self.improve.prepare_source_cache(source, dataset, config, "cpu", cache)
            self.assertTrue(all(value["reused"] for value in second_meta.values()))
            for name, values in first.items():
                torch.testing.assert_close(values, second[name], rtol=0, atol=0)
                self.assertLessEqual((values - dataset.images.float() / 255).abs().max().item(), 8 / 255 + 1e-6)
            with torch.no_grad():
                source[1].bias.add_(0.1)
            torch.save(source.state_dict(), data / "provided_model.pt")
            _, third_meta = self.improve.prepare_source_cache(source, dataset, config, "cpu", cache)
            self.assertTrue(all(not value["reused"] for value in third_meta.values()))
            self.assertNotEqual(first_meta["fgsm8"]["identity"], third_meta["fgsm8"]["identity"])

    def test_improvement_epoch_resume_matches_continuation(self):
        for mode in ["trades", "hybrid"]:
            model, source = TinyNet(), TinyNet().eval().requires_grad_(False)
            generator = torch.Generator().manual_seed(5577)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3)
            settings = dict(self.settings, mode=mode)
            self.improve.train_one_epoch(model, self.loader(generator), optimizer, "cpu", settings, source)
            scheduler.step()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "last.pt"
                save_checkpoint(path, model, optimizer, scheduler, 1, generator,
                                {"seed": 5577}, settings, {}, [], {})
                self.improve.train_one_epoch(model, self.loader(generator), optimizer, "cpu", settings, source)
                scheduler.step()
                expected = copy.deepcopy(model.state_dict())
                expected_rng = torch.get_rng_state().clone()
                expected_loader = generator.get_state().clone()
                fresh = TinyNet()
                fresh_optimizer = torch.optim.SGD(fresh.parameters(), lr=0.02, momentum=0.9)
                fresh_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fresh_optimizer, 3)
                fresh_generator = torch.Generator()
                load_checkpoint(path, fresh, fresh_optimizer, fresh_scheduler, fresh_generator)
                self.improve.train_one_epoch(fresh, self.loader(fresh_generator), fresh_optimizer,
                                               "cpu", settings, source)
                fresh_scheduler.step()
                for name, value in expected.items():
                    torch.testing.assert_close(fresh.state_dict()[name], value, rtol=0, atol=0)
                torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
                torch.testing.assert_close(fresh_generator.get_state(), expected_loader, rtol=0, atol=0)
                self.assertEqual(fresh_scheduler.state_dict(), scheduler.state_dict())


if __name__ == "__main__":
    unittest.main()
