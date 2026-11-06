"""Cached teacher attacks, paired augmentation and single-forward BN training."""
import copy
import importlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset3.common import ImageRecords, load_records
from dataset3.training import load_checkpoint, save_checkpoint


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))

    def forward(self, images):
        return self.head(self.bn(images))


class CachedSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.cached = importlib.import_module("dataset3.cached_source_training")
        except ModuleNotFoundError as error:
            if error.name != "dataset3.cached_source_training":
                raise
            cls.cached = None

    def setUp(self):
        self.assertIsNotNone(self.cached, "Cached source training module is missing")
        torch.set_num_threads(1)
        torch.manual_seed(5577)
        self.settings = {"augmentation": True, "clean_weight": 0.5, "fgsm_probability": 0.5}

    def loader(self, generator):
        clean = torch.linspace(0.1, 0.8, 8 * 3 * 8 * 8).reshape(8, 3, 8, 8)
        return DataLoader(TensorDataset(clean, clean + .01, clean + .02,
                                       torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])),
                          batch_size=4, shuffle=True, generator=generator)

    def test_paired_augmentation_uses_identical_crop_and_flip(self):
        clean = torch.arange(2 * 3 * 8 * 8).float().reshape(2, 3, 8, 8) / 1000 + .1
        adversarial = clean + .02
        a, b = self.cached.augment_pair(clean, adversarial)
        torch.testing.assert_close(b - a, torch.full_like(a, .02), rtol=0, atol=1e-7)
        self.assertFalse(torch.equal(a, clean))
        self.assertGreaterEqual(a.min().item(), 0)
        self.assertLessEqual(b.max().item(), 1)
        torch.manual_seed(5577)
        again = self.cached.augment_pair(clean, adversarial)
        torch.testing.assert_close(a, again[0], rtol=0, atol=0)
        torch.testing.assert_close(b, again[1], rtol=0, atol=0)

    def test_joint_forward_updates_parameters_and_batchnorm_once_per_batch(self):
        model = TinyNet()
        before = model.head[-1].weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        metrics = self.cached.train_one_epoch(model, self.loader(torch.Generator().manual_seed(5577)),
                                               optimizer, "cpu", self.settings)
        self.assertFalse(torch.equal(before, model.head[-1].weight))
        self.assertEqual(model.bn.num_batches_tracked.item(), 2)
        self.assertEqual(metrics["records"], 8)
        self.assertEqual(metrics["fgsm_records"] + metrics["pgd_records"], 8)
        self.assertAlmostEqual(metrics["loss"], .5 * metrics["clean_ce"] + .5 * metrics["adversarial_ce"], places=6)

    def test_dataset_yields_aligned_consistent_float_images_and_int_labels(self):
        class Records:
            images = torch.full((2, 3, 8, 8), 128, dtype=torch.uint8)
            labels = np.array([1, 0])
            def __len__(self):
                return 2
        attacks = {"fgsm": torch.full((2, 3, 8, 8), .5),
                   "pgd": torch.full((2, 3, 8, 8), .52)}
        dataset = self.cached.CachedPairs(Records(), attacks)
        clean, fgsm, pgd, label = dataset[1]
        self.assertEqual(label, 0)
        self.assertIsInstance(label, int)
        self.assertEqual(clean.dtype, torch.float32)
        torch.testing.assert_close(clean, torch.full_like(clean, 128 / 255))
        torch.testing.assert_close(fgsm, attacks["fgsm"][1])
        torch.testing.assert_close(pgd, attacks["pgd"][1])
        self.assertEqual(next(iter(DataLoader(dataset, batch_size=2)))[0].shape, (2, 3, 8, 8))

    def test_teacher_cache_identity_reuse_and_source_immutability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, split, cache = root / "data", root / "split", root / "cache"
            data.mkdir(); split.mkdir(); (data / "train_images").mkdir()
            for index in range(2):
                Image.fromarray(np.full((2, 2, 3), 64 + 64 * index, dtype=np.uint8)).save(data / "train_images" / f"{index}.png")
            train_csv = split / "train.csv"
            train_csv.write_text("id,label\n0,0\n1,1\n", encoding="utf-8")
            (data / "dataset3_model.py").write_text("# architecture identity\n", encoding="utf-8")
            source = nn.Sequential(nn.Flatten(), nn.Linear(12, 2)).eval().requires_grad_(False)
            torch.save(source.state_dict(), data / "provided_model.pt")
            before = copy.deepcopy(source.state_dict())
            records = ImageRecords(load_records(train_csv, True), data / "train_images", True)
            config = {"data_dir": str(data), "split_dir": str(split), "seed": 5577,
                      "batch_size": 2, "num_workers": 0}
            original_rng = torch.get_rng_state().clone()
            first, first_meta = self.cached.prepare_training_attacks(source, records, config, "cpu", cache, train_csv)
            torch.testing.assert_close(torch.get_rng_state(), original_rng, rtol=0, atol=0)
            second, second_meta = self.cached.prepare_training_attacks(source, records, config, "cpu", cache, train_csv)
            self.assertTrue(all(v["reused"] for v in second_meta.values()))
            for name in ["fgsm", "pgd"]:
                torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
                self.assertEqual(first_meta[name]["identity"]["training_csv_sha256"], second_meta[name]["identity"]["training_csv_sha256"])
                self.assertLessEqual((first[name] - records.images.float() / 255).abs().max().item(), 8 / 255 + 1e-6)
            for name, expected in before.items():
                torch.testing.assert_close(source.state_dict()[name], expected, rtol=0, atol=0)
            self.assertFalse(source.training)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in source.parameters()))
            train_csv.write_text("id,label\n0,1\n1,0\n", encoding="utf-8")
            _, changed = self.cached.prepare_training_attacks(source, records, config, "cpu", cache, train_csv)
            self.assertTrue(all(not v["reused"] for v in changed.values()))

    def test_exact_resume_preserves_bernoulli_augmentation_optimizer_and_loader(self):
        model = TinyNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3)
        generator = torch.Generator().manual_seed(5577)
        self.cached.train_one_epoch(model, self.loader(generator), optimizer, "cpu", self.settings)
        scheduler.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, model, optimizer, scheduler, 1, generator, {}, {}, {}, [], {})
            self.cached.train_one_epoch(model, self.loader(generator), optimizer, "cpu", self.settings)
            scheduler.step()
            expected = copy.deepcopy(model.state_dict())
            expected_rng, expected_loader = torch.get_rng_state().clone(), generator.get_state().clone()
            resumed = TinyNet()
            opt = torch.optim.SGD(resumed.parameters(), lr=.01, momentum=.9)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3)
            gen = torch.Generator()
            load_checkpoint(path, resumed, opt, sched, gen)
            self.cached.train_one_epoch(resumed, self.loader(gen), opt, "cpu", self.settings)
            sched.step()
            for name, value in expected.items():
                torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
            torch.testing.assert_close(gen.get_state(), expected_loader, rtol=0, atol=0)
            self.assertEqual(scheduler.state_dict(), sched.state_dict())

    def test_refit_protocol_has_no_validation_and_teacher_scope_is_diagnostic(self):
        scope = self.cached.experiment_scope(False)
        self.assertIn("teacher", scope)
        self.assertIn("diagnostic", scope)
        self.assertIn("no held-out metric", self.cached.experiment_scope(True))
