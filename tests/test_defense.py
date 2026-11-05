import hashlib
import tempfile
from pathlib import Path
import unittest

import numpy as np


class DefenseTests(unittest.TestCase):
    def test_mixed_f1_comes_from_mixed_predictions_and_exact_fixed_mask(self):
        # Averaging endpoint F1 values gives 1/15 across ten course classes;
        # recomputing the mixed confusion matrix gives 1/10.
        from dataset3.evaluate_defense import attack_metrics, fixed_mixed_mask
        labels = np.array([0, 0, 1, 1])
        clean = np.array([0, 0, 0, 0])
        attacked = np.array([1, 1, 1, 1])
        report = attack_metrics(labels, clean, attacked, clean,
                                np.array([True, False, True, False]))
        self.assertAlmostEqual(report["mixed_50pct_metrics"]["macro_f1"], 0.1)
        self.assertEqual(report["mixed_50pct_metrics"]["accuracy"], 0.5)
        self.assertEqual(report["attack_successes_on_initially_correct"], 2)
        self.assertEqual(report["attack_success_rate_on_initially_correct"], 1.0)
        self.assertEqual(report["quantized_attack_success_rate_on_initially_correct"], 0.0)
        self.assertAlmostEqual(report["mixed_50pct_quantized_8bit_metrics"]["macro_f1"], 1/15)
        mask = fixed_mixed_mask(1894)
        self.assertEqual(int(mask.sum()), 947)
        np.testing.assert_array_equal(mask, fixed_mixed_mask(1894))
        self.assertEqual(hashlib.sha256(mask.tobytes()).hexdigest(),
                         "64d2304ff025a400973ab1d2003a7a5948a395a96cc3f1612088a997da6e1277")

    def test_zero_initially_correct_asr_is_explicitly_undefined(self):
        from dataset3.evaluate_defense import attack_metrics
        report = attack_metrics(np.array([0, 1]), np.array([1, 0]),
                                np.array([0, 1]), np.array([0, 1]),
                                np.array([True, False]))
        self.assertEqual(report["attack_successes_on_initially_correct"], 0)
        self.assertIsNone(report["attack_success_rate_on_initially_correct"])

    def test_checkpoint_loads_trained_weights_strictly_and_records_identity(self):
        import torch
        from dataset3.evaluate_defense import load_defense
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "dataset3_model.py").write_text(
                "import torch\nfrom torch import nn\n"
                "class ProvidedNet(nn.Module):\n"
                " def __init__(self):\n"
                "  super().__init__(); self.weight=nn.Parameter(torch.ones(10, 2))\n"
                " def forward(self,x): return x.flatten(1) @ self.weight.T\n",
                encoding="utf-8")
            torch.save({"weight": torch.ones(10, 2)}, path / "provided_model.pt")
            checkpoint = path / "best.pt"
            torch.save({"model_state_dict": {"weight": torch.full((10, 2), 3.)},
                        "epoch": 2, "training_config": {"method": "pgd"},
                        "config": {"num_classes": 10, "input_range": [0., 1.]},
                        "provenance": {"scope": "diagnostic"},
                        "best_selection": {"mixed_macro_f1": 0.4}}, checkpoint)
            model, metadata = load_defense({"data_dir": str(path), "num_classes": 10,
                                            "input_range": [0., 1.]}, checkpoint,
                                           torch.device("cpu"))
            torch.testing.assert_close(model.weight, torch.full((10, 2), 3.))
            self.assertFalse(model.training)
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
            self.assertEqual(metadata["checkpoint_sha256"],
                             hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(metadata["epoch"], 2)
            self.assertEqual(metadata["provenance"]["scope"], "diagnostic")
            torch.save({"model_state_dict": {"weight": torch.ones(9, 2)}}, checkpoint)
            with self.assertRaises(RuntimeError):
                load_defense({"data_dir": str(path)}, checkpoint, torch.device("cpu"))

    def test_submission_rejects_reordered_ids_duplicate_ids_and_invalid_labels(self):
        from dataset3.predict_defense import validate_submission
        ids = ["2", "10", "1"]
        validate_submission(ids, np.array([0, 9, 3]), ids, expected_count=3)
        for bad_ids, labels, sample_ids in [
            (["10", "2", "1"], np.array([0, 9, 3]), ids),
            (["2", "2", "1"], np.array([0, 9, 3]), ["2", "2", "1"]),
            (ids, np.array([0, 10, 3]), ids),
            (ids, np.array([0., 9., 3.]), ids),
            (ids, np.array([0, 9]), ids),
        ]:
            with self.subTest(ids=bad_ids, labels=labels.tolist()):
                with self.assertRaises(ValueError):
                    validate_submission(bad_ids, labels, sample_ids, expected_count=3)

    def test_transfer_cache_reuses_source_attack_without_replacing_defense(self):
        import torch
        from torch import nn
        from torch.utils.data import TensorDataset
        from dataset3.evaluate_defense import evaluate_attack

        class BinaryPixels(nn.Module):
            def __init__(self, sign):
                super().__init__()
                self.sign = sign

            def forward(self, x):
                score = self.sign * (x.flatten(1).mean(1) - 0.5)
                return torch.stack([-score, score], dim=1)

        pixels = torch.tensor([[[[128/255]]], [[[127/255]]]])
        labels = torch.tensor([1, 0])
        dataset = TensorDataset(pixels, labels)
        config = {"seed": 5577, "batch_size": 2, "num_workers": 0}
        settings = {"name": "transfer_fgsm8", "method": "fgsm",
                    "epsilon": 8/255, "threat_model": "transfer"}
        clean = np.array([0, 1])  # Opposite-sign defense misclassifies both.
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(model=BinaryPixels(-1).eval(), source=BinaryPixels(1).eval(),
                          dataset=dataset, config=config, device=torch.device("cpu"),
                          settings=settings, clean=clean, mixed_mask=np.array([True, False]),
                          cache_dir=directory, cache_identity={"source": "source-a"})
            report, predictions, _ = evaluate_attack(**kwargs)
            self.assertFalse(report["cache"]["reused"])
            np.testing.assert_array_equal(predictions, [1, 0])
            again, repeated, _ = evaluate_attack(**kwargs)
            self.assertTrue(again["cache"]["reused"])
            np.testing.assert_array_equal(predictions, repeated)
            self.assertLessEqual(report["observed_linf"], 8/255 + 1e-6)
            self.assertLessEqual(report["observed_quantized_linf"], 8/255 + 1e-6)
            self.assertEqual(report["observed_pixel_range"][0] >= 0, True)
            self.assertEqual(report["observed_pixel_range"][1] <= 1, True)


if __name__ == "__main__":
    unittest.main()
