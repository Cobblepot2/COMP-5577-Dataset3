import math
import unittest

import torch
from torch import nn


class LeftPixel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(math.log(9.), dtype=torch.float64))

    def forward(self, images):
        value = self.scale * images[:, 0, 0, 0]
        return torch.stack([value, torch.zeros_like(value)], dim=1)


class CompetingClasses(nn.Module):
    """At x=.5, margin ascent moves right while CE ascent moves left."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))

    def forward(self, images):
        value = self.scale * (images.flatten(1).mean(1) - .5)
        return torch.stack([torch.zeros_like(value), value - .1, -2 * value - .2], dim=1)


class TTAStressTests(unittest.TestCase):
    def test_flip_averages_probabilities_and_differentiates_both_views(self):
        from dataset3.tta import HorizontalFlipAverage
        model = LeftPixel().eval().requires_grad_(False)
        x = torch.tensor([[[[1., 0.]]]], dtype=torch.float64, requires_grad=True)
        original = x.detach().clone()
        logits = HorizontalFlipAverage(model).eval()(x)
        # Softmaxes are [.9,.1] and [.5,.5]; averaging logits would give [.75,.25].
        torch.testing.assert_close(logits.exp(), torch.tensor([[.7, .3]], dtype=torch.float64))
        gradient, = torch.autograd.grad(logits.exp()[0, 0], x)
        expected = torch.tensor([[[[.045 * math.log(9.), .125 * math.log(9.)]]]], dtype=torch.float64)
        torch.testing.assert_close(gradient, expected)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(torch.all(gradient != 0))
        torch.testing.assert_close(x.detach(), original, rtol=0, atol=0)
        self.assertIsNone(model.scale.grad)
        self.assertFalse(model.scale.requires_grad)

    def test_default_predictor_preserves_original_logits_exactly(self):
        from dataset3.tta import configure_predictor, HorizontalFlipAverage
        model = LeftPixel().eval().requires_grad_(False)
        x = torch.tensor([[[[.8, .2]]]], dtype=torch.float64)
        predictor, identity = configure_predictor(model)
        self.assertIs(predictor, model)
        torch.testing.assert_close(predictor(x), model(x), rtol=0, atol=0)
        self.assertEqual(identity["views"], ["identity"])
        flipped, metadata = configure_predictor(model, horizontal_flip=True)
        self.assertIsInstance(flipped, HorizontalFlipAverage)
        self.assertFalse(flipped.training)
        self.assertEqual(metadata["aggregation"], "arithmetic_mean_softmax_probabilities")
        self.assertEqual(metadata["views"], ["identity", "horizontal_flip"])
        self.assertNotEqual(identity, metadata)

    def test_log_probability_average_remains_finite_for_extreme_logits(self):
        from dataset3.tta import HorizontalFlipAverage
        model = LeftPixel().eval().requires_grad_(False)
        with torch.no_grad():
            model.scale.fill_(2000.)
        x = torch.ones(1, 1, 1, 2, dtype=torch.float64, requires_grad=True)
        logits = HorizontalFlipAverage(model)(x)
        self.assertTrue(torch.isfinite(logits).all())
        torch.testing.assert_close(logits, torch.tensor([[0., -2000.]], dtype=torch.float64))
        gradient, = torch.autograd.grad(torch.nn.functional.cross_entropy(logits, torch.tensor([1])), x)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(torch.all(gradient > 0))

    def test_margin_attack_uses_margin_gradient_instead_of_cross_entropy(self):
        from dataset3.stress_attacks import margin_pgd, logit_margin
        from dataset3.evaluate_defense import _generate
        model = CompetingClasses().eval()
        x = torch.tensor([[[[.5]]]])
        y = torch.tensor([0])
        logits = model(x)
        torch.testing.assert_close(logit_margin(logits, y), torch.tensor([-.1]))
        result = margin_pgd(model, x, y, .02, .01, 1, random_start=False)
        torch.testing.assert_close(result, torch.tensor([[[[.51]]]]))
        settings = {"method": "margin_pgd", "epsilon": .02,
                    "step_size": .01, "steps": 1, "restarts": 1}
        actual = _generate(model, x, y, settings, torch.Generator().manual_seed(0))
        self.assertGreater(actual.item(), .5)  # CE dispatch would move left.
        zero = margin_pgd(model, x, y, 0, .01, 1,
                          generator=torch.Generator().manual_seed(5577))
        torch.testing.assert_close(zero, x, rtol=0, atol=0)
        self.assertIsNone(model.scale.grad)
        self.assertFalse(result.requires_grad)
        torch.testing.assert_close(x, torch.tensor([[[[.5]]]]), rtol=0, atol=0)

    def test_margin_retains_earlier_failure_when_last_step_recovers(self):
        from dataset3.stress_attacks import margin_pgd

        class Curved(nn.Module):
            def forward(self, images):
                z = .05 - 10 * (images.flatten(1).mean(1) - .53).square()
                return torch.stack([torch.zeros_like(z), z], dim=1)

        model = Curved().eval()
        x = torch.tensor([[[[.5]]]])
        y = torch.tensor([0])
        result = margin_pgd(model, x, y, .4, .2, 1, random_start=False)
        torch.testing.assert_close(result, x, rtol=0, atol=0)
        self.assertEqual(model(result).argmax(1).item(), 1)
        self.assertEqual(model(torch.tensor([[[[.7]]]])).argmax(1).item(), 0)

    def test_margin_random_starts_are_seeded_bounded_and_keep_found_failures(self):
        from dataset3.stress_attacks import margin_pgd
        model = CompetingClasses().eval()
        started = margin_pgd(model, torch.tensor([[[[.5]]]]), torch.tensor([0]),
                             .02, 0, 1, generator=torch.Generator().manual_seed(1))
        self.assertGreater(started.item(), .5)
        x = torch.tensor([[[[0.]]], [[[.5]]], [[[1.]]]])
        y = torch.tensor([0, 0, 0])
        first = margin_pgd(model, x, y, .2, .03, 2, restarts=1,
                           generator=torch.Generator().manual_seed(5577))
        result = margin_pgd(model, x, y, .2, .03, 7, restarts=3,
                            generator=torch.Generator().manual_seed(5577))
        repeat = margin_pgd(model, x, y, .2, .03, 7, restarts=3,
                            generator=torch.Generator().manual_seed(5577))
        torch.testing.assert_close(result, repeat, rtol=0, atol=0)
        self.assertLessEqual((result - x).abs().max().item(), .2 + 1e-6)
        self.assertGreaterEqual(result.min().item(), 0)
        self.assertLessEqual(result.max().item(), 1)
        self.assertTrue(torch.all((model(first).argmax(1) == y) | (model(result).argmax(1) != y)))
        self.assertIsNone(model.scale.grad)

    def test_stress_is_opt_in_and_ce_search_preserves_shorter_failures(self):
        from dataset3.evaluate_defense import attack_suite, _generate
        regular = attack_suite("full")
        self.assertEqual([s["name"] for s in regular], [
            "white_box_fgsm8", "white_box_pgd10", "white_box_pgd20x3",
            "transfer_fgsm8", "transfer_pgd10", "transfer_pgd20x3"])
        stress = attack_suite("full", stress=True)
        self.assertEqual([s["name"] for s in stress[-2:]],
                         ["white_box_pgd50x5", "white_box_margin_pgd50x5"])
        with self.assertRaises(ValueError):
            attack_suite("quick", stress=True)
        model = CompetingClasses().eval()
        x = torch.tensor([[[[.48]]], [[[.5]]], [[[.52]]]])
        y = torch.tensor([0, 0, 0])
        # A smaller synthetic search budget keeps this CPU regression quick.
        weak = dict(regular[2], steps=3, restarts=2, epsilon=.2, step_size=.03)
        strong = dict(stress[-2], steps=6, restarts=3, epsilon=.2, step_size=.03)
        a = _generate(model, x, y, weak, torch.Generator().manual_seed(5577))
        b = _generate(model, x, y, strong, torch.Generator().manual_seed(5577))
        self.assertTrue(torch.all((model(a).argmax(1) == y) | (model(b).argmax(1) != y)))


if __name__ == "__main__":
    unittest.main()
