import unittest

import torch
from torch import nn

from dataset3.attacks import fgsm, pgd


class LogisticExample(nn.Module):
    """Week 5 worked example: z = 0.9*x1 - 1.1*x2 + 0.1."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.9, -1.1]))

    def forward(self, images):
        z = images.flatten(1) @ self.weight + 0.1
        return torch.stack([torch.zeros_like(z), z], dim=1)


class AttackTests(unittest.TestCase):
    def setUp(self):
        self.model = LogisticExample().eval()
        self.x = torch.tensor([[[[0.8, 0.3]]]])
        self.y = torch.tensor([1])

    def test_fgsm_matches_analytic_course_example(self):
        result = fgsm(self.model, self.x, self.y, 0.25)
        torch.testing.assert_close(result, torch.tensor([[[[0.55, 0.55]]]]))
        self.assertEqual(self.model(result).argmax(1).item(), 0)
        self.assertIsNone(self.model.weight.grad)
        torch.testing.assert_close(self.model.weight, torch.tensor([0.9, -1.1]))
        torch.testing.assert_close(self.x, torch.tensor([[[[0.8, 0.3]]]]))
        self.assertFalse(result.requires_grad)

    def test_zero_budget_is_identity(self):
        torch.testing.assert_close(fgsm(self.model, self.x, self.y, 0), self.x)
        result = pgd(self.model, self.x, self.y, 0, 0.1, 3, generator=torch.Generator().manual_seed(3))
        torch.testing.assert_close(result, self.x)

    def test_single_step_pgd_matches_fgsm_without_random_start(self):
        expected = fgsm(self.model, self.x, self.y, 0.25)
        actual = pgd(self.model, self.x, self.y, 0.25, 0.25, 1, random_start=False)
        torch.testing.assert_close(actual, expected)

    def test_projection_clipping_seed_and_restart_preservation(self):
        images = torch.tensor([[[[0., 1.]]], [[[0.8, 0.3]]], [[[1., 0.]]]])
        labels = torch.tensor([0, 1, 1])
        result = pgd(self.model, images, labels, 0.25, 0.2, 6, restarts=3,
                     generator=torch.Generator().manual_seed(5577))
        again = pgd(self.model, images, labels, 0.25, 0.2, 6, restarts=3,
                    generator=torch.Generator().manual_seed(5577))
        torch.testing.assert_close(result, again, rtol=0, atol=0)
        self.assertLessEqual((result - images).abs().max().item(), 0.25 + 1e-6)
        self.assertGreaterEqual(result.min().item(), 0)
        self.assertLessEqual(result.max().item(), 1)
        first = pgd(self.model, images, labels, 0.25, 0.2, 3,
                    generator=torch.Generator().manual_seed(5577))
        failed_first = self.model(first).argmax(1) != labels
        failed_all = self.model(result).argmax(1) != labels
        self.assertTrue(torch.all(~failed_first | failed_all))
        self.assertIsNone(self.model.weight.grad)


if __name__ == "__main__":
    unittest.main()
