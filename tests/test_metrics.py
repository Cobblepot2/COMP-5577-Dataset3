import unittest
import numpy as np
from dataset3.metrics import classification_metrics


class MetricsTests(unittest.TestCase):
    def test_hand_calculated_confusion_and_macro_f1(self):
        result = classification_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 3)
        self.assertEqual(result["confusion_matrix_rows_true_columns_predicted"], [[1, 1, 0], [0, 2, 0], [0, 0, 0]])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], (2/3 + 4/5 + 0) / 3)
        self.assertEqual(result["per_class"][2]["f1"], 0)


if __name__ == "__main__":
    unittest.main()
