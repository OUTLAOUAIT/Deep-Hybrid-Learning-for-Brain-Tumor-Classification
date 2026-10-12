import unittest

from src.evaluation.metrics import classification_metrics


class MetricTests(unittest.TestCase):
    def test_perfect_predictions(self):
        result = classification_metrics([0, 1, 2], [0, 1, 2])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["f1_macro"], 1.0)


if __name__ == "__main__":
    unittest.main()
