"""
Unit tests for classification evaluation metrics.
"""

import os
import sys
import unittest
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.evaluate import compute_classification_metrics


class TestEvaluateModule(unittest.TestCase):

    def test_compute_classification_metrics_perfect(self):
        y_true = np.array([1, 1, 0, 0], dtype=int)
        y_pred_prob = np.array([0.9, 0.8, 0.1, 0.2], dtype=float)

        metrics = compute_classification_metrics(y_true, y_pred_prob, threshold=0.5)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [0, 2]])

    def test_compute_classification_metrics_imperfect(self):
        y_true = np.array([1, 1, 0, 0], dtype=int)
        y_pred_prob = np.array([0.9, 0.3, 0.1, 0.8], dtype=float)  # 1 false negative, 1 false positive

        metrics = compute_classification_metrics(y_true, y_pred_prob, threshold=0.5)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])

    def test_two_column_probability_shape(self):
        y_true = np.array([0, 1], dtype=int)
        # 2D softmax probabilities: [[prob_0, prob_1], ...]
        y_pred_prob = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=float)

        metrics = compute_classification_metrics(y_true, y_pred_prob, threshold=0.5)
        self.assertEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
