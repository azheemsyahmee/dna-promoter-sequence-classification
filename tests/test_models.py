"""
Unit tests for model architectures and factory registry.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models import TF_AVAILABLE, MODEL_REGISTRY, get_model_by_name


class TestModelsModule(unittest.TestCase):

    def test_model_registry_keys(self):
        expected_keys = [
            "exp1_bioaware_bilstm",
            "exp2_deep_stacked_lstm",
            "exp3_fast_sgd_lstm",
            "exp4_softmax_adam_lstm",
            "exp5_stable_adagrad_lstm"
        ]
        for key in expected_keys:
            self.assertIn(key, MODEL_REGISTRY)

    def test_invalid_model_identifier(self):
        with self.assertRaises(ValueError):
            get_model_by_name("non_existent_model")

    def test_bilstm_architecture_instantiation(self):
        if not TF_AVAILABLE:
            self.skipTest("TensorFlow is not installed in the active environment.")

        model = get_model_by_name("exp1_bioaware_bilstm", input_shape=(57, 4))
        self.assertIsNotNone(model)
        # Verify output layer shape: (None, 1)
        self.assertEqual(model.output_shape, (None, 1))

    def test_softmax_architecture_instantiation(self):
        if not TF_AVAILABLE:
            self.skipTest("TensorFlow is not installed in the active environment.")

        model = get_model_by_name("exp4_softmax_adam_lstm", input_shape=(57, 4))
        self.assertIsNotNone(model)
        # Verify output layer shape for 2 classes: (None, 2)
        self.assertEqual(model.output_shape, (None, 2))


if __name__ == "__main__":
    unittest.main()
