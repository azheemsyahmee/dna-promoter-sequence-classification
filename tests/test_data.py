"""
Unit tests for data loading, sequence sanitization, one-hot encoding, and FASTA parsing.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data import (
    load_promoter_data,
    clean_sequences,
    one_hot_encode_sequence,
    one_hot_encode_sequences,
    parse_fasta,
    prepare_dataset
)


class TestDataModule(unittest.TestCase):

    def setUp(self):
        self.data_path = os.path.join(REPO_ROOT, "data", "promoters.data")

    def test_load_promoter_data(self):
        if not os.path.exists(self.data_path):
            self.skipTest("promoters.data file not found at default location.")
        df = load_promoter_data(self.data_path)
        self.assertEqual(len(df), 106)
        self.assertEqual(list(df.columns), ["label", "id", "sequence"])
        # Check label distribution: 53 positive, 53 negative
        counts = df["label"].value_counts().to_dict()
        self.assertEqual(counts.get("+"), 53)
        self.assertEqual(counts.get("-"), 53)

    def test_clean_sequences(self):
        raw_df = pd.DataFrame({
            "label": ["+", "-", "+"],
            "id": ["G1", "G2", "G3"],
            "sequence": ["\t tacta ", "CGTA", "invalid_base_x"]
        })
        cleaned = clean_sequences(raw_df)
        # Invalid sequence should be dropped
        self.assertEqual(len(cleaned), 2)
        # Sequences should be lowercased and stripped
        self.assertEqual(cleaned.iloc[0]["sequence"], "tacta")
        self.assertEqual(cleaned.iloc[1]["sequence"], "cgta")
        # Target should be mapped
        self.assertEqual(list(cleaned["target"]), [1, 0])

    def test_one_hot_encode_sequence(self):
        # 57-bp sequence of 'a's
        seq_a = "a" * 57
        enc_a = one_hot_encode_sequence(seq_a, target_length=57)
        self.assertEqual(enc_a.shape, (57, 4))
        # Base 'a' is [1, 0, 0, 0]
        np.testing.assert_array_equal(enc_a[0], np.array([1, 0, 0, 0], dtype=np.float32))

        # Test length mismatch error
        with self.assertRaises(ValueError):
            one_hot_encode_sequence("acgt", target_length=57)

        # Test invalid base error
        with self.assertRaises(ValueError):
            one_hot_encode_sequence("n" * 57, target_length=57)

    def test_one_hot_encode_sequences_batch(self):
        seqs = ["a" * 57, "c" * 57, "g" * 57]
        tensor = one_hot_encode_sequences(seqs, target_length=57)
        self.assertEqual(tensor.shape, (3, 57, 4))
        # Second sequence is 'c' -> [0, 1, 0, 0]
        np.testing.assert_array_equal(tensor[1, 0], np.array([0, 1, 0, 0], dtype=np.float32))

    def test_parse_fasta(self):
        fasta_text = """>seq1 E. coli promoter candidate
tactagcaatacgcttgcgttcggtggttaagtatgtataatgcgcgggcttgtcgt
>seq2 Non-promoter region
tgctatcctgacagttgtcacgctgattggtgtcgttacaatctaacgcatcgccaa
"""
        records = parse_fasta(fasta_text)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0][0], "seq1 E. coli promoter candidate")
        self.assertEqual(len(records[0][1]), 57)
        self.assertEqual(records[1][0], "seq2 Non-promoter region")
        self.assertEqual(len(records[1][1]), 57)

    def test_prepare_dataset(self):
        if not os.path.exists(self.data_path):
            self.skipTest("promoters.data file not found at default location.")
        X_train, X_test, y_train, y_test, df_clean = prepare_dataset(
            path_or_url=self.data_path,
            test_size=0.2,
            random_state=42
        )
        self.assertEqual(X_train.shape, (84, 57, 4))
        self.assertEqual(X_test.shape, (22, 57, 4))
        self.assertEqual(len(y_train), 84)
        self.assertEqual(len(y_test), 22)
        # Stratification check on test set: exactly 11 positive and 11 negative
        self.assertEqual(int(np.sum(y_test == 1)), 11)
        self.assertEqual(int(np.sum(y_test == 0)), 11)


if __name__ == "__main__":
    unittest.main()
