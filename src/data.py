"""
Data loading, cleaning, validation, and sequence encoding utilities
for DNA promoter sequence classification.
"""

import os
import re
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_TO_VECTOR: Dict[str, List[int]] = {
    "a": [1, 0, 0, 0],
    "c": [0, 1, 0, 0],
    "g": [0, 0, 1, 0],
    "t": [0, 0, 0, 1]
}

VECTOR_TO_BASE: Dict[int, str] = {
    0: "a",
    1: "c",
    2: "g",
    3: "t"
}


def load_promoter_data(path_or_url: str = "data/promoters.data") -> pd.DataFrame:
    """
    Load the UCI Molecular Biology Promoter Gene Sequences dataset.

    Args:
        path_or_url: Local file path or HTTP(S) URL to promoters.data.

    Returns:
        DataFrame with columns ['label', 'id', 'sequence'].
    """
    # If a relative path is passed, try fallback to common directory locations
    candidates = [
        path_or_url,
        os.path.join("data", "promoters.data"),
        os.path.join("..", "data", "promoters.data")
    ]
    resolved_path = next((p for p in candidates if os.path.exists(p)), path_or_url)

    df = pd.read_csv(resolved_path, header=None, names=["label", "id", "sequence"])
    return df


def clean_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize DNA sequences by removing whitespace, converting to lowercase,
    and filtering for valid canonical nucleotide bases (a, c, g, t).

    Args:
        df: Input DataFrame with 'label' and 'sequence' columns.

    Returns:
        Cleaned copy of the DataFrame with added binary 'target' column.
    """
    cleaned_df = df.copy()
    cleaned_df["sequence"] = cleaned_df["sequence"].astype(str)
    cleaned_df["sequence"] = cleaned_df["sequence"].str.replace(r"\s+", "", regex=True).str.lower()

    valid_bases = set("acgt")
    is_valid = cleaned_df["sequence"].apply(lambda seq: set(seq).issubset(valid_bases))
    cleaned_df = cleaned_df[is_valid].copy()

    # Map '+' to 1 (Promoter) and '-' to 0 (Non-Promoter)
    cleaned_df["target"] = cleaned_df["label"].map({"+": 1, "-": 0})
    return cleaned_df


def one_hot_encode_sequence(sequence: str, target_length: int = 57) -> np.ndarray:
    """
    One-hot encode a single DNA nucleotide sequence into a (target_length, 4) tensor.

    Args:
        sequence: String of nucleotide characters (a, c, g, t).
        target_length: Expected sequence length (default: 57).

    Returns:
        Numpy array of shape (target_length, 4) with dtype float32.
    """
    seq_clean = re.sub(r"\s+", "", sequence).lower()
    if len(seq_clean) != target_length:
        raise ValueError(
            f"Expected sequence length of {target_length} bp, but received {len(seq_clean)} bp."
        )

    for idx, base in enumerate(seq_clean):
        if base not in BASE_TO_VECTOR:
            raise ValueError(f"Invalid non-canonical nucleotide base '{base}' at position {idx}.")

    encoded = np.array([BASE_TO_VECTOR[b] for b in seq_clean], dtype=np.float32)
    return encoded


def one_hot_encode_sequences(sequences: Union[pd.Series, List[str]], target_length: int = 57) -> np.ndarray:
    """
    Encode an array or Series of DNA sequences into a 3D tensor of shape (N, target_length, 4).

    Args:
        sequences: Iterable of DNA sequence strings.
        target_length: Expected sequence length (default: 57).

    Returns:
        Numpy array of shape (N, target_length, 4).
    """
    encoded_list = [one_hot_encode_sequence(seq, target_length=target_length) for seq in sequences]
    return np.stack(encoded_list, axis=0)


def parse_fasta(fasta_input: str) -> List[Tuple[str, str]]:
    """
    Parse a FASTA-formatted string or file into a list of (header, sequence) tuples.

    Args:
        fasta_input: Multi-line FASTA string or path to a .fasta / .fa file.

    Returns:
        List of tuples: [(header_1, seq_1), (header_2, seq_2), ...]
    """
    if os.path.exists(fasta_input):
        with open(fasta_input, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = fasta_input

    records = []
    current_header = None
    current_seq_lines = []

    for line in content.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header is not None:
                records.append((current_header, "".join(current_seq_lines).lower()))
                current_seq_lines = []
            current_header = line[1:].strip()
        else:
            current_seq_lines.append(line)

    if current_header is not None:
        records.append((current_header, "".join(current_seq_lines).lower()))

    return records


def prepare_dataset(
    path_or_url: str = "data/promoters.data",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Complete end-to-end dataset loading, cleaning, encoding, and stratified splitting.

    Args:
        path_or_url: Path or URL to the promoters.data file.
        test_size: Proportion of dataset to hold out for testing (default: 0.2).
        random_state: Random state seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test, df_cleaned
    """
    df_raw = load_promoter_data(path_or_url)
    df_clean = clean_sequences(df_raw)

    X = one_hot_encode_sequences(df_clean["sequence"], target_length=57)
    y = df_clean["target"].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, df_clean
