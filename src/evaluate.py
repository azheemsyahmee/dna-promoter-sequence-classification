"""
Evaluation metrics, confusion matrix computation, and single-sequence inference.
"""

from typing import Dict, Any, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from src.data import one_hot_encode_sequence


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute binary classification metrics from ground truth and predicted probabilities.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred_prob: Predicted probabilities. If shape is (N, 2), probability for class 1 is used.
        threshold: Decision threshold for positive classification.

    Returns:
        Dictionary of computed metrics (accuracy, precision, recall, f1, roc_auc, confusion_matrix).
    """
    if y_pred_prob.ndim == 2 and y_pred_prob.shape[1] == 2:
        probs = y_pred_prob[:, 1]
    else:
        probs = y_pred_prob.flatten()

    y_pred = (probs >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, probs))
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "support_total": int(len(y_true)),
        "support_positive": int(np.sum(y_true == 1)),
        "support_negative": int(np.sum(y_true == 0))
    }


def predict_sequence(
    model: Any,
    sequence: str,
    target_length: int = 57,
    threshold: float = 0.5
) -> Dict[str, Union[float, str, bool]]:
    """
    Predict whether an arbitrary DNA sequence is an E. coli promoter.

    Args:
        model: Trained Keras model instance.
        sequence: Raw nucleotide sequence string.
        target_length: Expected sequence length (default: 57 bp).
        threshold: Decision threshold.

    Returns:
        Dictionary with probability, binary prediction, and biological interpretation.
    """
    encoded = one_hot_encode_sequence(sequence, target_length=target_length)
    input_tensor = np.expand_dims(encoded, axis=0)  # Shape: (1, 57, 4)

    raw_pred = model.predict(input_tensor, verbose=0)
    if raw_pred.ndim == 2 and raw_pred.shape[1] == 2:
        prob = float(raw_pred[0, 1])
    else:
        prob = float(raw_pred[0, 0])

    is_promoter = bool(prob >= threshold)
    return {
        "probability": prob,
        "is_promoter": is_promoter,
        "label": "Promoter (+)" if is_promoter else "Non-Promoter (-)"
    }
