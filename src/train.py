"""
Training routines and experiment execution for DNA promoter sequence classification.
"""

from typing import Dict, Any, Tuple
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from src.data import prepare_dataset
from src.models import get_model_by_name


def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 20,
    batch_size: int = 16,
    validation_split: float = 0.2,
    patience: int = 10,
    verbose: int = 1
) -> Any:
    """
    Train a compiled Keras model using early stopping on validation loss.

    Args:
        model: Compiled Keras model instance.
        X_train: Feature tensor of shape (N, 57, 4).
        y_train: Target labels.
        epochs: Maximum training epochs.
        batch_size: Batch size for gradient descent.
        validation_split: Proportion of training data for validation monitoring.
        patience: Number of epochs without improvement before stopping.
        verbose: Keras verbosity level.

    Returns:
        Keras History object.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required to train Keras models.")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose
    )
    return history


def run_experiment(
    exp_name: str,
    data_path: str = "data/promoters.data",
    epochs: int = 20,
    batch_size: int = 16,
    random_state: int = 42
) -> Tuple[Any, Any, Dict[str, float], Tuple[np.ndarray, np.ndarray]]:
    """
    Execute an end-to-end experiment pipeline: data preparation, model instantiation,
    training with early stopping, and holdout test evaluation.

    Args:
        exp_name: Identifier registered in MODEL_REGISTRY.
        data_path: Path to promoters.data file.
        epochs: Maximum training epochs.
        batch_size: Batch size.
        random_state: Splitting random seed.

    Returns:
        model, history, test_metrics, (X_test, y_test)
    """
    X_train, X_test, y_train, y_test, _ = prepare_dataset(
        path_or_url=data_path,
        test_size=0.2,
        random_state=random_state
    )

    model = get_model_by_name(exp_name, input_shape=(X_train.shape[1], X_train.shape[2]))
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    eval_results = model.evaluate(X_test, y_test, verbose=0)
    test_metrics = {
        "test_loss": float(eval_results[0]),
        "test_accuracy": float(eval_results[1])
    }

    return model, history, test_metrics, (X_test, y_test)
