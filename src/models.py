"""
TensorFlow/Keras model architectures for DNA promoter classification.
Faithfully preserves the 5 experimental architectures explored in coursework:
- Exp 1: Bio-Aware Bidirectional LSTM with RMSprop
- Exp 2: Deep Stacked LSTM with LeakyReLU and Adamax
- Exp 3: Lightweight Single LSTM with SGD
- Exp 4: Multi-Class Softmax with Sparse Categorical Cross-Entropy
- Exp 5: Regularized LSTM with High Dropout and Adagrad
"""

from typing import Tuple, Dict, Any

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, LeakyReLU
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def build_bilstm_model(
    input_shape: Tuple[int, int] = (57, 4),
    lstm_units: int = 31,
    dense_units: int = 32,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.0031
) -> Any:
    """
    [EXPERIMENT 1: BIO-AWARE MODEL]
    Bidirectional LSTM with Tanh activation and RMSprop optimizer.
    Captures both upstream (5' -> 3') and downstream (3' -> 5') sequence context.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required to build Keras models. Install via `pip install tensorflow`.")

    model = Sequential([
        Bidirectional(LSTM(lstm_units, return_sequences=False), input_shape=input_shape),
        Dropout(dropout_rate),
        Dense(dense_units, activation="tanh"),
        Dropout(dropout_rate),
        Dense(1, activation="sigmoid")
    ], name="Exp1_BioAware_BiLSTM")

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_deep_lstm_model(
    input_shape: Tuple[int, int] = (57, 4),
    lstm_units: int = 31,
    dense_units: int = 32,
    learning_rate: float = 0.0031
) -> Any:
    """
    [EXPERIMENT 2: DEEP MEMORY MODEL]
    Stacked 2-layer LSTM with LeakyReLU activation and Adamax optimizer.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required to build Keras models.")

    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        LSTM(lstm_units),
        Dropout(0.5),
        Dense(dense_units),
        LeakyReLU(),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ], name="Exp2_DeepMemory_StackedLSTM")

    optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_fast_lstm_model(
    input_shape: Tuple[int, int] = (57, 4),
    lstm_units: int = 31,
    dense_units: int = 32,
    learning_rate: float = 0.01
) -> Any:
    """
    [EXPERIMENT 3: FAST & AGGRESSIVE MODEL]
    Lightweight single LSTM with ReLU activation and SGD optimizer.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required to build Keras models.")

    model = Sequential([
        LSTM(lstm_units, input_shape=input_shape),
        Dropout(0.2),
        Dense(dense_units, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ], name="Exp3_FastAggressive_SGD")

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_softmax_lstm_model(
    input_shape: Tuple[int, int] = (57, 4),
    lstm_units: int = 31,
    dense_units: int = 32,
    learning_rate: float = 0.0031
) -> Any:
    """
    [EXPERIMENT 4: SOFTMAX + SPARSE CE MODEL]
    Single LSTM with 2-unit Softmax output and Sparse Categorical Cross-Entropy.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required to build Keras models.")

    model = Sequential([
        LSTM(lstm_units, input_shape=input_shape),
        Dropout(0.3),
        Dense(dense_units, activation="relu"),
        Dropout(0.3),
        Dense(2, activation="softmax")
    ], name="Exp4_Softmax_SparseCE")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_stable_lstm_model(
    input_shape: Tuple[int, int] = (57, 4),
    lstm_units: int = 31,
    dense_units: int = 32,
    learning_rate: float = 0.001
) -> Any:
    """
    [EXPERIMENT 5: STABLE GENERALIZATION MODEL]
    Single LSTM with high dropout (0.5), Tanh activation, and Adagrad optimizer.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required to build Keras models.")

    model = Sequential([
        LSTM(lstm_units, input_shape=input_shape),
        Dropout(0.5),
        Dense(dense_units, activation="tanh"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ], name="Exp5_StableGeneralization_Adagrad")

    optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


MODEL_REGISTRY: Dict[str, Any] = {
    "exp1_bioaware_bilstm": build_bilstm_model,
    "exp2_deep_stacked_lstm": build_deep_lstm_model,
    "exp3_fast_sgd_lstm": build_fast_lstm_model,
    "exp4_softmax_adam_lstm": build_softmax_lstm_model,
    "exp5_stable_adagrad_lstm": build_stable_lstm_model
}


def get_model_by_name(name: str, input_shape: Tuple[int, int] = (57, 4)) -> Any:
    """Factory accessor returning compiled model instance by identifier."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model identifier '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](input_shape=input_shape)
