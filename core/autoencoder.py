"""Autoencoder-based anomaly detector.

Extracts the autoencoder component from the deprecated
``intrusion_detection_pipeline.py`` into a focused, testable module.

Requires TensorFlow (``pip install tensorflow>=2.14.0``).
"""

import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class AutoencoderDetector:
    """Reconstruction-error anomaly detector using a symmetric autoencoder."""

    def __init__(
        self,
        encoding_dim: int = 16,
        epochs: int = 50,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
    ):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._autoencoder: Sequential | None = None
        self._threshold: float | None = None

    def _build_model(self, input_dim: int) -> Sequential:
        encoder = Sequential(
            [
                Input(shape=(input_dim,)),
                Dense(64, activation="relu"),
                BatchNormalization(),
                Dropout(0.2),
                Dense(32, activation="relu"),
                Dense(self.encoding_dim, activation="relu"),
            ],
            name="encoder",
        )
        decoder = Sequential(
            [
                Dense(32, activation="relu", input_shape=(self.encoding_dim,)),
                BatchNormalization(),
                Dropout(0.2),
                Dense(64, activation="relu"),
                Dense(input_dim, activation="sigmoid"),
            ],
            name="decoder",
        )
        autoencoder = Sequential([encoder, decoder], name="autoencoder")
        autoencoder.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        return autoencoder

    def fit(self, features: np.ndarray) -> "AutoencoderDetector":
        self._autoencoder = self._build_model(features.shape[1])
        self._autoencoder.fit(
            features,
            features,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0,
        )
        errors = self._raw_errors(features)
        self._threshold = float(np.percentile(errors, 95))
        return self

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        if self._autoencoder is None:
            raise RuntimeError("AutoencoderDetector must be fit before scoring")
        errors = self._raw_errors(features)
        if self._threshold is None or self._threshold == 0:
            return np.zeros_like(errors)
        normalized = errors / self._threshold
        return np.clip(normalized, 0.0, 1.0)

    def _raw_errors(self, features: np.ndarray) -> np.ndarray:
        reconstructed = self._autoencoder.predict(features, verbose=0)
        return np.mean(np.square(features - reconstructed), axis=1)
