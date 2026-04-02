"""Tests for core/autoencoder.py — autoencoder-based anomaly detection."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")


class TestAutoencoderDetector:
    @pytest.fixture
    def sample_features(self):
        rng = np.random.RandomState(42)
        return rng.randn(200, 10).astype(np.float32)

    @pytest.fixture
    def anomalous_features(self):
        rng = np.random.RandomState(99)
        return (rng.randn(20, 10) * 10 + 5).astype(np.float32)

    def test_fit_returns_self(self, sample_features):
        from core.autoencoder import AutoencoderDetector
        det = AutoencoderDetector(encoding_dim=4, epochs=5)
        result = det.fit(sample_features)
        assert result is det

    def test_score_samples_shape(self, sample_features):
        from core.autoencoder import AutoencoderDetector
        det = AutoencoderDetector(encoding_dim=4, epochs=5)
        det.fit(sample_features)
        scores = det.score_samples(sample_features)
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (200,)

    def test_scores_between_0_and_1(self, sample_features):
        from core.autoencoder import AutoencoderDetector
        det = AutoencoderDetector(encoding_dim=4, epochs=5)
        det.fit(sample_features)
        scores = det.score_samples(sample_features)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_anomalies_score_higher(self, sample_features, anomalous_features):
        from core.autoencoder import AutoencoderDetector
        det = AutoencoderDetector(encoding_dim=4, epochs=10)
        det.fit(sample_features)
        normal_scores = det.score_samples(sample_features)
        anomaly_scores = det.score_samples(anomalous_features)
        assert anomaly_scores.mean() > normal_scores.mean()

    def test_score_before_fit_raises(self):
        from core.autoencoder import AutoencoderDetector
        det = AutoencoderDetector()
        with pytest.raises(RuntimeError, match="must be fit"):
            det.score_samples(np.zeros((10, 5)))

    def test_integrates_with_anomaly_scorer(self, sample_features):
        from core.autoencoder import AutoencoderDetector
        from log_anomaly_detection_lite import AnomalyScorer

        det = AutoencoderDetector(encoding_dim=4, epochs=5)
        det.fit(sample_features)
        ae_scores = det.score_samples(sample_features)

        scorer = AnomalyScorer()
        scorer.weights["autoencoder"] = 0.33
        scorer.weights["isolation_forest"] = 0.34
        scorer.weights["statistical"] = 0.33

        combined = scorer.combine_scores({
            "isolation_forest": np.random.rand(200),
            "statistical": np.random.rand(200),
            "autoencoder": ae_scores,
        })
        assert combined.shape == (200,)
