#!/usr/bin/env python3
"""
Runtime chat anomaly detector.
Loads trained ChatFeaturePipeline + SGDClassifier and scores messages.
Target: <50ms inference per message.
"""

import time
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Optional


class ChatAnomalyDetector:
    """Runtime inference wrapper for chat injection detection."""

    SEVERITY_THRESHOLDS = {
        "critical": 0.95,
        "high": 0.85,
        "medium": 0.70,
        "low": 0.50,
    }

    THREAT_TYPES = {
        "injection": ["ignore", "disregard", "forget", "bypass", "override", "previous instructions"],
        "jailbreak": ["dan", "jailbreak", "unrestricted", "no restrictions", "god mode", "fictional universe"],
        "credential_theft": ["password", "credential", "api key", "secret", "token", ".env"],
        "system_probe": ["system prompt", "your instructions", "your prompt", "your rules", "core directives"],
        "social_engineering": ["it support", "security audit", "authorized", "compliance", "supervisor", "ceo"],
        "encoding_attack": ["base64", "decode", "encode", "hex", "rot13"],
        "logic_trap": ["you must answer", "obligated", "proves you're biased", "if you refuse"],
    }

    def __init__(self):
        self.pipeline = None
        self.classifier = None
        self.metadata = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self, model_dir: str) -> bool:
        """Load model artifacts from directory."""
        model_path = Path(model_dir)

        # Try combined package first
        package_path = model_path / "chat_inference_package.pkl"
        if package_path.exists():
            package = joblib.load(package_path)
            self.pipeline = package["feature_pipeline"]
            self.classifier = package["classifier"]
            self.metadata = package.get("metadata", {})
            self._loaded = True
            return True

        # Fall back to individual files
        pipeline_path = model_path / "chat_feature_pipeline.pkl"
        classifier_path = model_path / "chat_classifier.pkl"

        if pipeline_path.exists() and classifier_path.exists():
            self.pipeline = joblib.load(pipeline_path)
            self.classifier = joblib.load(classifier_path)
            self.metadata = {}
            self._loaded = True
            return True

        return False

    def score(self, message: str) -> Dict:
        """
        Score a single chat message for injection probability.

        Returns dict with:
            score: float [0.0, 1.0] - probability of being malicious
            threat_type: str - classified threat category
            severity: str - low/medium/high/critical
            confidence: float - model confidence
            processing_time_ms: float - inference latency
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start = time.perf_counter()

        # Extract features and predict
        features = self.pipeline.transform([message])
        probabilities = self.classifier.predict_proba(features)[0]

        # Score is probability of malicious class (index 1)
        score = float(probabilities[1])
        confidence = float(max(probabilities))

        # Classify threat type
        threat_type = self._classify_threat(message, score)

        # Assign severity
        severity = self._assign_severity(score)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "score": round(score, 4),
            "threat_type": threat_type,
            "severity": severity,
            "confidence": round(confidence, 4),
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def _classify_threat(self, message: str, score: float) -> str:
        """Classify the threat type based on message content."""
        if score < 0.5:
            return "none"

        text_lower = message.lower()
        best_match = "unknown"
        best_count = 0

        for threat_type, keywords in self.THREAT_TYPES.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > best_count:
                best_count = count
                best_match = threat_type

        return best_match if best_count > 0 else "unknown"

    def _assign_severity(self, score: float) -> str:
        """Map score to severity level."""
        if score >= self.SEVERITY_THRESHOLDS["critical"]:
            return "critical"
        elif score >= self.SEVERITY_THRESHOLDS["high"]:
            return "high"
        elif score >= self.SEVERITY_THRESHOLDS["medium"]:
            return "medium"
        elif score >= self.SEVERITY_THRESHOLDS["low"]:
            return "low"
        return "info"

    def get_info(self) -> Dict:
        """Return model metadata."""
        if not self._loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "n_features": self.metadata.get("n_features"),
            "n_samples": self.metadata.get("n_samples"),
            "test_f1": self.metadata.get("test_f1"),
            "test_roc_auc": self.metadata.get("test_roc_auc"),
            "model_type": self.metadata.get("model_type"),
            "trained_at": self.metadata.get("trained_at"),
        }
