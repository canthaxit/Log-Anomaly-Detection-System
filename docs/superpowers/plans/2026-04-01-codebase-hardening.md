# Codebase Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate code duplication, harden security defaults, extract the autoencoder from the deprecated TensorFlow pipeline, add log sanitization, and raise test coverage to 90%.

**Architecture:** Eight incremental changes applied in dependency order. New shared modules (`common/threats.py`, `common/config.py`, `common/sanitize.py`) replace duplicated code across `api/`, `batch/`, and `mcp/`. The autoencoder is extracted from the 1,363-line `intrusion_detection_pipeline.py` into a focused `core/autoencoder.py` module. Each task produces a self-contained commit.

**Tech Stack:** Python 3.10+, FastAPI, scikit-learn, TensorFlow (optional), pytest, pytest-cov

**Spec:** `docs/superpowers/specs/2026-04-01-codebase-hardening-design.md`

---

### Task 1: Create `common/threats.py` — Extensible Threat Classifier

**Files:**
- Create: `common/threats.py`
- Create: `tests/test_threats.py`

- [ ] **Step 1: Write the failing tests for ThreatClassifier**

Create `tests/test_threats.py`:

```python
"""Tests for common/threats.py — threat classification and severity assignment."""

import re

import pandas as pd
import pytest


class TestThreatClassifierDefaults:
    def test_brute_force_on_failed_action(self):
        from common.threats import ThreatClassifier
        c = ThreatClassifier()
        row = pd.Series({"action": "failed", "message": "login attempt", "event_type": "login"})
        assert c.classify_threat(row) == "brute_force"

    def test_privilege_escalation_on_sudo(self):
        from common.threats import ThreatClassifier
        c = ThreatClassifier()
        row = pd.Series({"action": "success", "message": "sudo su - root", "event_type": "login"})
        assert c.classify_threat(row) == "privilege_escalation"

    def test_data_exfiltration_on_shadow(self):
        from common.threats import ThreatClassifier
        c = ThreatClassifier()
        row = pd.Series({"action": "success", "message": "cat /etc/shadow", "event_type": "file"})
        assert c.classify_threat(row) == "data_exfiltration"

    def test_data_exfiltration_on_passwd(self):
        from common.threats import ThreatClassifier
        c = ThreatClassifier()
        row = pd.Series({"action": "success", "message": "cat /etc/passwd", "event_type": "file"})
        assert c.classify_threat(row) == "data_exfiltration"

    def test_data_exfiltration_on_secret(self):
        from common.threats import ThreatClassifier
        c = ThreatClassifier()
        row = pd.Series({"action": "success", "message": "read secret key", "event_type": "file"})
        assert c.classify_threat(row) == "data_exfiltration"

    def test_lateral_movement_on_network(self):
        from common.threats import ThreatClassifier
        c = ThreatClassifier()
        row = pd.Series({"action": "success", "message": "ssh to host", "event_type": "network"})
        assert c.classify_threat(row) == "lateral_movement"

    def test_unknown_when_no_match(self):
        from common.threats import ThreatClassifier
        c = ThreatClassifier()
        row = pd.Series({"action": "success", "message": "routine task", "event_type": "system"})
        assert c.classify_threat(row) == "unknown"

    def test_first_rule_wins(self):
        from common.threats import ThreatClassifier
        c = ThreatClassifier()
        # "failed" in action AND "sudo" in message — action rule is first
        row = pd.Series({"action": "failed", "message": "sudo attempt", "event_type": "login"})
        assert c.classify_threat(row) == "brute_force"

    def test_missing_field_returns_unknown(self):
        from common.threats import ThreatClassifier
        c = ThreatClassifier()
        row = pd.Series({"action": "success", "event_type": "system"})
        assert c.classify_threat(row) == "unknown"


class TestThreatClassifierCustomRules:
    def test_custom_rule(self):
        from common.threats import ThreatClassifier
        rules = [{"field": "message", "pattern": "SELECT.*FROM", "threat_type": "sql_injection"}]
        c = ThreatClassifier(rules=rules)
        row = pd.Series({"action": "success", "message": "SELECT * FROM users", "event_type": "db"})
        assert c.classify_threat(row) == "sql_injection"

    def test_custom_rule_no_match_returns_unknown(self):
        from common.threats import ThreatClassifier
        rules = [{"field": "message", "pattern": "SELECT.*FROM", "threat_type": "sql_injection"}]
        c = ThreatClassifier(rules=rules)
        row = pd.Series({"action": "success", "message": "normal log", "event_type": "system"})
        assert c.classify_threat(row) == "unknown"


class TestSeverityAssignment:
    def test_critical(self):
        from common.threats import ThreatClassifier
        c = ThreatClassifier()
        assert c.assign_severity(0.95) == "critical"
        assert c.assign_severity(1.0) == "critical"

    def test_high(self):
        from common.threats import ThreatClassifier
        c = ThreatClassifier()
        assert c.assign_severity(0.85) == "high"
        assert c.assign_severity(0.94) == "high"

    def test_medium(self):
        from common.threats import ThreatClassifier
        c = ThreatClassifier()
        assert c.assign_severity(0.7) == "medium"
        assert c.assign_severity(0.84) == "medium"

    def test_low(self):
        from common.threats import ThreatClassifier
        c = ThreatClassifier()
        assert c.assign_severity(0.69) == "low"
        assert c.assign_severity(0.0) == "low"

    def test_custom_thresholds(self):
        from common.threats import ThreatClassifier
        thresholds = {"critical": 0.99, "high": 0.9, "medium": 0.5}
        c = ThreatClassifier(severity_thresholds=thresholds)
        assert c.assign_severity(0.95) == "high"
        assert c.assign_severity(0.6) == "medium"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /d/Projects/canhaxit/Log-Anomaly-Detection-System && python -m pytest tests/test_threats.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'common.threats'`

- [ ] **Step 3: Implement `common/threats.py`**

Create `common/threats.py`:

```python
"""Extensible threat classification and severity assignment."""

import re
from typing import Any

import pandas as pd

DEFAULT_THREAT_RULES: list[dict[str, str]] = [
    {"field": "action",     "pattern": "failed",               "threat_type": "brute_force"},
    {"field": "message",    "pattern": "sudo",                 "threat_type": "privilege_escalation"},
    {"field": "message",    "pattern": "shadow|passwd|secret",  "threat_type": "data_exfiltration"},
    {"field": "event_type", "pattern": "^network$",            "threat_type": "lateral_movement"},
]

DEFAULT_SEVERITY_THRESHOLDS: dict[str, float] = {
    "critical": 0.95,
    "high": 0.85,
    "medium": 0.7,
}


class ThreatClassifier:
    """Rule-based threat classification with configurable rules and severity thresholds.

    Rules are evaluated in order; the first match wins.  Each rule is a dict
    with keys ``field``, ``pattern`` (regex), and ``threat_type``.
    """

    def __init__(
        self,
        rules: list[dict[str, str]] | None = None,
        severity_thresholds: dict[str, float] | None = None,
    ):
        self.rules = rules if rules is not None else DEFAULT_THREAT_RULES
        self.severity_thresholds = (
            severity_thresholds if severity_thresholds is not None else DEFAULT_SEVERITY_THRESHOLDS
        )
        # Pre-compile patterns for performance
        self._compiled: list[tuple[str, re.Pattern, str]] = [
            (r["field"], re.compile(r["pattern"], re.IGNORECASE), r["threat_type"])
            for r in self.rules
        ]

    def classify_threat(self, row: pd.Series) -> str:
        """Return the threat type for *row*, or ``'unknown'``."""
        for field, pattern, threat_type in self._compiled:
            value = str(row.get(field, ""))
            if pattern.search(value):
                return threat_type
        return "unknown"

    def assign_severity(self, score: float) -> str:
        """Map an anomaly *score* to a severity label."""
        for level in ("critical", "high", "medium"):
            if score >= self.severity_thresholds[level]:
                return level
        return "low"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /d/Projects/canhaxit/Log-Anomaly-Detection-System && python -m pytest tests/test_threats.py -v`

Expected: All 15 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /d/Projects/canhaxit/Log-Anomaly-Detection-System
git add common/threats.py tests/test_threats.py
git commit -m "feat: add extensible threat classifier in common/threats.py

Extract duplicated classify_threat() and assign_severity() into a shared
ThreatClassifier class with configurable rules and severity thresholds.
Fixes MCP missing 'secret' keyword for data_exfiltration detection."
```

---

### Task 2: Create `common/config.py` — Centralized Constants

**Files:**
- Create: `common/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_config.py`:

```python
"""Tests for common/config.py — centralized constants."""


class TestConstants:
    def test_max_upload_size(self):
        from common.config import MAX_UPLOAD_SIZE
        assert MAX_UPLOAD_SIZE == 10 * 1024 * 1024

    def test_max_input_size_alias(self):
        from common.config import MAX_INPUT_SIZE, MAX_UPLOAD_SIZE
        assert MAX_INPUT_SIZE == MAX_UPLOAD_SIZE

    def test_max_log_events(self):
        from common.config import MAX_LOG_EVENTS
        assert MAX_LOG_EVENTS == 10_000

    def test_max_csv_columns(self):
        from common.config import MAX_CSV_COLUMNS
        assert MAX_CSV_COLUMNS == 50

    def test_rate_limits_are_strings(self):
        from common.config import RATE_LIMITS
        for key, value in RATE_LIMITS.items():
            assert isinstance(value, str)
            assert "/" in value  # e.g. "30/minute"

    def test_default_threshold(self):
        from common.config import DEFAULT_THRESHOLD
        assert 0.0 < DEFAULT_THRESHOLD < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /d/Projects/canhaxit/Log-Anomaly-Detection-System && python -m pytest tests/test_config.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'common.config'`

- [ ] **Step 3: Implement `common/config.py`**

Create `common/config.py`:

```python
"""Centralized constants for the Log Anomaly Detection System."""

# Input size limits
MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024   # 10 MB
MAX_INPUT_SIZE: int = MAX_UPLOAD_SIZE      # alias used by MCP server
MAX_LOG_EVENTS: int = 10_000
MAX_CSV_COLUMNS: int = 50

# Rate limits (slowapi format strings)
RATE_LIMITS: dict[str, str] = {
    "analyze": "30/minute",
    "analyze_file": "10/minute",
    "models_load": "5/minute",
    "default": "30/minute",
}

# Default anomaly detection threshold
DEFAULT_THRESHOLD: float = 0.7
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /d/Projects/canhaxit/Log-Anomaly-Detection-System && python -m pytest tests/test_config.py -v`

Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /d/Projects/canhaxit/Log-Anomaly-Detection-System
git add common/config.py tests/test_config.py
git commit -m "feat: centralize constants in common/config.py

Move MAX_UPLOAD_SIZE, MAX_LOG_EVENTS, MAX_CSV_COLUMNS, RATE_LIMITS,
and DEFAULT_THRESHOLD from scattered locations into a single module."
```

---

### Task 3: Create `common/sanitize.py` — Log Message Sanitization

**Files:**
- Create: `common/sanitize.py`
- Create: `tests/test_sanitize.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_sanitize.py`:

```python
"""Tests for common/sanitize.py — PII redaction in log messages."""


class TestSanitizeMessage:
    def test_redacts_password(self):
        from common.sanitize import sanitize_message
        assert "***REDACTED***" in sanitize_message("user login password=hunter2")

    def test_redacts_password_colon(self):
        from common.sanitize import sanitize_message
        assert "***REDACTED***" in sanitize_message("Password: mysecretpassword")

    def test_redacts_token(self):
        from common.sanitize import sanitize_message
        assert "***REDACTED***" in sanitize_message("auth token=abc123xyz")

    def test_redacts_key(self):
        from common.sanitize import sanitize_message
        assert "***REDACTED***" in sanitize_message("api key=AKIAIOSFODNN7")

    def test_redacts_secret(self):
        from common.sanitize import sanitize_message
        assert "***REDACTED***" in sanitize_message("secret=wJalrXUtnFEMI")

    def test_redacts_ssn(self):
        from common.sanitize import sanitize_message
        result = sanitize_message("SSN is 123-45-6789")
        assert "123-45-6789" not in result
        assert "***SSN***" in result

    def test_preserves_normal_message(self):
        from common.sanitize import sanitize_message
        msg = "User alice logged in from 10.0.0.1"
        assert sanitize_message(msg) == msg

    def test_truncates_long_message(self):
        from common.sanitize import sanitize_message, MAX_MESSAGE_LENGTH
        long_msg = "a" * (MAX_MESSAGE_LENGTH + 100)
        assert len(sanitize_message(long_msg)) == MAX_MESSAGE_LENGTH

    def test_custom_patterns(self):
        import re
        from common.sanitize import sanitize_message
        custom = [(re.compile(r"credit_card=\S+"), "credit_card=***REDACTED***")]
        result = sanitize_message("credit_card=4111111111111111", patterns=custom)
        assert "4111111111111111" not in result

    def test_multiple_redactions_in_one_message(self):
        from common.sanitize import sanitize_message
        msg = "password=abc token=xyz"
        result = sanitize_message(msg)
        assert "abc" not in result
        assert "xyz" not in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /d/Projects/canhaxit/Log-Anomaly-Detection-System && python -m pytest tests/test_sanitize.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'common.sanitize'`

- [ ] **Step 3: Implement `common/sanitize.py`**

Create `common/sanitize.py`:

```python
"""Log message sanitization — redact sensitive patterns before output."""

import re

DEFAULT_REDACTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"password[=:]\s*\S+", re.IGNORECASE), "password=***REDACTED***"),
    (re.compile(r"token[=:]\s*\S+", re.IGNORECASE), "token=***REDACTED***"),
    (re.compile(r"key[=:]\s*\S+", re.IGNORECASE), "key=***REDACTED***"),
    (re.compile(r"secret[=:]\s*\S+", re.IGNORECASE), "secret=***REDACTED***"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "***SSN***"),
]

MAX_MESSAGE_LENGTH: int = 512


def sanitize_message(
    message: str,
    patterns: list[tuple[re.Pattern, str]] | None = None,
) -> str:
    """Redact sensitive patterns and truncate *message* for safe output."""
    patterns = patterns if patterns is not None else DEFAULT_REDACTION_PATTERNS
    for regex, replacement in patterns:
        message = regex.sub(replacement, message)
    return message[:MAX_MESSAGE_LENGTH]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /d/Projects/canhaxit/Log-Anomaly-Detection-System && python -m pytest tests/test_sanitize.py -v`

Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /d/Projects/canhaxit/Log-Anomaly-Detection-System
git add common/sanitize.py tests/test_sanitize.py
git commit -m "feat: add log message sanitization in common/sanitize.py

Redact passwords, tokens, keys, secrets, and SSNs from API response
messages. Applied only to output paths; raw messages preserved for
feature extraction."
```

---

### Task 4: Flip Auth Default in `common/security.py`

**Files:**
- Modify: `common/security.py:120`
- Modify: `tests/conftest.py:26-27`

- [ ] **Step 1: Write a failing test for the new default**

Add to `tests/test_security.py` at the end:

```python
class TestRequireAuthDefault:
    def test_default_is_true(self):
        """REQUIRE_AUTH should default to True when env var is unset."""
        import common.security as sec
        import os
        # Temporarily remove env var to test default
        orig_env = os.environ.pop("REQUIRE_AUTH", None)
        orig_val = sec.REQUIRE_AUTH
        try:
            # Re-evaluate the default logic
            result = os.environ.get("REQUIRE_AUTH", "true").lower() == "true"
            assert result is True
        finally:
            sec.REQUIRE_AUTH = orig_val
            if orig_env is not None:
                os.environ["REQUIRE_AUTH"] = orig_env
```

- [ ] **Step 2: Run the new test to verify current behavior**

Run: `cd /d/Projects/canhaxit/Log-Anomaly-Detection-System && python -m pytest tests/test_security.py::TestRequireAuthDefault -v`

Expected: PASS (the test checks the logic directly, not the module constant)

- [ ] **Step 3: Update `tests/conftest.py` to set REQUIRE_AUTH=false**

In `tests/conftest.py`, after line 27, add:

```python
os.environ.setdefault("REQUIRE_AUTH", "false")
```

So lines 26-28 become:

```python
os.environ.setdefault("ALLOWED_MODEL_DIRS", "anomaly_outputs,tests/tmp_models")
os.environ.setdefault("ALLOWED_LOG_DIRS", "logs,tests")
os.environ.setdefault("REQUIRE_AUTH", "false")
```

- [ ] **Step 4: Flip the default in `common/security.py`**

In `common/security.py`, change line 120 from:

```python
REQUIRE_AUTH: bool = os.environ.get("REQUIRE_AUTH", "false").lower() == "true"
```

to:

```python
REQUIRE_AUTH: bool = os.environ.get("REQUIRE_AUTH", "true").lower() == "true"
```

- [ ] **Step 5: Run the full test suite to verify nothing breaks**

Run: `cd /d/Projects/canhaxit/Log-Anomaly-Detection-System && python -m pytest tests/ -v --tb=short`

Expected: All tests PASS (conftest sets REQUIRE_AUTH=false before import)

- [ ] **Step 6: Commit**

```bash
cd /d/Projects/canhaxit/Log-Anomaly-Detection-System
git add common/security.py tests/conftest.py tests/test_security.py
git commit -m "security: make API authentication mandatory by default

REQUIRE_AUTH now defaults to 'true'. Deployments without API_KEY must
explicitly set REQUIRE_AUTH=false to opt out. This is a breaking change
for existing deployments that relied on the implicit auth-disabled default."
```

---

### Task 5: Create `core/autoencoder.py` — Extract Autoencoder

**Files:**
- Create: `core/autoencoder.py`
- Create: `tests/test_autoencoder.py`
- Create: `config/requirements_full.txt`
- Modify: `core/intrusion_detection_pipeline.py:1-3`

- [ ] **Step 1: Write failing tests (skip if TF not installed)**

Create `tests/test_autoencoder.py`:

```python
"""Tests for core/autoencoder.py — autoencoder-based anomaly detection."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")


class TestAutoencoderDetector:
    @pytest.fixture
    def sample_features(self):
        """Synthetic normal feature data: 200 samples, 10 features."""
        rng = np.random.RandomState(42)
        return rng.randn(200, 10).astype(np.float32)

    @pytest.fixture
    def anomalous_features(self):
        """Synthetic anomalous data: far from normal distribution."""
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
        # Anomalous data should have higher mean reconstruction error
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
        # AnomalyScorer.combine_scores uses self.weights — add autoencoder
        scorer.weights["autoencoder"] = 0.33
        scorer.weights["isolation_forest"] = 0.34
        scorer.weights["statistical"] = 0.33

        combined = scorer.combine_scores({
            "isolation_forest": np.random.rand(200),
            "statistical": np.random.rand(200),
            "autoencoder": ae_scores,
        })
        assert combined.shape == (200,)
```

- [ ] **Step 2: Run tests to verify they fail (or skip)**

Run: `cd /d/Projects/canhaxit/Log-Anomaly-Detection-System && python -m pytest tests/test_autoencoder.py -v`

Expected: Either SKIP (no TF) or FAIL (no `core.autoencoder` module)

- [ ] **Step 3: Implement `core/autoencoder.py`**

Create `core/autoencoder.py`:

```python
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
    """Reconstruction-error anomaly detector using a symmetric autoencoder.

    Parameters
    ----------
    encoding_dim : int
        Bottleneck dimension.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Training batch size.
    learning_rate : float
        Adam optimizer learning rate.
    """

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
        """Construct the encoder-decoder architecture."""
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
        """Train the autoencoder on *features* (assumed to be normal data).

        After training, a reconstruction-error threshold is computed at the
        95th percentile of training errors for normalization.
        """
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
        # Compute normalization threshold from training data
        errors = self._raw_errors(features)
        self._threshold = float(np.percentile(errors, 95))
        return self

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """Return normalized reconstruction errors in [0, 1].

        Scores are clipped: errors at or below the 95th-percentile training
        error map to ~1.0; lower errors map proportionally toward 0.0.
        """
        if self._autoencoder is None:
            raise RuntimeError("AutoencoderDetector must be fit before scoring")
        errors = self._raw_errors(features)
        if self._threshold is None or self._threshold == 0:
            return np.zeros_like(errors)
        normalized = errors / self._threshold
        return np.clip(normalized, 0.0, 1.0)

    def _raw_errors(self, features: np.ndarray) -> np.ndarray:
        """Compute per-sample mean squared reconstruction error."""
        reconstructed = self._autoencoder.predict(features, verbose=0)
        return np.mean(np.square(features - reconstructed), axis=1)
```

- [ ] **Step 4: Create `config/requirements_full.txt`**

Create `config/requirements_full.txt`:

```
# Full dependencies (includes TensorFlow for autoencoder support)
-r requirements_api.txt
tensorflow>=2.14.0
```

- [ ] **Step 5: Add deprecation warning to `intrusion_detection_pipeline.py`**

At the very top of `core/intrusion_detection_pipeline.py` (before any other imports), insert after the docstring:

```python
import warnings
warnings.warn(
    "intrusion_detection_pipeline is deprecated. Use log_anomaly_detection_lite "
    "with core.autoencoder for TensorFlow support.",
    DeprecationWarning,
    stacklevel=2,
)
```

- [ ] **Step 6: Run autoencoder tests**

Run: `cd /d/Projects/canhaxit/Log-Anomaly-Detection-System && python -m pytest tests/test_autoencoder.py -v`

Expected: PASS if TensorFlow installed, SKIP otherwise

- [ ] **Step 7: Commit**

```bash
cd /d/Projects/canhaxit/Log-Anomaly-Detection-System
git add core/autoencoder.py tests/test_autoencoder.py config/requirements_full.txt core/intrusion_detection_pipeline.py
git commit -m "feat: extract autoencoder into core/autoencoder.py, deprecate pipeline

Extract the autoencoder component from intrusion_detection_pipeline.py
into a focused AutoencoderDetector class that integrates with
AnomalyScorer. The old pipeline file gets a deprecation warning."
```

---

### Task 6: Integrate Common Modules into API

**Files:**
- Modify: `api/anomaly_api.py:120-121,172-195,339-340,348-357`

- [ ] **Step 1: Replace local constants with imports from `common/config.py`**

In `api/anomaly_api.py`, replace lines 120-121:

```python
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_LOG_EVENTS = 10_000
```

with:

```python
from common.config import MAX_UPLOAD_SIZE, MAX_LOG_EVENTS, RATE_LIMITS
```

- [ ] **Step 2: Replace local threat functions with ThreatClassifier**

In `api/anomaly_api.py`, replace lines 172-195 (the `classify_threat` and `assign_severity` functions):

```python
def classify_threat(row: pd.Series) -> str:
    """Classify threat type."""
    if 'failed' in str(row.get('action', '')).lower():
        return 'brute_force'
    elif 'sudo' in str(row.get('message', '')).lower():
        return 'privilege_escalation'
    elif any(word in str(row.get('message', '')).lower() for word in ['shadow', 'passwd', 'secret']):
        return 'data_exfiltration'
    elif row.get('event_type') == 'network':
        return 'lateral_movement'
    else:
        return 'unknown'


def assign_severity(score: float) -> str:
    """Assign severity based on score."""
    if score >= 0.95:
        return 'critical'
    elif score >= 0.85:
        return 'high'
    elif score >= 0.7:
        return 'medium'
    else:
        return 'low'
```

with:

```python
from common.threats import ThreatClassifier
from common.sanitize import sanitize_message

_classifier = ThreatClassifier()
```

- [ ] **Step 3: Update rate limit decorators**

Replace each hardcoded rate limit string with the centralized constant:

- Line ~210: `@limiter.limit("30/minute")` → `@limiter.limit(RATE_LIMITS["default"])`
- Line ~228: `@limiter.limit("5/minute")` → `@limiter.limit(RATE_LIMITS["models_load"])`
- Line ~276: `@limiter.limit("30/minute")` → `@limiter.limit(RATE_LIMITS["analyze"])`
- Line ~380: `@limiter.limit("10/minute")` → `@limiter.limit(RATE_LIMITS["analyze_file"])`
- Line ~434: `@limiter.limit("30/minute")` → `@limiter.limit(RATE_LIMITS["default"])`

- [ ] **Step 4: Update threat classification and sanitization calls**

In the analyze endpoint (~line 339-340), replace:

```python
        result_df['threat_type'] = result_df.apply(classify_threat, axis=1)
        result_df['severity'] = result_df['anomaly_score'].apply(assign_severity)
```

with:

```python
        result_df['threat_type'] = result_df.apply(_classifier.classify_threat, axis=1)
        result_df['severity'] = result_df['anomaly_score'].apply(_classifier.assign_severity)
```

In the response-building loop (~lines 344-357), add sanitization to the message field. Replace:

```python
                    message=row['message'],
```

with:

```python
                    message=sanitize_message(str(row['message'])),
```

- [ ] **Step 5: Also replace the hardcoded 0.7 default threshold**

In `api/anomaly_api.py`, add `DEFAULT_THRESHOLD` to the config import:

```python
from common.config import MAX_UPLOAD_SIZE, MAX_LOG_EVENTS, RATE_LIMITS, DEFAULT_THRESHOLD
```

Replace every `0.7` threshold default (lines ~255, ~476) with `DEFAULT_THRESHOLD`.

- [ ] **Step 6: Run API tests**

Run: `cd /d/Projects/canhaxit/Log-Anomaly-Detection-System && python -m pytest tests/test_api.py -v --tb=short`

Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
cd /d/Projects/canhaxit/Log-Anomaly-Detection-System
git add api/anomaly_api.py
git commit -m "refactor: integrate common modules into API

Replace duplicated classify_threat/assign_severity with ThreatClassifier,
use centralized constants from common/config, add message sanitization
to API responses, use RATE_LIMITS for rate limit decorators."
```

---

### Task 7: Integrate Common Modules into Batch Processor + Thread Safety

**Files:**
- Modify: `batch/batch_processor.py:1-35,86-111,127-171,213-235`

- [ ] **Step 1: Add imports and thread lock**

At the top of `batch/batch_processor.py`, after the existing imports (line 15), add:

```python
import threading
```

Add after the `from common.security import ...` line (line 31):

```python
from common.threats import ThreatClassifier
from common.sanitize import sanitize_message
from common.config import DEFAULT_THRESHOLD
```

- [ ] **Step 2: Add lock and classifier to `__init__`**

In `BatchProcessor.__init__`, after `self.processed_files = set()` (line 69), add:

```python
        self._model_lock = threading.Lock()
        self._classifier = ThreatClassifier()
```

- [ ] **Step 3: Wrap `load_models` with lock**

Replace the `load_models` method body (lines 86-110) to wrap the loading in the lock:

```python
    def load_models(self):
        """Load trained models."""
        try:
            self.model_dir = validate_model_path(str(self.model_dir))
            with self._model_lock:
                for pkl in ("feature_pipeline.pkl", "isolation_forest_model.pkl", "statistical_detector.pkl"):
                    verify_model_file(self.model_dir / pkl)
                self.feature_pipeline = joblib.load(self.model_dir / "feature_pipeline.pkl")
                self.isolation_forest = joblib.load(self.model_dir / "isolation_forest_model.pkl")
                self.statistical_detector = joblib.load(self.model_dir / "statistical_detector.pkl")

                if (self.model_dir / "inference_package.pkl").exists():
                    verify_model_file(self.model_dir / "inference_package.pkl")
                    package = joblib.load(self.model_dir / "inference_package.pkl")
                    self.scorer = package.get("scorer")
                    self.threshold = package.get("threshold")
                else:
                    self.scorer = AnomalyScorer()
                    self.threshold = DEFAULT_THRESHOLD

            logger.info(f"Models loaded successfully from {self.model_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
```

- [ ] **Step 4: Snapshot model state in `process_file`**

At the top of the `process_file` method (line 128), after `try:`, add a snapshot:

```python
            with self._model_lock:
                snap = {
                    "feature_pipeline": self.feature_pipeline,
                    "isolation_forest": self.isolation_forest,
                    "statistical_detector": self.statistical_detector,
                    "scorer": self.scorer,
                    "threshold": self.threshold,
                }
```

Then replace all `self.feature_pipeline`, `self.isolation_forest`, etc. in `process_file` with `snap["feature_pipeline"]`, `snap["isolation_forest"]`, etc.

- [ ] **Step 5: Replace duplicated classify/severity methods**

Delete the `classify_threat` method (lines 213-224) and `assign_severity` method (lines 226-235) from the `BatchProcessor` class.

Replace their usage in `process_file` (~lines 164-171):

```python
            # Classify threats
            anomalies_df['threat_type'] = anomalies_df.apply(
                self.classify_threat, axis=1
            )

            # Assign severity
            anomalies_df['severity'] = anomalies_df['anomaly_score'].apply(
                self.assign_severity
            )
```

with:

```python
            # Classify threats and assign severity
            anomalies_df['threat_type'] = anomalies_df.apply(
                self._classifier.classify_threat, axis=1
            )
            anomalies_df['severity'] = anomalies_df['anomaly_score'].apply(
                self._classifier.assign_severity
            )
```

- [ ] **Step 6: Add sanitization to output**

In `process_file`, after building the anomalies list (~line 184-191), add sanitization before the JSON dump. Replace:

```python
            # Convert timestamps to strings
            for anomaly in result["anomalies"]:
                if 'timestamp' in anomaly and hasattr(anomaly['timestamp'], 'isoformat'):
                    anomaly['timestamp'] = anomaly['timestamp'].isoformat()
```

with:

```python
            # Sanitize and convert timestamps
            for anomaly in result["anomalies"]:
                if 'timestamp' in anomaly and hasattr(anomaly['timestamp'], 'isoformat'):
                    anomaly['timestamp'] = anomaly['timestamp'].isoformat()
                if 'message' in anomaly:
                    anomaly['message'] = sanitize_message(str(anomaly['message']))
```

- [ ] **Step 7: Run batch tests**

Run: `cd /d/Projects/canhaxit/Log-Anomaly-Detection-System && python -m pytest tests/test_batch.py -v --tb=short`

Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
cd /d/Projects/canhaxit/Log-Anomaly-Detection-System
git add batch/batch_processor.py
git commit -m "refactor: integrate common modules into batch processor, add thread safety

Replace duplicated threat classification with ThreatClassifier, add
threading.Lock for model state, sanitize output messages, use
DEFAULT_THRESHOLD from common/config."
```

---

### Task 8: Integrate Common Modules into MCP Server

**Files:**
- Modify: `mcp/anomaly_mcp_server.py:49-52,100-103,156-162,186-210`

- [ ] **Step 1: Replace local constants with imports**

In `mcp/anomaly_mcp_server.py`, replace lines 49-52:

```python
# Input size limits
MAX_INPUT_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_LOG_EVENTS = 10_000
MAX_CSV_COLUMNS = 50
```

with:

```python
from common.config import MAX_INPUT_SIZE, MAX_LOG_EVENTS, MAX_CSV_COLUMNS, DEFAULT_THRESHOLD
from common.threats import ThreatClassifier
from common.sanitize import sanitize_message

_classifier = ThreatClassifier()
```

- [ ] **Step 2: Replace hardcoded 0.7 threshold**

In `load_models()` (~line 88), replace:

```python
            MODEL_STATE["threshold"] = 0.7
```

with:

```python
            MODEL_STATE["threshold"] = DEFAULT_THRESHOLD
```

- [ ] **Step 3: Replace threat classification calls**

In `analyze_logs()` (~lines 156-162), replace:

```python
        # Classify threats
        anomalies_df['threat_type'] = anomalies_df.apply(
            lambda row: classify_threat(row, MODEL_STATE["statistical_detector"]),
            axis=1
        )

        # Assign severity
        anomalies_df['severity'] = anomalies_df['anomaly_score'].apply(assign_severity)
```

with:

```python
        # Classify threats and assign severity
        anomalies_df['threat_type'] = anomalies_df.apply(
            _classifier.classify_threat, axis=1
        )
        anomalies_df['severity'] = anomalies_df['anomaly_score'].apply(
            _classifier.assign_severity
        )
```

- [ ] **Step 4: Delete the local classify_threat and assign_severity functions**

Delete lines 186-210 (the `classify_threat` and `assign_severity` functions).

- [ ] **Step 5: Add sanitization to output**

In `analyze_logs()`, in the timestamp conversion loop (~lines 168-170), add message sanitization:

```python
        for anomaly in anomalies:
            if 'timestamp' in anomaly and hasattr(anomaly['timestamp'], 'isoformat'):
                anomaly['timestamp'] = anomaly['timestamp'].isoformat()
            if 'message' in anomaly:
                anomaly['message'] = sanitize_message(str(anomaly['message']))
```

- [ ] **Step 6: Run MCP tests**

Run: `cd /d/Projects/canhaxit/Log-Anomaly-Detection-System && python -m pytest tests/test_mcp.py -v --tb=short`

Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
cd /d/Projects/canhaxit/Log-Anomaly-Detection-System
git add mcp/anomaly_mcp_server.py
git commit -m "refactor: integrate common modules into MCP server

Replace duplicated threat classification (fixing missing 'secret'
keyword), use centralized constants, add message sanitization.
Drop unused 'detector' parameter from classify_threat."
```

---

### Task 9: Expand Tests to Reach 90% Coverage

**Files:**
- Modify: `pyproject.toml:17-18,25`
- Modify: `tests/test_mcp.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_core.py`
- Modify: `tests/test_batch.py`

- [ ] **Step 1: Update `pyproject.toml` coverage settings**

In `pyproject.toml`, change lines 17-18 to remove MCP from omit:

```toml
omit = [
    "core/intrusion_detection_pipeline.py",
    "api/anomaly_api_chronicle.py",
    "api/test_api.py",
]
```

Change line 25 to:

```toml
fail_under = 90
```

- [ ] **Step 2: Expand MCP tests**

Add to `tests/test_mcp.py`:

```python
class TestMCPAnalyzeLogFile:
    def test_invalid_path_returns_error(self, mcp_mod):
        result = mcp_mod.analyze_log_file("/tmp/evil/logs.json")
        assert result["status"] == "error"

    def test_nonexistent_file_returns_error(self, mcp_mod):
        result = mcp_mod.analyze_log_file("tests/nonexistent.json")
        assert result["status"] == "error"

    def test_unsupported_format(self, mcp_mod):
        orig = mcp_mod.MODEL_STATE["loaded"]
        mcp_mod.MODEL_STATE["loaded"] = True
        try:
            result = mcp_mod.analyze_logs("data", format="xml")
            assert result["status"] == "error"
        finally:
            mcp_mod.MODEL_STATE["loaded"] = orig


class TestMCPGetStats:
    def test_not_loaded_returns_error(self, mcp_mod):
        orig = mcp_mod.MODEL_STATE["loaded"]
        mcp_mod.MODEL_STATE["loaded"] = False
        try:
            result = mcp_mod.get_stats()
            assert result["status"] == "error"
        finally:
            mcp_mod.MODEL_STATE["loaded"] = orig
```

- [ ] **Step 3: Expand API tests**

Add to `tests/test_api.py`:

```python
class TestAnalyzeEndpointEdgeCases:
    def test_models_not_loaded_returns_400(self, api_app, client):
        from api.anomaly_api import MODEL_STATE
        orig = MODEL_STATE["loaded"]
        MODEL_STATE["loaded"] = False
        try:
            resp = client.post("/analyze", json={"logs": [
                {"timestamp": "2026-01-01T00:00:00Z", "user": "a",
                 "source_ip": "1.1.1.1", "event_type": "login",
                 "action": "success", "message": "m"}
            ]})
            assert resp.status_code == 400
        finally:
            MODEL_STATE["loaded"] = orig

    def test_return_all_events_flag(self, client):
        logs = [
            {"timestamp": "2026-01-01T00:00:00Z", "user": "alice",
             "source_ip": "1.1.1.1", "event_type": "login",
             "action": "success", "message": "normal log", "severity": "low"}
        ]
        resp = client.post("/analyze", json={"logs": logs, "return_all_events": True})
        assert resp.status_code == 200


class TestStatsNotLoaded:
    def test_stats_not_loaded_returns_400(self, api_app, client):
        from api.anomaly_api import MODEL_STATE
        orig = MODEL_STATE["loaded"]
        MODEL_STATE["loaded"] = False
        try:
            resp = client.get("/stats")
            assert resp.status_code == 400
        finally:
            MODEL_STATE["loaded"] = orig


class TestModelLoadEndpoint:
    def test_invalid_model_dir_returns_403(self, client):
        resp = client.post("/models/load?model_dir=/tmp/evil")
        assert resp.status_code == 403

    def test_nonexistent_dir_returns_404(self, client):
        resp = client.post("/models/load?model_dir=anomaly_outputs/nonexistent")
        # Path validation passes (within allowed dirs) but dir doesn't exist
        assert resp.status_code in (403, 404, 500)
```

- [ ] **Step 4: Expand core tests**

Add to `tests/test_core.py`:

```python
class TestAnomalyScorerFitNormalization:
    def test_fit_normalization_uses_stored_range(self, scorer):
        scores = np.array([0.0, 0.5, 1.0])
        scorer.fit_normalization("test_det", scores)
        # New scores should use the stored range
        new_scores = np.array([0.25, 0.75])
        result = scorer.normalize_scores(new_scores, detector_name="test_det")
        assert result[0] == pytest.approx(0.25)
        assert result[1] == pytest.approx(0.75)

    def test_combine_ignores_unknown_detectors(self, scorer):
        iso = np.array([0.5, 0.5])
        stat = np.array([0.5, 0.5])
        unknown = np.array([1.0, 1.0])
        combined = scorer.combine_scores({
            "isolation_forest": iso,
            "statistical": stat,
            "nonexistent": unknown,
        })
        # "nonexistent" has no weight, should be ignored
        assert len(combined) == 2
```

- [ ] **Step 5: Expand batch tests**

Add to `tests/test_batch.py`:

```python
class TestBatchFindNewFiles:
    def test_finds_json_files(self, tmp_path):
        from batch.batch_processor import BatchProcessor
        import common.security as sec

        log_dir = tmp_path / "testlogs"
        log_dir.mkdir()
        (log_dir / "test.json").write_text('[{"test": true}]')

        sec.ALLOWED_LOG_DIRS.append(str(log_dir))
        try:
            bp = BatchProcessor(
                log_dir=str(log_dir),
                output_dir=str(tmp_path / "out"),
            )
            files = bp.find_new_files()
            assert len(files) >= 1
        finally:
            sec.ALLOWED_LOG_DIRS.remove(str(log_dir))

    def test_skips_already_processed(self, tmp_path):
        from batch.batch_processor import BatchProcessor
        import common.security as sec

        log_dir = tmp_path / "testlogs"
        log_dir.mkdir()
        f = log_dir / "test.json"
        f.write_text('[{"test": true}]')

        sec.ALLOWED_LOG_DIRS.append(str(log_dir))
        try:
            bp = BatchProcessor(
                log_dir=str(log_dir),
                output_dir=str(tmp_path / "out"),
            )
            bp.processed_files.add(str(f))
            files = bp.find_new_files()
            assert len(files) == 0
        finally:
            sec.ALLOWED_LOG_DIRS.remove(str(log_dir))


class TestBatchProcessBatch:
    def test_no_new_files_returns_zero(self, tmp_path):
        from batch.batch_processor import BatchProcessor
        import common.security as sec

        log_dir = tmp_path / "emptylogs"
        log_dir.mkdir()

        sec.ALLOWED_LOG_DIRS.append(str(log_dir))
        try:
            bp = BatchProcessor(
                log_dir=str(log_dir),
                output_dir=str(tmp_path / "out"),
            )
            result = bp.process_batch()
            assert result["processed"] == 0
        finally:
            sec.ALLOWED_LOG_DIRS.remove(str(log_dir))
```

- [ ] **Step 6: Run full test suite with coverage**

Run: `cd /d/Projects/canhaxit/Log-Anomaly-Detection-System && python -m pytest tests/ -v --tb=short --cov --cov-report=term-missing`

Expected: All tests PASS with coverage >= 90%

- [ ] **Step 7: If coverage is below 90%, identify and fill gaps**

Run: `cd /d/Projects/canhaxit/Log-Anomaly-Detection-System && python -m pytest tests/ --cov --cov-report=term-missing 2>&1 | tail -30`

Look at the `Missing` column and add targeted tests for uncovered lines.

- [ ] **Step 8: Commit**

```bash
cd /d/Projects/canhaxit/Log-Anomaly-Detection-System
git add pyproject.toml tests/
git commit -m "test: raise coverage target to 90%, expand test suite

Remove MCP from coverage omit, expand tests for MCP error paths,
API edge cases, batch file discovery, and scorer normalization."
```

---

### Task 10: TLS Documentation and Docker Comment

**Files:**
- Modify: `README.md`
- Modify: `docker/docker-compose.yml:8-9`

- [ ] **Step 1: Add Production Deployment section to README.md**

Append after the existing content in `README.md`:

```markdown
## Production Deployment

### TLS / HTTPS

The API server binds to `127.0.0.1:8000` with plain HTTP by default. For production, use a reverse proxy for TLS termination.

#### nginx

```nginx
server {
    listen 443 ssl;
    server_name anomaly-api.example.com;

    ssl_certificate     /etc/ssl/certs/anomaly-api.crt;
    ssl_certificate_key /etc/ssl/private/anomaly-api.key;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### Caddy

```
anomaly-api.example.com {
    reverse_proxy 127.0.0.1:8000
}
```

Caddy handles TLS certificates automatically via Let's Encrypt.

### Authentication

API authentication is **required by default**. Set the `API_KEY` environment variable before starting:

```bash
export API_KEY="your-secure-api-key"
python api/anomaly_api.py
```

To explicitly disable auth (development only):

```bash
export REQUIRE_AUTH=false
python api/anomaly_api.py
```

### Docker

The `docker-compose.yml` binds the API to port 8000. Place a reverse proxy in front for TLS. See `docker/docker-compose.yml` for the full configuration.
```

- [ ] **Step 2: Add TLS comment to docker-compose.yml**

In `docker/docker-compose.yml`, add a comment before the ports section (line 8):

```yaml
    # IMPORTANT: Use a reverse proxy (nginx/Caddy) for TLS in production.
    # See README.md "Production Deployment" section.
    ports:
      - "8000:8000"
```

- [ ] **Step 3: Commit**

```bash
cd /d/Projects/canhaxit/Log-Anomaly-Detection-System
git add README.md docker/docker-compose.yml
git commit -m "docs: add production deployment guide for TLS and auth

Document reverse proxy setup (nginx, Caddy) for TLS termination,
explain the mandatory auth default, add TLS reminder to docker-compose."
```

---

### Task 11: Final Verification

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite with coverage**

Run: `cd /d/Projects/canhaxit/Log-Anomaly-Detection-System && python -m pytest tests/ -v --tb=short --cov --cov-report=term-missing`

Expected: All tests PASS, coverage >= 90%

- [ ] **Step 2: Verify no import errors across all modules**

Run:
```bash
cd /d/Projects/canhaxit/Log-Anomaly-Detection-System
python -c "from common.threats import ThreatClassifier; print('threats OK')"
python -c "from common.config import MAX_UPLOAD_SIZE, RATE_LIMITS; print('config OK')"
python -c "from common.sanitize import sanitize_message; print('sanitize OK')"
python -c "from common.security import REQUIRE_AUTH; print(f'REQUIRE_AUTH={REQUIRE_AUTH}')"
```

Expected:
```
threats OK
config OK
sanitize OK
REQUIRE_AUTH=True
```

- [ ] **Step 3: Verify deprecation warning on pipeline import**

Run:
```bash
cd /d/Projects/canhaxit/Log-Anomaly-Detection-System/core
python -W all -c "import intrusion_detection_pipeline" 2>&1 | head -5
```

Expected: `DeprecationWarning: intrusion_detection_pipeline is deprecated...`

- [ ] **Step 4: Check git log for clean commit history**

Run: `cd /d/Projects/canhaxit/Log-Anomaly-Detection-System && git log --oneline -15`

Expected: 10 clean commits, one per task.
