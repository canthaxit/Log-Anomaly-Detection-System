# Codebase Hardening & Quality Improvements

**Date:** 2026-04-01
**Status:** Approved
**Approach:** Incremental refactor (one commit per change, dependency order)

---

## Overview

Eight targeted improvements to the Log Anomaly Detection System addressing code duplication, security defaults, test coverage, and the unused TensorFlow pipeline. Each change is independently reviewable and revertible.

---

## 1. Extensible Threat Classification (`common/threats.py`)

### Problem
`classify_threat()` and `assign_severity()` are duplicated in three files:
- `api/anomaly_api.py:172-195`
- `batch/batch_processor.py:213-235`
- `mcp/anomaly_mcp_server.py:186-210`

The MCP version has a divergent signature (unused `detector` param) and misses the `secret` keyword for `data_exfiltration`.

### Design

New module `common/threats.py` with a `ThreatClassifier` class:

```python
DEFAULT_THREAT_RULES = [
    {"field": "action",     "pattern": "failed",              "threat_type": "brute_force"},
    {"field": "message",    "pattern": "sudo",                "threat_type": "privilege_escalation"},
    {"field": "message",    "pattern": "shadow|passwd|secret", "threat_type": "data_exfiltration"},
    {"field": "event_type", "pattern": "^network$",           "threat_type": "lateral_movement"},
]

DEFAULT_SEVERITY_THRESHOLDS = {
    "critical": 0.95,
    "high": 0.85,
    "medium": 0.7,
}

class ThreatClassifier:
    def __init__(self, rules=None, severity_thresholds=None):
        self.rules = rules or DEFAULT_THREAT_RULES
        self.severity_thresholds = severity_thresholds or DEFAULT_SEVERITY_THRESHOLDS

    def classify_threat(self, row: pd.Series) -> str:
        """Check rules in order, return first matching threat_type or 'unknown'."""
        for rule in self.rules:
            value = str(row.get(rule["field"], "")).lower()
            if re.search(rule["pattern"], value):
                return rule["threat_type"]
        return "unknown"

    def assign_severity(self, score: float) -> str:
        """Walk thresholds descending, return first match or 'low'."""
        for level in ("critical", "high", "medium"):
            if score >= self.severity_thresholds[level]:
                return level
        return "low"
```

### Consumers

All three modules replace local functions with:
```python
from common.threats import ThreatClassifier
classifier = ThreatClassifier()
```

Custom rules can be passed at init for extensibility.

### Fixes
- MCP `data_exfiltration` now includes `secret` keyword (was missing)
- MCP `classify_threat` drops unused `detector` parameter

---

## 2. Mandatory Auth by Default (`common/security.py`)

### Problem
`REQUIRE_AUTH` defaults to `"false"`. For a security tool, auth should be on by default.

### Design

Change line 120 of `common/security.py`:
```python
# Before
REQUIRE_AUTH: bool = os.environ.get("REQUIRE_AUTH", "false").lower() == "true"

# After
REQUIRE_AUTH: bool = os.environ.get("REQUIRE_AUTH", "true").lower() == "true"
```

### Behavior

| API_KEY set | REQUIRE_AUTH | Result |
|-------------|-------------|--------|
| Yes | unset (default true) | Auth enabled |
| No | unset (default true) | **RuntimeError at startup** |
| No | `false` | Auth disabled, warning logged |
| Yes | `false` | Auth disabled, warning logged |

### Breaking change
Existing deployments without `API_KEY` will fail at startup. They must either set `API_KEY` or explicitly set `REQUIRE_AUTH=false`.

### Test impact
`tests/conftest.py` must set `os.environ["REQUIRE_AUTH"] = "false"` before importing `common.security`. This is the simplest approach — no test API keys to manage. The existing `os.environ.setdefault()` block in conftest.py (line 26-27) is the right place to add this.

### Docker
`docker/docker-compose.yml` already passes both env vars. No change needed.

---

## 3. Centralized Constants (`common/config.py`)

### Problem
Magic numbers scattered across `api/anomaly_api.py:120-121`, `mcp/anomaly_mcp_server.py:50-52`, and rate limit strings hardcoded in decorators.

### Design

New module `common/config.py`:

```python
# Input limits
MAX_UPLOAD_SIZE = 10 * 1024 * 1024   # 10 MB
MAX_INPUT_SIZE = MAX_UPLOAD_SIZE      # alias for MCP
MAX_LOG_EVENTS = 10_000
MAX_CSV_COLUMNS = 50

# Rate limits (slowapi format strings)
RATE_LIMITS = {
    "analyze": "30/minute",
    "analyze_file": "10/minute",
    "models_load": "5/minute",
    "default": "30/minute",
}

# Default detection threshold
DEFAULT_THRESHOLD = 0.7
```

Severity thresholds live in `common/threats.py` (Section 1) since they're part of threat classification. The `Config` class in `core/log_anomaly_detection_lite.py` imports defaults from `common/threats.py`.

### Consumers

```python
from common.config import MAX_UPLOAD_SIZE, MAX_LOG_EVENTS, RATE_LIMITS
```

Rate limit decorators change from `@limiter.limit("30/minute")` to `@limiter.limit(RATE_LIMITS["analyze"])`.

---

## 4. Autoencoder Extraction & Pipeline Deprecation

### Problem
`core/intrusion_detection_pipeline.py` is 1,363 lines, excluded from tests, requires TensorFlow, and duplicates 80% of the lite version. Its only unique value is the autoencoder.

### Design

#### 4a. Extract `core/autoencoder.py`

```python
class AutoencoderDetector:
    def __init__(self, encoding_dim=16, epochs=50, batch_size=1024):
        ...

    def fit(self, features: np.ndarray) -> "AutoencoderDetector":
        """Build & train autoencoder, compute reconstruction error threshold."""
        ...

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """Return normalized reconstruction errors (0-1 range)."""
        ...
```

- TensorFlow imported at the top of this module only
- Integrates with `AnomalyScorer.combine_scores()` as a third signal
- Optional: only used when explicitly requested

#### 4b. Deprecate `intrusion_detection_pipeline.py`

Add a module-level deprecation warning:
```python
import warnings
warnings.warn(
    "intrusion_detection_pipeline is deprecated. Use log_anomaly_detection_lite "
    "with core.autoencoder for TensorFlow support.",
    DeprecationWarning, stacklevel=2
)
```

File stays in repo for backward compatibility but remains in coverage omit list.

#### 4c. New dependency file

`config/requirements_full.txt` extends `requirements_api.txt` with `tensorflow>=2.14.0`.

#### 4d. Tests

`tests/test_autoencoder.py` with `@pytest.mark.skipif(no tensorflow)`. Tests fit, score, and integration with `AnomalyScorer`.

---

## 5. Coverage Target 90%

### Changes to `pyproject.toml`

```toml
[tool.coverage.run]
source = ["core", "common", "api", "batch", "mcp"]
omit = [
    "core/intrusion_detection_pipeline.py",
    "api/anomaly_api_chronicle.py",
    "api/test_api.py",
]
# mcp/anomaly_mcp_server.py REMOVED from omit list

[tool.coverage.report]
fail_under = 90
```

### New/expanded test files

| File | Coverage target |
|------|----------------|
| `tests/test_threats.py` | `common/threats.py` — default rules, custom rules, severity, edge cases |
| `tests/test_config.py` | `common/config.py` — constants sanity checks |
| `tests/test_sanitize.py` | `common/sanitize.py` — redaction patterns, truncation |
| `tests/test_autoencoder.py` | `core/autoencoder.py` — skip without TF |
| `tests/test_mcp.py` (expand) | `analyze_log_file()`, `get_stats()`, error paths, size limits |
| `tests/test_core.py` (expand) | Fill remaining gaps |
| `tests/test_api.py` (expand) | Fill remaining gaps |

### CI

`.github/workflows/ci.yml` keeps the existing job. Optionally add a second job with TF for full-coverage runs.

---

## 6. Log Message Sanitization (`common/sanitize.py`)

### Problem
API responses echo raw `message` fields that may contain passwords, tokens, or PII.

### Design

```python
# common/sanitize.py
import re

DEFAULT_REDACTION_PATTERNS = [
    (re.compile(r'password[=:]\s*\S+', re.IGNORECASE), 'password=***REDACTED***'),
    (re.compile(r'token[=:]\s*\S+', re.IGNORECASE), 'token=***REDACTED***'),
    (re.compile(r'key[=:]\s*\S+', re.IGNORECASE), 'key=***REDACTED***'),
    (re.compile(r'secret[=:]\s*\S+', re.IGNORECASE), 'secret=***REDACTED***'),
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '***SSN***'),
]

MAX_MESSAGE_LENGTH = 512

def sanitize_message(message: str, patterns=None) -> str:
    """Redact sensitive patterns and truncate."""
    patterns = patterns or DEFAULT_REDACTION_PATTERNS
    for regex, replacement in patterns:
        message = regex.sub(replacement, message)
    return message[:MAX_MESSAGE_LENGTH]
```

### Application

Sanitization applies **only to output paths** — the raw message is preserved in the DataFrame for feature extraction (`classify_threat()` still sees `shadow`, `passwd`, etc.).

Applied in:
- `api/anomaly_api.py` response-building loop (line ~344)
- `batch/batch_processor.py` output serialization (line ~184)
- `mcp/anomaly_mcp_server.py` result building (line ~165)

---

## 7. Thread Safety in Batch Processor

### Problem
`batch/batch_processor.py` has no locking around model state. The API has `_model_lock` but batch doesn't.

### Design

Add `threading.Lock` to `BatchProcessor`:

```python
import threading

class BatchProcessor:
    def __init__(self, ...):
        ...
        self._model_lock = threading.Lock()

    def load_models(self):
        with self._model_lock:
            # existing load logic
            ...

    def process_file(self, filepath):
        with self._model_lock:
            snapshot = {
                "feature_pipeline": self.feature_pipeline,
                "isolation_forest": self.isolation_forest,
                "statistical_detector": self.statistical_detector,
                "scorer": self.scorer,
                "threshold": self.threshold,
            }
        # Processing happens outside the lock using snapshot
        ...
```

Lock is held briefly for snapshotting, released for processing.

---

## 8. TLS / Reverse Proxy Documentation

### Problem
No production TLS guidance. API serves plain HTTP.

### Design

Add a `## Production Deployment` section to `README.md` covering:
- Recommendation to use a reverse proxy (nginx, Caddy, Traefik)
- Minimal nginx config snippet for TLS proxy to `127.0.0.1:8000`
- Minimal Caddy config (auto-TLS)
- Note that `docker-compose.yml` binds to `127.0.0.1` by default
- Add a comment in `docker-compose.yml` pointing to docs

No code changes — documentation only.

---

## Implementation Order

Changes are ordered by dependency:

1. `common/threats.py` — no dependencies
2. `common/config.py` — no dependencies
3. `common/sanitize.py` — no dependencies
4. `common/security.py` auth flip — depends on nothing, but tests depend on 1-3 being done
5. `core/autoencoder.py` + deprecate pipeline — depends on nothing
6. Integrate 1-4 into `api/`, `batch/`, `mcp/` — depends on 1-4
7. Thread safety in batch — can be done with step 6
8. Tests + coverage to 90% — depends on all above
9. TLS docs — independent, can be done anytime

---

## Files Created

| File | Purpose |
|------|---------|
| `common/threats.py` | Extensible threat classification |
| `common/config.py` | Centralized constants |
| `common/sanitize.py` | Log message redaction |
| `core/autoencoder.py` | Extracted autoencoder detector |
| `config/requirements_full.txt` | TF dependency for full mode |
| `tests/test_threats.py` | Threat classifier tests |
| `tests/test_config.py` | Config constants tests |
| `tests/test_sanitize.py` | Sanitization tests |
| `tests/test_autoencoder.py` | Autoencoder tests (skip without TF) |

## Files Modified

| File | Changes |
|------|---------|
| `common/security.py` | Flip `REQUIRE_AUTH` default |
| `api/anomaly_api.py` | Import from common modules, remove local duplicates, add sanitization |
| `batch/batch_processor.py` | Import from common, remove duplicates, add lock + sanitization |
| `mcp/anomaly_mcp_server.py` | Import from common, remove duplicates, add sanitization |
| `core/log_anomaly_detection_lite.py` | Import severity defaults from common/threats |
| `core/intrusion_detection_pipeline.py` | Add deprecation warning |
| `pyproject.toml` | Coverage to 90%, remove MCP from omit |
| `tests/conftest.py` | Set REQUIRE_AUTH=false for tests |
| `tests/test_mcp.py` | Expand coverage |
| `tests/test_core.py` | Expand coverage |
| `tests/test_api.py` | Expand coverage |
| `.github/workflows/ci.yml` | Optional TF job |
| `docker/docker-compose.yml` | Add TLS docs comment |
| `README.md` | Production deployment section |
