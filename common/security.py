"""
Shared security utilities for Log Anomaly Detection System.
Path validation, API authentication, CORS configuration, and model signing.
"""

import hashlib
import hmac
import logging
import os
from pathlib import Path

logger = logging.getLogger("security")

# ---------------------------------------------------------------------------
# Path validation — prevents path-traversal and arbitrary pickle loads
# ---------------------------------------------------------------------------

ALLOWED_MODEL_DIRS = [
    d.strip()
    for d in os.environ.get("ALLOWED_MODEL_DIRS", "anomaly_outputs").split(",")
    if d.strip()
]

ALLOWED_LOG_DIRS = [
    d.strip()
    for d in os.environ.get("ALLOWED_LOG_DIRS", "logs,tests").split(",")
    if d.strip()
]


def _is_within_allowed(resolved: Path, allowed_names: list[str]) -> bool:
    """Check whether *resolved* sits inside one of the *allowed_names* dirs."""
    cwd = Path.cwd().resolve()
    for name in allowed_names:
        allowed = (cwd / name).resolve()
        try:
            resolved.relative_to(allowed)
            return True
        except ValueError:
            continue
    return False


def validate_model_path(model_dir: str) -> Path:
    """Resolve *model_dir* and ensure it is inside an allowed model directory.

    Returns the resolved ``Path`` on success; raises ``ValueError`` otherwise.
    """
    resolved = Path(model_dir).resolve()
    if not _is_within_allowed(resolved, ALLOWED_MODEL_DIRS):
        raise ValueError(
            f"Model directory '{model_dir}' is outside allowed directories: "
            f"{ALLOWED_MODEL_DIRS}"
        )
    return resolved


def validate_log_path(filepath: str) -> Path:
    """Resolve *filepath* and ensure it is inside an allowed log directory.

    Returns the resolved ``Path`` on success; raises ``ValueError`` otherwise.
    """
    resolved = Path(filepath).resolve()
    if not _is_within_allowed(resolved, ALLOWED_LOG_DIRS):
        raise ValueError(
            f"Log path '{filepath}' is outside allowed directories: "
            f"{ALLOWED_LOG_DIRS}"
        )
    return resolved


# ---------------------------------------------------------------------------
# HMAC model signing — prevents deserialization of tampered pickle files
# ---------------------------------------------------------------------------

MODEL_SIGNING_KEY: str | None = os.environ.get("MODEL_SIGNING_KEY")


def sign_model_file(filepath: str | Path) -> Path:
    """Compute HMAC-SHA256 of *filepath* and write a ``.sig`` sidecar file.

    Requires ``MODEL_SIGNING_KEY`` to be set; raises ``RuntimeError`` otherwise.
    Returns the path to the ``.sig`` file.
    """
    if not MODEL_SIGNING_KEY:
        raise RuntimeError("MODEL_SIGNING_KEY is not set; cannot sign model files")
    fp = Path(filepath)
    mac = hmac.new(MODEL_SIGNING_KEY.encode(), fp.read_bytes(), hashlib.sha256)
    sig_path = fp.with_suffix(fp.suffix + ".sig")
    sig_path.write_text(mac.hexdigest())
    logger.info("Signed model file: %s", fp.name)
    return sig_path


def verify_model_file(filepath: str | Path) -> None:
    """Verify the HMAC-SHA256 signature of a model file.

    * If ``MODEL_SIGNING_KEY`` is **not** set the check is a no-op (backward compat).
    * If the key **is** set but the ``.sig`` file is missing or the digest does not
      match, a ``ValueError`` is raised to prevent loading a tampered file.
    """
    if not MODEL_SIGNING_KEY:
        return  # signing disabled — no-op
    fp = Path(filepath)
    sig_path = fp.with_suffix(fp.suffix + ".sig")
    if not sig_path.exists():
        raise ValueError(f"Model signature file missing for {fp.name}; refusing to load")
    expected = sig_path.read_text().strip()
    actual = hmac.new(MODEL_SIGNING_KEY.encode(), fp.read_bytes(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, actual):
        raise ValueError(f"Model signature verification failed for {fp.name}; file may be tampered")
    logger.info("Model signature verified: %s", fp.name)


# ---------------------------------------------------------------------------
# API Key authentication (optional -- disabled when API_KEY env var is unset)
# ---------------------------------------------------------------------------

API_KEY: str | None = os.environ.get("API_KEY")
REQUIRE_AUTH: bool = os.environ.get("REQUIRE_AUTH", "false").lower() == "true"

if API_KEY is None:
    if REQUIRE_AUTH:
        raise RuntimeError(
            "REQUIRE_AUTH is set but API_KEY is not configured. "
            "Set the API_KEY environment variable or disable REQUIRE_AUTH."
        )
    logger.warning("API_KEY not set -- authentication is DISABLED")


def get_verify_api_key():
    """Return a FastAPI dependency that checks the X-API-Key header.

    Import FastAPI lazily so this module can be used without FastAPI installed
    (e.g. by the batch processor).
    """
    from fastapi import HTTPException, Security
    from fastapi.security import APIKeyHeader

    _header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def _verify(key: str | None = Security(_header)):
        if API_KEY is None:
            return  # auth disabled
        if key is None or not hmac.compare_digest(key, API_KEY):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return _verify


# ---------------------------------------------------------------------------
# CORS allowed origins
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS: list[str] = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]
