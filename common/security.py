"""
Shared security utilities for Log Anomaly Detection System.
Path validation, API authentication, and CORS configuration.
"""

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
# API Key authentication (optional -- disabled when API_KEY env var is unset)
# ---------------------------------------------------------------------------

API_KEY: str | None = os.environ.get("API_KEY")


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
