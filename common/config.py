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
