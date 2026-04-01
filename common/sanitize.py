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
