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
