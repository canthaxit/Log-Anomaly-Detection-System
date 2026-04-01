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
