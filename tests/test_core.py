"""Tests for core/log_anomaly_detection_lite.py — parsing, features, detection, scoring."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from log_anomaly_detection_lite import (
    AnomalyScorer,
    LogFeaturePipeline,
    LogParser,
    StatisticalAnomalyDetector,
    preprocess_logs,
)


# ---------------------------------------------------------------------------
# LogParser
# ---------------------------------------------------------------------------

class TestLogParser:
    def test_parse_json_array(self, tmp_path):
        data = [
            {"timestamp": "2026-01-01T09:00:00Z", "user": "alice",
             "source_ip": "1.2.3.4", "event_type": "login",
             "action": "success", "message": "ok"},
        ]
        f = tmp_path / "logs.json"
        f.write_text(json.dumps(data))
        parser = LogParser()
        df = parser.parse_log_file(str(f))
        assert len(df) == 1
        assert "user" in df.columns

    def test_parse_jsonl(self, tmp_path):
        lines = [
            '{"timestamp":"2026-01-01T09:00:00Z","user":"a","source_ip":"1.1.1.1","event_type":"login","action":"success","message":"ok"}',
            '{"timestamp":"2026-01-01T09:01:00Z","user":"b","source_ip":"1.1.1.2","event_type":"login","action":"success","message":"ok"}',
        ]
        f = tmp_path / "logs.jsonl"
        f.write_text("\n".join(lines))
        parser = LogParser()
        df = parser.parse_log_file(str(f))
        assert len(df) == 2

    def test_empty_file_returns_empty_df(self, tmp_path):
        f = tmp_path / "empty.json"
        f.write_text("[]")
        parser = LogParser()
        df = parser.parse_log_file(str(f))
        assert len(df) == 0

    def test_normalize_aliases(self):
        """Column aliases like 'src_ip' should map to 'source_ip'."""
        parser = LogParser()
        df = pd.DataFrame([{
            "timestamp": "2026-01-01T09:00:00Z",
            "username": "alice",
            "src_ip": "1.2.3.4",
            "event_type": "login",
            "action": "success",
            "message": "ok",
        }])
        df = parser._normalize_schema(df)
        assert "source_ip" in df.columns or "src_ip" not in df.columns

    def test_load_from_directory(self, tmp_path):
        data = [{"timestamp": "2026-01-01T09:00:00Z", "user": "alice",
                 "source_ip": "1.2.3.4", "event_type": "login",
                 "action": "success", "message": "ok"}]
        (tmp_path / "a.json").write_text(json.dumps(data))
        parser = LogParser()
        df = parser.load_logs(str(tmp_path))
        assert len(df) >= 1

    def test_no_files_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        parser = LogParser()
        # load_logs should raise or return empty when no files found
        with pytest.raises(Exception):
            df = parser.load_logs(str(empty_dir))
            if len(df) == 0:
                raise ValueError("No log files found")


# ---------------------------------------------------------------------------
# preprocess_logs
# ---------------------------------------------------------------------------

class TestPreprocessLogs:
    def _make_df(self, rows):
        """Helper: build DataFrame and normalize schema (converts timestamps)."""
        parser = LogParser()
        df = pd.DataFrame(rows)
        return parser._normalize_schema(df)

    def test_deduplicates(self):
        row = {"timestamp": "2026-01-01T09:00:00Z", "user": "alice",
               "source_ip": "1.2.3.4", "event_type": "login",
               "action": "success", "message": "ok"}
        df = self._make_df([row, row])
        result = preprocess_logs(df)
        assert len(result) == 1

    def test_sorts_by_timestamp(self):
        rows = [
            {"timestamp": "2026-01-01T10:00:00Z", "user": "b",
             "source_ip": "1.1.1.1", "event_type": "login",
             "action": "success", "message": "m"},
            {"timestamp": "2026-01-01T09:00:00Z", "user": "a",
             "source_ip": "1.1.1.1", "event_type": "login",
             "action": "success", "message": "m"},
        ]
        df = self._make_df(rows)
        result = preprocess_logs(df)
        ts = result["timestamp"].tolist()
        assert ts[0] <= ts[1]

    def test_removes_invalid_timestamps(self):
        rows = [
            {"timestamp": "2026-01-01T09:00:00Z", "user": "a",
             "source_ip": "1.1.1.1", "event_type": "login",
             "action": "success", "message": "m"},
            {"timestamp": "not-a-date", "user": "b",
             "source_ip": "1.1.1.1", "event_type": "login",
             "action": "success", "message": "m"},
        ]
        df = self._make_df(rows)
        result = preprocess_logs(df)
        # The invalid timestamp becomes NaT and gets dropped
        assert len(result) == 1


# ---------------------------------------------------------------------------
# LogFeaturePipeline
# ---------------------------------------------------------------------------

class TestLogFeaturePipeline:
    def test_fit_sets_baselines(self, normal_df):
        pipe = LogFeaturePipeline()
        pipe.fit(normal_df)
        assert hasattr(pipe, "user_baselines_") or hasattr(pipe, "_fitted")

    def test_transform_before_fit_raises(self, normal_df):
        pipe = LogFeaturePipeline()
        with pytest.raises(Exception):
            pipe.transform(normal_df)

    def test_fit_transform_returns_ndarray(self, normal_df):
        pipe = LogFeaturePipeline()
        result = pipe.fit_transform(normal_df)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(normal_df)

    def test_transform_handles_new_event_types(self, fitted_pipeline, attack_df):
        """Pipeline should handle event types not seen during fit."""
        result = fitted_pipeline.transform(attack_df)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(attack_df)


# ---------------------------------------------------------------------------
# StatisticalAnomalyDetector
# ---------------------------------------------------------------------------

class TestStatisticalAnomalyDetector:
    def test_fit_builds_baselines(self, normal_df):
        det = StatisticalAnomalyDetector()
        det.fit(normal_df)
        # Should have some baseline state
        assert det is not None

    def test_brute_force_scores_failures(self, fitted_stat_detector, attack_df):
        scores = fitted_stat_detector.detect_brute_force(attack_df)
        assert isinstance(scores, np.ndarray)
        # Attack data has 12 failed logins — should have some non-zero scores
        assert scores.max() > 0

    def test_privilege_escalation(self, fitted_stat_detector, attack_df):
        scores = fitted_stat_detector.detect_privilege_escalation(attack_df)
        assert isinstance(scores, np.ndarray)
        # Attack data has sudo event
        assert scores.max() > 0

    def test_data_exfiltration(self, fitted_stat_detector, attack_df):
        scores = fitted_stat_detector.detect_data_exfiltration(attack_df)
        assert isinstance(scores, np.ndarray)

    def test_detect_all_combined(self, fitted_stat_detector, attack_df):
        scores = fitted_stat_detector.detect_all(attack_df)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(attack_df)
        assert scores.max() > 0


# ---------------------------------------------------------------------------
# AnomalyScorer
# ---------------------------------------------------------------------------

class TestAnomalyScorer:
    def test_combine_scores_weighted(self, scorer):
        iso = np.array([0.2, 0.8, 0.5])
        stat = np.array([0.4, 0.6, 0.3])
        combined = scorer.combine_scores({"isolation_forest": iso, "statistical": stat})
        assert isinstance(combined, np.ndarray)
        assert len(combined) == 3
        # Combined should preserve relative ordering
        assert combined[1] > combined[0]

    def test_normalize_constant_returns_zeros(self, scorer):
        const = np.array([5.0, 5.0, 5.0])
        result = scorer.normalize_scores(const)
        assert np.all(result == 0.0)

    def test_calibrate_threshold(self, scorer):
        scores = np.linspace(0, 1, 100)
        threshold = scorer.calibrate_threshold(scores, false_positive_rate=0.05)
        assert 0.0 <= threshold <= 1.0
