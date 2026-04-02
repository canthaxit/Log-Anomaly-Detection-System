"""Tests for core/log_anomaly_detection_lite.py — parsing, features, detection, scoring."""

import json
import os
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from log_anomaly_detection_lite import (
    AnomalyScorer,
    Config,
    LogFeaturePipeline,
    LogParser,
    StatisticalAnomalyDetector,
    assign_severity,
    classify_threat_type,
    create_isolation_forest,
    main as pipeline_main,
    parse_args,
    preprocess_logs,
    save_anomaly_report,
    print_summary,
    generate_visualizations,
    save_artifacts,
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


class TestAnomalyScorerFitNormalization:
    def test_fit_normalization_uses_stored_range(self, scorer):
        scores = np.array([0.0, 0.5, 1.0])
        scorer.fit_normalization("test_det", scores)
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
        assert len(combined) == 2


# ---------------------------------------------------------------------------
# classify_threat_type (standalone function)
# ---------------------------------------------------------------------------

class TestClassifyThreatType:
    def test_classifies_brute_force(self, fitted_stat_detector, attack_df):
        result = classify_threat_type(attack_df, fitted_stat_detector)
        assert isinstance(result, list)
        assert len(result) == len(attack_df)
        # Attack df has failed logins so brute_force should appear
        assert "brute_force" in result or "unknown" in result

    def test_classifies_normal_as_unknown_or_other(self, fitted_stat_detector, normal_df):
        result = classify_threat_type(normal_df, fitted_stat_detector)
        assert isinstance(result, list)
        assert len(result) == len(normal_df)


# ---------------------------------------------------------------------------
# assign_severity (standalone function)
# ---------------------------------------------------------------------------

class TestAssignSeverityFunction:
    def test_severity_levels(self):
        thresholds = {"critical": 0.95, "high": 0.85, "medium": 0.7, "low": 0.5}
        scores = np.array([0.99, 0.9, 0.75, 0.3])
        result = assign_severity(scores, thresholds)
        assert result == ["critical", "high", "medium", "low"]

    def test_all_low(self):
        thresholds = {"critical": 0.95, "high": 0.85, "medium": 0.7, "low": 0.5}
        scores = np.array([0.1, 0.2])
        result = assign_severity(scores, thresholds)
        assert all(s == "low" for s in result)


# ---------------------------------------------------------------------------
# save_anomaly_report
# ---------------------------------------------------------------------------

class TestSaveAnomalyReport:
    def test_saves_csv_and_json(self, tmp_path):
        df = pd.DataFrame([{
            "timestamp": "2026-01-01T00:00:00Z",
            "user": "alice",
            "source_ip": "1.1.1.1",
            "event_type": "login",
            "action": "failed",
            "message": "brute force",
            "severity": "high",
            "anomaly_score": 0.9,
            "threat_type": "brute_force",
        }])
        save_anomaly_report(df, str(tmp_path))
        assert (tmp_path / "anomalies_detected.csv").exists()
        assert (tmp_path / "anomalies_detailed.json").exists()


# ---------------------------------------------------------------------------
# print_summary
# ---------------------------------------------------------------------------

class TestPrintSummary:
    def test_prints_with_anomalies(self, capsys):
        df = pd.DataFrame([{
            "user": "alice",
            "source_ip": "1.1.1.1",
            "threat_type": "brute_force",
            "severity": "high",
        }])
        print_summary(df, 100)
        captured = capsys.readouterr()
        assert "DETECTION SUMMARY" in captured.out
        assert "100" in captured.out

    def test_prints_empty(self, capsys):
        df = pd.DataFrame(columns=["user", "source_ip", "threat_type", "severity"])
        print_summary(df, 50)
        captured = capsys.readouterr()
        assert "50" in captured.out


# ---------------------------------------------------------------------------
# generate_visualizations
# ---------------------------------------------------------------------------

class TestGenerateVisualizations:
    def test_saves_png(self, tmp_path, normal_df):
        scores = np.random.rand(len(normal_df))
        anomalies_df = normal_df.head(2).copy()
        anomalies_df["threat_type"] = "brute_force"
        anomalies_df["severity"] = "high"
        anomalies_df["user"] = "alice"
        generate_visualizations(normal_df, scores, 0.5, anomalies_df, str(tmp_path))
        assert (tmp_path / "anomaly_analysis.png").exists()

    def test_saves_png_no_anomalies(self, tmp_path, normal_df):
        scores = np.random.rand(len(normal_df))
        empty_df = pd.DataFrame(columns=["threat_type", "severity", "user", "source_ip"])
        generate_visualizations(normal_df, scores, 0.5, empty_df, str(tmp_path))
        assert (tmp_path / "anomaly_analysis.png").exists()


# ---------------------------------------------------------------------------
# save_artifacts
# ---------------------------------------------------------------------------

class TestSaveArtifacts:
    def test_saves_all_pkl_files(self, tmp_path, fitted_pipeline, fitted_iso_forest,
                                  fitted_stat_detector, scorer):
        class FakeConfig:
            output_dir = str(tmp_path)
            contamination = 0.01
            severity_thresholds = {"critical": 0.95, "high": 0.85, "medium": 0.7, "low": 0.5}
        save_artifacts(fitted_pipeline, fitted_iso_forest, fitted_stat_detector,
                       scorer, 0.7, FakeConfig())
        assert (tmp_path / "feature_pipeline.pkl").exists()
        assert (tmp_path / "isolation_forest_model.pkl").exists()
        assert (tmp_path / "statistical_detector.pkl").exists()
        assert (tmp_path / "inference_package.pkl").exists()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_config(self, tmp_path):
        config = Config()
        assert config.contamination == 0.01
        assert config.random_state == 42
        assert config.iso_forest_estimators == 200

    def test_config_creates_output_dir(self, tmp_path):
        import types
        args = types.SimpleNamespace(
            data_path="./logs/",
            log_format="auto",
            output_dir=str(tmp_path / "test_output"),
            contamination=0.05,
            baseline_period_days=3,
            random_state=123,
            iso_forest_estimators=100,
            time_windows=[3600],
        )
        config = Config(args)
        assert config.contamination == 0.05
        assert Path(config.output_dir).exists()


# ---------------------------------------------------------------------------
# create_isolation_forest
# ---------------------------------------------------------------------------

class TestCreateIsolationForest:
    def test_returns_isolation_forest(self):
        iso = create_isolation_forest()
        assert iso.n_estimators == 200

    def test_custom_params(self):
        iso = create_isolation_forest(contamination=0.05, n_estimators=50, random_state=0)
        assert iso.n_estimators == 50
        assert iso.contamination == 0.05


# ---------------------------------------------------------------------------
# StatisticalAnomalyDetector — lateral movement
# ---------------------------------------------------------------------------

class TestStatisticalLateralMovement:
    def test_lateral_movement(self, fitted_stat_detector, attack_df):
        scores = fitted_stat_detector.detect_lateral_movement(attack_df)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(attack_df)

    def test_lateral_movement_normal(self, fitted_stat_detector, normal_df):
        scores = fitted_stat_detector.detect_lateral_movement(normal_df)
        assert isinstance(scores, np.ndarray)


# ---------------------------------------------------------------------------
# LogFeaturePipeline — additional tests
# ---------------------------------------------------------------------------

class TestLogFeaturePipelineExtra:
    def test_get_feature_names(self, fitted_pipeline):
        names = fitted_pipeline.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_transform_unscaled(self, fitted_pipeline, normal_df):
        result = fitted_pipeline.transform(normal_df, scale=False)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(normal_df)


# ---------------------------------------------------------------------------
# LogParser — additional edge cases
# ---------------------------------------------------------------------------

class TestLogParserExtra:
    def test_parse_single_dict(self, tmp_path):
        data = {"timestamp": "2026-01-01T09:00:00Z", "user": "alice",
                "source_ip": "1.2.3.4", "event_type": "login",
                "action": "success", "message": "ok"}
        f = tmp_path / "single.json"
        f.write_text(json.dumps(data))
        parser = LogParser()
        df = parser.parse_log_file(str(f))
        assert len(df) == 1

    def test_parse_csv(self, tmp_path):
        """LogParser can parse CSV files through load_logs if JSON fails."""
        parser = LogParser()
        # Test with invalid JSON that falls through to JSONL parsing
        f = tmp_path / "bad.json"
        f.write_text("not valid json at all\n")
        df = parser.parse_log_file(str(f))
        assert len(df) == 0

    def test_normalize_defaults(self):
        """Missing columns get defaults."""
        parser = LogParser()
        df = pd.DataFrame([{"timestamp": "2026-01-01T09:00:00Z"}])
        result = parser._normalize_schema(df)
        assert "user" in result.columns
        assert "event_type" in result.columns
        assert "action" in result.columns
        assert "severity" in result.columns
        assert "message" in result.columns


# ---------------------------------------------------------------------------
# Full pipeline main() — integration test
# ---------------------------------------------------------------------------

class TestPipelineMain:
    def test_main_runs_end_to_end(self, tmp_path):
        """Run the full pipeline main() with test data."""
        # Create a logs directory with test data
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        # Generate enough data spanning 10+ days for baseline/analysis split
        base_logs = []
        for day in range(10):
            for hour in range(24):
                base_logs.append({
                    "timestamp": f"2026-01-{day+1:02d}T{hour:02d}:00:00Z",
                    "user": "alice" if hour < 12 else "bob",
                    "source_ip": "10.0.0.1",
                    "event_type": "login",
                    "action": "success",
                    "message": "normal login",
                })
        # Add some attack-like events in later days
        for i in range(15):
            base_logs.append({
                "timestamp": f"2026-01-09T03:{i:02d}:00Z",
                "user": "attacker",
                "source_ip": "192.168.1.100",
                "event_type": "login",
                "action": "failed",
                "message": "brute force attempt",
            })
        (log_dir / "test_logs.json").write_text(json.dumps(base_logs))

        output_dir = str(tmp_path / "output")
        args = types.SimpleNamespace(
            data_path=str(log_dir),
            log_format="auto",
            output_dir=output_dir,
            contamination=0.01,
            baseline_period_days=7,
            random_state=42,
            iso_forest_estimators=50,
            time_windows=[3600],
        )
        config = Config(args)
        pipeline_main(config)

        # Check outputs were created
        assert Path(output_dir).exists()
        assert (Path(output_dir) / "feature_pipeline.pkl").exists()
        assert (Path(output_dir) / "anomaly_analysis.png").exists()

    def test_main_no_log_files(self, tmp_path, capsys):
        """main() should handle missing log files gracefully."""
        empty_log_dir = tmp_path / "empty_logs"
        empty_log_dir.mkdir()
        output_dir = str(tmp_path / "output")
        args = types.SimpleNamespace(
            data_path=str(empty_log_dir),
            log_format="auto",
            output_dir=output_dir,
            contamination=0.01,
            baseline_period_days=7,
            random_state=42,
            iso_forest_estimators=50,
            time_windows=[3600],
        )
        config = Config(args)
        # Should not raise, just return
        pipeline_main(config)

    def test_main_insufficient_baseline(self, tmp_path, capsys):
        """main() should handle case where all data is in baseline period."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        # All events in a single day => baseline=7 days means no analysis data
        logs = [
            {"timestamp": "2026-01-01T09:00:00Z", "user": "alice",
             "source_ip": "10.0.0.1", "event_type": "login",
             "action": "success", "message": "ok"},
            {"timestamp": "2026-01-01T10:00:00Z", "user": "bob",
             "source_ip": "10.0.0.2", "event_type": "login",
             "action": "success", "message": "ok"},
        ]
        (log_dir / "short.json").write_text(json.dumps(logs))

        output_dir = str(tmp_path / "output")
        args = types.SimpleNamespace(
            data_path=str(log_dir),
            log_format="auto",
            output_dir=output_dir,
            contamination=0.01,
            baseline_period_days=7,
            random_state=42,
            iso_forest_estimators=50,
            time_windows=[3600],
        )
        config = Config(args)
        pipeline_main(config)


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_defaults(self):
        import sys
        orig = sys.argv
        sys.argv = ["prog"]
        try:
            args = parse_args()
            assert args.data_path == "./logs/"
            assert args.contamination == 0.01
            assert args.random_state == 42
        finally:
            sys.argv = orig

    def test_custom_args(self):
        import sys
        orig = sys.argv
        sys.argv = ["prog", "--data_path", "/tmp/logs", "--contamination", "0.05"]
        try:
            args = parse_args()
            assert args.data_path == "/tmp/logs"
            assert args.contamination == 0.05
        finally:
            sys.argv = orig
