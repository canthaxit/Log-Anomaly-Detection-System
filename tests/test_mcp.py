"""Tests for mcp/anomaly_mcp_server.py — input limits and validation."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import asyncio

# ---------------------------------------------------------------------------
# Import the MCP server module directly by file path because the directory
# name ``mcp/`` collides with the ``mcp`` PyPI package.
# ---------------------------------------------------------------------------
_MCP_FILE = Path(__file__).resolve().parent.parent / "mcp" / "anomaly_mcp_server.py"

@pytest.fixture(scope="module")
def mcp_mod():
    # The MCP server calls exit(1) if the mcp SDK is not installed.
    # Provide a lightweight stub so the module can load without the SDK.
    mcp_stub_installed = "mcp.server" in sys.modules
    if not mcp_stub_installed:
        import types
        # Create minimal stubs for mcp.server, mcp.server.stdio, mcp.types
        mcp_pkg = types.ModuleType("mcp")
        mcp_server = types.ModuleType("mcp.server")
        mcp_server_stdio = types.ModuleType("mcp.server.stdio")
        mcp_types = types.ModuleType("mcp.types")

        class _FakeServer:
            def __init__(self, name): pass
            def list_tools(self): return lambda fn: fn
            def call_tool(self): return lambda fn: fn
        mcp_server.Server = _FakeServer
        mcp_server_stdio.stdio_server = None
        mcp_types.Tool = object
        mcp_types.TextContent = object

        sys.modules["mcp"] = mcp_pkg
        sys.modules["mcp.server"] = mcp_server
        sys.modules["mcp.server.stdio"] = mcp_server_stdio
        sys.modules["mcp.types"] = mcp_types

    spec = importlib.util.spec_from_file_location("anomaly_mcp_server", _MCP_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Input-limit constants
# ---------------------------------------------------------------------------

class TestMCPInputLimits:
    def test_constants_correct(self, mcp_mod):
        assert mcp_mod.MAX_INPUT_SIZE == 10 * 1024 * 1024  # 10 MB
        assert mcp_mod.MAX_LOG_EVENTS == 10_000
        assert mcp_mod.MAX_CSV_COLUMNS == 50

    def test_rejects_oversized_input(self, mcp_mod):
        huge = "x" * (mcp_mod.MAX_INPUT_SIZE + 1)
        result = mcp_mod.analyze_logs(huge)
        assert result.get("status") == "error"

    def test_rejects_too_many_events(self, mcp_mod):
        # Temporarily mark models as loaded
        orig = mcp_mod.MODEL_STATE["loaded"]
        mcp_mod.MODEL_STATE["loaded"] = True
        try:
            events = [{"timestamp": f"2026-01-01T00:00:{i:02d}Z", "user": "a",
                        "source_ip": "1.1.1.1", "event_type": "login",
                        "action": "success", "message": "m"} for i in range(50)]
            big_data = json.dumps(events * (mcp_mod.MAX_LOG_EVENTS // len(events) + 1))
            result = mcp_mod.analyze_logs(big_data)
            assert result.get("status") == "error"
        finally:
            mcp_mod.MODEL_STATE["loaded"] = orig

    def test_rejects_csv_column_bomb(self, mcp_mod):
        orig = mcp_mod.MODEL_STATE["loaded"]
        mcp_mod.MODEL_STATE["loaded"] = True
        try:
            header = ",".join([f"col{i}" for i in range(60)])
            values = ",".join(["x"] * 60)
            csv_data = f"{header}\n{values}"
            result = mcp_mod.analyze_logs(csv_data, format="csv")
            assert result.get("status") == "error"
        finally:
            mcp_mod.MODEL_STATE["loaded"] = orig

    def test_not_loaded_returns_error(self, mcp_mod):
        orig = mcp_mod.MODEL_STATE["loaded"]
        mcp_mod.MODEL_STATE["loaded"] = False
        try:
            result = mcp_mod.analyze_logs('[{"timestamp":"2026-01-01T00:00:00Z"}]')
            assert result.get("status") == "error"
        finally:
            mcp_mod.MODEL_STATE["loaded"] = orig

    def test_load_models_validates_path(self, mcp_mod):
        result = mcp_mod.load_models(model_dir="/tmp/evil_models")
        assert result.get("status") == "error"

    def test_get_stats_returns_dict(self, mcp_mod):
        result = mcp_mod.get_stats()
        assert isinstance(result, dict)


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


class TestMCPLoadModels:
    def test_load_models_with_valid_dir(self, mcp_mod, model_dir):
        """Test loading models from a valid directory."""
        import common.security as sec
        sec.ALLOWED_MODEL_DIRS.append(str(model_dir))
        try:
            result = mcp_mod.load_models(str(model_dir))
            assert result["status"] == "success"
            assert mcp_mod.MODEL_STATE["loaded"] is True
        finally:
            sec.ALLOWED_MODEL_DIRS.remove(str(model_dir))

    def test_load_models_nonexistent_dir(self, mcp_mod):
        """Loading from a nonexistent but allowed dir returns error."""
        import common.security as sec
        sec.ALLOWED_MODEL_DIRS.append("anomaly_outputs/nonexistent_dir_for_test")
        try:
            result = mcp_mod.load_models("anomaly_outputs/nonexistent_dir_for_test")
            assert result["status"] == "error"
        finally:
            sec.ALLOWED_MODEL_DIRS.remove("anomaly_outputs/nonexistent_dir_for_test")


class TestMCPAnalyzeLogsSuccess:
    def test_analyze_json_logs(self, mcp_mod, model_dir):
        """Test full analysis path with loaded models."""
        import common.security as sec
        sec.ALLOWED_MODEL_DIRS.append(str(model_dir))
        try:
            mcp_mod.load_models(str(model_dir))
            logs = json.dumps([
                {"timestamp": "2026-01-01T00:00:00Z", "user": "alice",
                 "source_ip": "1.1.1.1", "event_type": "login",
                 "action": "success", "message": "normal log"},
                {"timestamp": "2026-01-01T00:01:00Z", "user": "bob",
                 "source_ip": "1.1.1.2", "event_type": "login",
                 "action": "success", "message": "another log"},
            ])
            result = mcp_mod.analyze_logs(logs)
            assert result["status"] == "success"
            assert result["total_events"] == 2
            assert "anomalies" in result
        finally:
            sec.ALLOWED_MODEL_DIRS.remove(str(model_dir))

    def test_analyze_single_dict(self, mcp_mod, model_dir):
        """Test analysis with a single dict (not array) JSON."""
        import common.security as sec
        sec.ALLOWED_MODEL_DIRS.append(str(model_dir))
        try:
            mcp_mod.load_models(str(model_dir))
            log = json.dumps(
                {"timestamp": "2026-01-01T00:00:00Z", "user": "alice",
                 "source_ip": "1.1.1.1", "event_type": "login",
                 "action": "success", "message": "single log"}
            )
            result = mcp_mod.analyze_logs(log)
            assert result["status"] == "success"
            assert result["total_events"] == 1
        finally:
            sec.ALLOWED_MODEL_DIRS.remove(str(model_dir))

    def test_analyze_csv_logs(self, mcp_mod, model_dir):
        """Test CSV format analysis."""
        import common.security as sec
        sec.ALLOWED_MODEL_DIRS.append(str(model_dir))
        try:
            mcp_mod.load_models(str(model_dir))
            csv_data = "timestamp,user,source_ip,event_type,action,message\n"
            csv_data += "2026-01-01T00:00:00Z,alice,1.1.1.1,login,success,ok\n"
            csv_data += "2026-01-01T00:01:00Z,bob,1.1.1.2,login,success,ok\n"
            result = mcp_mod.analyze_logs(csv_data, format="csv")
            assert result["status"] == "success"
            assert result["total_events"] == 2
        finally:
            sec.ALLOWED_MODEL_DIRS.remove(str(model_dir))

    def test_analyze_empty_logs(self, mcp_mod, model_dir):
        """Test analysis with empty list returns success with 0 events."""
        import common.security as sec
        sec.ALLOWED_MODEL_DIRS.append(str(model_dir))
        try:
            mcp_mod.load_models(str(model_dir))
            result = mcp_mod.analyze_logs("[]")
            assert result["status"] == "success"
            assert result["total_events"] == 0
        finally:
            sec.ALLOWED_MODEL_DIRS.remove(str(model_dir))


class TestMCPGetStatsSuccess:
    def test_get_stats_when_loaded(self, mcp_mod, model_dir):
        """Test get_stats returns model info when loaded."""
        import common.security as sec
        sec.ALLOWED_MODEL_DIRS.append(str(model_dir))
        try:
            mcp_mod.load_models(str(model_dir))
            result = mcp_mod.get_stats()
            assert result["status"] == "success"
            assert result["models_loaded"] is True
            assert "threshold" in result
            assert "feature_pipeline" in result
            assert "isolation_forest" in result
        finally:
            sec.ALLOWED_MODEL_DIRS.remove(str(model_dir))


class TestMCPAnalyzeLogFileWithModel:
    def test_analyze_valid_file(self, mcp_mod, model_dir, tmp_path):
        """Test analyze_log_file with a valid log file."""
        import common.security as sec
        sec.ALLOWED_MODEL_DIRS.append(str(model_dir))
        log_dir = tmp_path / "testlogs"
        log_dir.mkdir()
        log_file = log_dir / "test.json"
        log_file.write_text(json.dumps([
            {"timestamp": "2026-01-01T00:00:00Z", "user": "alice",
             "source_ip": "1.1.1.1", "event_type": "login",
             "action": "success", "message": "ok"},
        ]))
        sec.ALLOWED_LOG_DIRS.append(str(log_dir))
        try:
            mcp_mod.load_models(str(model_dir))
            result = mcp_mod.analyze_log_file(str(log_file))
            assert result["status"] == "success"
        finally:
            sec.ALLOWED_MODEL_DIRS.remove(str(model_dir))
            sec.ALLOWED_LOG_DIRS.remove(str(log_dir))

    def test_analyze_csv_file(self, mcp_mod, model_dir, tmp_path):
        """Test analyze_log_file with a CSV file."""
        import common.security as sec
        sec.ALLOWED_MODEL_DIRS.append(str(model_dir))
        log_dir = tmp_path / "testlogs"
        log_dir.mkdir()
        log_file = log_dir / "test.csv"
        csv_data = "timestamp,user,source_ip,event_type,action,message\n"
        csv_data += "2026-01-01T00:00:00Z,alice,1.1.1.1,login,success,ok\n"
        log_file.write_text(csv_data)
        sec.ALLOWED_LOG_DIRS.append(str(log_dir))
        try:
            mcp_mod.load_models(str(model_dir))
            result = mcp_mod.analyze_log_file(str(log_file))
            assert result["status"] == "success"
        finally:
            sec.ALLOWED_MODEL_DIRS.remove(str(model_dir))
            sec.ALLOWED_LOG_DIRS.remove(str(log_dir))


class TestMCPAnalyzeLogsWithAttackData:
    def test_analyze_attack_logs_detects_anomalies(self, mcp_mod, model_dir):
        """Test that attack logs produce anomalies."""
        import common.security as sec
        sec.ALLOWED_MODEL_DIRS.append(str(model_dir))
        try:
            mcp_mod.load_models(str(model_dir))
            attack_file = Path(__file__).parent / "test_logs_attack.json"
            log_data = attack_file.read_text()
            result = mcp_mod.analyze_logs(log_data)
            assert result["status"] == "success"
            assert result["total_events"] > 0
            # Attack logs should have some anomalies (threshold dependent)
            assert "anomalies" in result
        finally:
            sec.ALLOWED_MODEL_DIRS.remove(str(model_dir))


class TestMCPAnalyzeLogsErrorHandling:
    def test_invalid_json_returns_error(self, mcp_mod, model_dir):
        """Test that invalid JSON returns an error."""
        import common.security as sec
        sec.ALLOWED_MODEL_DIRS.append(str(model_dir))
        try:
            mcp_mod.load_models(str(model_dir))
            result = mcp_mod.analyze_logs("not valid json {{{")
            assert result["status"] == "error"
        finally:
            sec.ALLOWED_MODEL_DIRS.remove(str(model_dir))
