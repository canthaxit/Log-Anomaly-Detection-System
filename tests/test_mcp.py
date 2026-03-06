"""Tests for mcp/anomaly_mcp_server.py — input limits and validation."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

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
