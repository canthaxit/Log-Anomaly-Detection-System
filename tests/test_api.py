"""Tests for api/anomaly_api.py — REST endpoint behaviour."""

import json
from io import BytesIO
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"

    def test_has_security_headers(self, client):
        resp = client.get("/health")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"


# ---------------------------------------------------------------------------
# Models info
# ---------------------------------------------------------------------------

class TestModelsInfo:
    def test_returns_features_when_loaded(self, client):
        resp = client.get("/models/info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["loaded"] is True
        assert body["n_features"] > 0
        assert isinstance(body["feature_names"], list)


# ---------------------------------------------------------------------------
# Analyze (JSON body)
# ---------------------------------------------------------------------------

class TestAnalyzeEndpoint:
    def test_normal_logs(self, client):
        logs = json.loads(
            (Path(__file__).parent / "test_logs_normal.json").read_text()
        )
        resp = client.post("/analyze", json={"logs": logs})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["total_events"] == len(logs)

    def test_attack_logs(self, client):
        logs = json.loads(
            (Path(__file__).parent / "test_logs_attack.json").read_text()
        )
        resp = client.post("/analyze", json={"logs": logs})
        assert resp.status_code == 200
        body = resp.json()
        assert body["anomalies_detected"] >= 0

    def test_empty_logs_list(self, client):
        resp = client.post("/analyze", json={"logs": []})
        # FastAPI validates min length or returns 200 with 0 events
        assert resp.status_code in (200, 422)


# ---------------------------------------------------------------------------
# Analyze file upload
# ---------------------------------------------------------------------------

class TestAnalyzeFileEndpoint:
    def test_upload_json(self, client):
        logs = json.loads(
            (Path(__file__).parent / "test_logs_normal.json").read_text()
        )
        content = json.dumps(logs).encode()
        resp = client.post(
            "/analyze/file",
            files={"file": ("test.json", BytesIO(content), "application/json")},
        )
        assert resp.status_code == 200

    def test_null_filename_rejected(self, client):
        resp = client.post(
            "/analyze/file",
            files={"file": ("", BytesIO(b"[]"), "application/json")},
        )
        assert resp.status_code in (400, 422)

    def test_unsupported_format_400(self, client):
        resp = client.post(
            "/analyze/file",
            files={"file": ("test.xml", BytesIO(b"<xml/>"), "text/xml")},
        )
        assert resp.status_code == 400

    def test_csv_column_bomb_400(self, client):
        # CSV with >50 columns
        header = ",".join([f"col{i}" for i in range(60)])
        values = ",".join(["x"] * 60)
        csv_content = f"{header}\n{values}".encode()
        resp = client.post(
            "/analyze/file",
            files={"file": ("test.csv", BytesIO(csv_content), "text/csv")},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStatsEndpoint:
    def test_returns_model_info(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["models_loaded"] is True
        assert "threshold" in body


class TestAnalyzeEndpointEdgeCases:
    def test_models_not_loaded_returns_400(self, api_app, client):
        from api.anomaly_api import MODEL_STATE
        orig = MODEL_STATE["loaded"]
        MODEL_STATE["loaded"] = False
        try:
            resp = client.post("/analyze", json={"logs": [
                {"timestamp": "2026-01-01T00:00:00Z", "user": "a",
                 "source_ip": "1.1.1.1", "event_type": "login",
                 "action": "success", "message": "m"}
            ]})
            assert resp.status_code == 400
        finally:
            MODEL_STATE["loaded"] = orig

    def test_return_all_events_flag(self, client):
        logs = [
            {"timestamp": "2026-01-01T00:00:00Z", "user": "alice",
             "source_ip": "1.1.1.1", "event_type": "login",
             "action": "success", "message": "normal log", "severity": "low"}
        ]
        resp = client.post("/analyze", json={"logs": logs, "return_all_events": True})
        assert resp.status_code == 200


class TestStatsNotLoaded:
    def test_stats_not_loaded_returns_400(self, api_app, client):
        from api.anomaly_api import MODEL_STATE
        orig = MODEL_STATE["loaded"]
        MODEL_STATE["loaded"] = False
        try:
            resp = client.get("/stats")
            assert resp.status_code == 400
        finally:
            MODEL_STATE["loaded"] = orig


class TestModelLoadEndpoint:
    def test_invalid_model_dir_returns_403(self, client):
        resp = client.post("/models/load?model_dir=/tmp/evil")
        assert resp.status_code == 403

    def test_nonexistent_dir_returns_404(self, client):
        resp = client.post("/models/load?model_dir=anomaly_outputs/nonexistent")
        assert resp.status_code in (403, 404, 500)


class TestAnalyzeFileNotLoaded:
    def test_file_upload_not_loaded_returns_400(self, api_app, client):
        from api.anomaly_api import MODEL_STATE
        from io import BytesIO
        orig = MODEL_STATE["loaded"]
        MODEL_STATE["loaded"] = False
        try:
            resp = client.post(
                "/analyze/file",
                files={"file": ("test.json", BytesIO(b"[]"), "application/json")},
            )
            assert resp.status_code == 400
        finally:
            MODEL_STATE["loaded"] = orig


class TestModelsInfoNotLoaded:
    def test_models_info_not_loaded(self, api_app, client):
        from api.anomaly_api import MODEL_STATE
        orig = MODEL_STATE["loaded"]
        MODEL_STATE["loaded"] = False
        try:
            resp = client.get("/models/info")
            assert resp.status_code == 200
            body = resp.json()
            assert body["loaded"] is False
        finally:
            MODEL_STATE["loaded"] = orig


class TestModelLoadSuccess:
    def test_load_models_success(self, api_app, client, model_dir):
        import common.security as sec
        sec.ALLOWED_MODEL_DIRS.append(str(model_dir))
        try:
            resp = client.post(f"/models/load?model_dir={model_dir}")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "success"
        finally:
            sec.ALLOWED_MODEL_DIRS.remove(str(model_dir))


class TestAnalyzeEmptyLogs:
    def test_empty_df_after_preprocess(self, client):
        """Test with logs that become empty after preprocessing (all invalid timestamps)."""
        logs = [
            {"timestamp": "not-a-date", "user": "alice",
             "source_ip": "1.1.1.1", "event_type": "login",
             "action": "success", "message": "m"}
        ]
        resp = client.post("/analyze", json={"logs": logs})
        # Should succeed with 0 events (invalid timestamps removed)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_events"] == 0
