"""Tests for common/security.py — path validation, HMAC signing, API auth."""

import os
import hmac
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest


class TestValidateModelPath:
    def test_valid_path_resolves(self):
        from common.security import validate_model_path
        # "anomaly_outputs" is in ALLOWED_MODEL_DIRS by default
        result = validate_model_path("anomaly_outputs")
        assert result == (Path.cwd() / "anomaly_outputs").resolve()

    def test_traversal_rejected(self):
        from common.security import validate_model_path
        with pytest.raises(ValueError, match="outside allowed"):
            validate_model_path("anomaly_outputs/../../etc")

    def test_absolute_outside_rejected(self):
        from common.security import validate_model_path
        with pytest.raises(ValueError, match="outside allowed"):
            validate_model_path("/tmp/evil_models")


class TestValidateLogPath:
    def test_valid_path_resolves(self):
        from common.security import validate_log_path
        result = validate_log_path("tests/test_logs_normal.json")
        assert result.name == "test_logs_normal.json"

    def test_traversal_rejected(self):
        from common.security import validate_log_path
        with pytest.raises(ValueError, match="outside allowed"):
            validate_log_path("tests/../../etc/passwd")


class TestModelSigning:
    def test_sign_verify_roundtrip(self, tmp_path):
        from common.security import sign_model_file, verify_model_file

        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"model-data-here")

        with patch.dict(os.environ, {"MODEL_SIGNING_KEY": "test-key-123"}):
            # Patch the module-level constant too
            import common.security as sec
            orig = sec.MODEL_SIGNING_KEY
            sec.MODEL_SIGNING_KEY = "test-key-123"
            try:
                sig_path = sign_model_file(model_file)
                assert sig_path.exists()
                # verify should not raise
                verify_model_file(model_file)
            finally:
                sec.MODEL_SIGNING_KEY = orig

    def test_sign_without_key_raises(self, tmp_path):
        from common.security import sign_model_file
        import common.security as sec

        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"data")

        orig = sec.MODEL_SIGNING_KEY
        sec.MODEL_SIGNING_KEY = None
        try:
            with pytest.raises(RuntimeError, match="MODEL_SIGNING_KEY"):
                sign_model_file(model_file)
        finally:
            sec.MODEL_SIGNING_KEY = orig

    def test_verify_no_key_is_noop(self, tmp_path):
        """When MODEL_SIGNING_KEY is unset, verify is a no-op (backward compat)."""
        from common.security import verify_model_file
        import common.security as sec

        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"data")

        orig = sec.MODEL_SIGNING_KEY
        sec.MODEL_SIGNING_KEY = None
        try:
            verify_model_file(model_file)  # should not raise
        finally:
            sec.MODEL_SIGNING_KEY = orig

    def test_missing_sig_raises(self, tmp_path):
        from common.security import verify_model_file
        import common.security as sec

        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"data")

        orig = sec.MODEL_SIGNING_KEY
        sec.MODEL_SIGNING_KEY = "test-key"
        try:
            with pytest.raises(ValueError, match="signature file missing"):
                verify_model_file(model_file)
        finally:
            sec.MODEL_SIGNING_KEY = orig

    def test_tampered_file_raises(self, tmp_path):
        from common.security import sign_model_file, verify_model_file
        import common.security as sec

        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"original-data")

        orig = sec.MODEL_SIGNING_KEY
        sec.MODEL_SIGNING_KEY = "test-key"
        try:
            sign_model_file(model_file)
            # Tamper with the file
            model_file.write_bytes(b"tampered-data")
            with pytest.raises(ValueError, match="verification failed"):
                verify_model_file(model_file)
        finally:
            sec.MODEL_SIGNING_KEY = orig


class TestAPIKeyAuth:
    def test_returns_callable(self):
        from common.security import get_verify_api_key
        dep = get_verify_api_key()
        assert callable(dep)

    @pytest.mark.asyncio
    async def test_rejects_wrong_key(self):
        import common.security as sec
        orig = sec.API_KEY
        sec.API_KEY = "correct-key"
        try:
            dep = sec.get_verify_api_key()
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await dep(key="wrong-key")
            assert exc_info.value.status_code == 401
        finally:
            sec.API_KEY = orig

    @pytest.mark.asyncio
    async def test_accepts_correct_key(self):
        import common.security as sec
        orig = sec.API_KEY
        sec.API_KEY = "correct-key"
        try:
            dep = sec.get_verify_api_key()
            await dep(key="correct-key")  # should not raise
        finally:
            sec.API_KEY = orig

    @pytest.mark.asyncio
    async def test_skips_when_no_key_configured(self):
        import common.security as sec
        orig = sec.API_KEY
        sec.API_KEY = None
        try:
            dep = sec.get_verify_api_key()
            await dep(key=None)  # should not raise
        finally:
            sec.API_KEY = orig
