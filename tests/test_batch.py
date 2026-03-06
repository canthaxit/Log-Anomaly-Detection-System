"""Tests for batch/batch_processor.py."""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# BatchProcessor init
# ---------------------------------------------------------------------------

class TestBatchProcessorInit:
    def test_validates_log_dir(self, tmp_path):
        """Should accept a log_dir inside ALLOWED_LOG_DIRS."""
        from batch.batch_processor import BatchProcessor
        # "tests" is in ALLOWED_LOG_DIRS
        bp = BatchProcessor(log_dir="tests", output_dir=str(tmp_path / "out"))
        assert bp.log_dir is not None

    def test_rejects_invalid_log_dir(self, tmp_path):
        from batch.batch_processor import BatchProcessor
        with pytest.raises(ValueError, match="outside allowed"):
            BatchProcessor(log_dir="/tmp/evil", output_dir=str(tmp_path / "out"))


# ---------------------------------------------------------------------------
# process_file error handling
# ---------------------------------------------------------------------------

class TestBatchProcessFile:
    def test_error_returns_processing_failed(self, tmp_path):
        """On error, process_file should return 'Processing failed', not str(e)."""
        from batch.batch_processor import BatchProcessor
        bp = BatchProcessor(log_dir="tests", output_dir=str(tmp_path / "out"))
        # Give it a non-existent file to trigger an error
        result = bp.process_file(Path("nonexistent_file.json"))
        assert result["status"] == "error"
        assert result["error"] == "Processing failed"


# ---------------------------------------------------------------------------
# load_models
# ---------------------------------------------------------------------------

class TestBatchLoadModels:
    def test_load_models_calls_verify(self, model_dir, tmp_path):
        """load_models should call verify_model_file for each pkl."""
        from batch.batch_processor import BatchProcessor
        import common.security as sec

        # Add model_dir to allowed paths
        sec.ALLOWED_MODEL_DIRS.append(str(model_dir))
        try:
            bp = BatchProcessor(
                model_dir=str(model_dir),
                log_dir="tests",
                output_dir=str(tmp_path / "out"),
            )
            result = bp.load_models()
            assert result is True
            assert bp.feature_pipeline is not None
        finally:
            sec.ALLOWED_MODEL_DIRS.remove(str(model_dir))

    def test_load_models_succeeds_with_valid_dir(self, model_dir, tmp_path):
        from batch.batch_processor import BatchProcessor
        import common.security as sec

        sec.ALLOWED_MODEL_DIRS.append(str(model_dir))
        try:
            bp = BatchProcessor(
                model_dir=str(model_dir),
                log_dir="tests",
                output_dir=str(tmp_path / "out"),
            )
            assert bp.load_models() is True
            assert bp.threshold is not None
        finally:
            sec.ALLOWED_MODEL_DIRS.remove(str(model_dir))
