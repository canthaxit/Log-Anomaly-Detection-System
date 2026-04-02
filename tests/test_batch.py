"""Tests for batch/batch_processor.py."""

import json
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


class TestBatchFindNewFiles:
    def test_finds_json_files(self, tmp_path):
        from batch.batch_processor import BatchProcessor
        import common.security as sec

        log_dir = tmp_path / "testlogs"
        log_dir.mkdir()
        (log_dir / "test.json").write_text('[{"test": true}]')

        sec.ALLOWED_LOG_DIRS.append(str(log_dir))
        try:
            bp = BatchProcessor(
                log_dir=str(log_dir),
                output_dir=str(tmp_path / "out"),
            )
            files = bp.find_new_files()
            assert len(files) >= 1
        finally:
            sec.ALLOWED_LOG_DIRS.remove(str(log_dir))

    def test_skips_already_processed(self, tmp_path):
        from batch.batch_processor import BatchProcessor
        import common.security as sec

        log_dir = tmp_path / "testlogs"
        log_dir.mkdir()
        f = log_dir / "test.json"
        f.write_text('[{"test": true}]')

        sec.ALLOWED_LOG_DIRS.append(str(log_dir))
        try:
            bp = BatchProcessor(
                log_dir=str(log_dir),
                output_dir=str(tmp_path / "out"),
            )
            bp.processed_files.add(str(f))
            files = bp.find_new_files()
            assert len(files) == 0
        finally:
            sec.ALLOWED_LOG_DIRS.remove(str(log_dir))


class TestBatchProcessBatch:
    def test_no_new_files_returns_zero(self, tmp_path):
        from batch.batch_processor import BatchProcessor
        import common.security as sec

        log_dir = tmp_path / "emptylogs"
        log_dir.mkdir()

        sec.ALLOWED_LOG_DIRS.append(str(log_dir))
        try:
            bp = BatchProcessor(
                log_dir=str(log_dir),
                output_dir=str(tmp_path / "out"),
            )
            result = bp.process_batch()
            assert result["processed"] == 0
        finally:
            sec.ALLOWED_LOG_DIRS.remove(str(log_dir))


class TestBatchLoadProcessedFiles:
    def test_loads_existing_processed_list(self, tmp_path):
        from batch.batch_processor import BatchProcessor
        import common.security as sec

        log_dir = tmp_path / "testlogs"
        log_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        # Create a processed_files.txt
        (out_dir / "processed_files.txt").write_text("file1.json\nfile2.json\n")

        sec.ALLOWED_LOG_DIRS.append(str(log_dir))
        try:
            bp = BatchProcessor(
                log_dir=str(log_dir),
                output_dir=str(out_dir),
            )
            assert "file1.json" in bp.processed_files
            assert "file2.json" in bp.processed_files
        finally:
            sec.ALLOWED_LOG_DIRS.remove(str(log_dir))

    def test_save_processed_file(self, tmp_path):
        from batch.batch_processor import BatchProcessor
        import common.security as sec

        log_dir = tmp_path / "testlogs"
        log_dir.mkdir()
        out_dir = tmp_path / "out"

        sec.ALLOWED_LOG_DIRS.append(str(log_dir))
        try:
            bp = BatchProcessor(
                log_dir=str(log_dir),
                output_dir=str(out_dir),
            )
            bp.save_processed_file("test_file.json")
            assert "test_file.json" in bp.processed_files
            content = (out_dir / "processed_files.txt").read_text()
            assert "test_file.json" in content
        finally:
            sec.ALLOWED_LOG_DIRS.remove(str(log_dir))


class TestBatchProcessFileWithModels:
    def test_process_file_success(self, model_dir, tmp_path):
        from batch.batch_processor import BatchProcessor
        import common.security as sec

        log_dir = tmp_path / "testlogs"
        log_dir.mkdir()
        # Read test log data and write to a file
        test_logs = Path(__file__).parent / "test_logs_normal.json"
        log_data = test_logs.read_text()
        log_file = log_dir / "test_normal.json"
        log_file.write_text(log_data)

        sec.ALLOWED_MODEL_DIRS.append(str(model_dir))
        sec.ALLOWED_LOG_DIRS.append(str(log_dir))
        try:
            bp = BatchProcessor(
                model_dir=str(model_dir),
                log_dir=str(log_dir),
                output_dir=str(tmp_path / "out"),
            )
            bp.load_models()
            result = bp.process_file(log_file)
            assert result["status"] == "success"
            assert result["total_events"] > 0
        finally:
            sec.ALLOWED_MODEL_DIRS.remove(str(model_dir))
            sec.ALLOWED_LOG_DIRS.remove(str(log_dir))

    def test_process_file_empty(self, model_dir, tmp_path):
        from batch.batch_processor import BatchProcessor
        import common.security as sec

        log_dir = tmp_path / "testlogs"
        log_dir.mkdir()
        log_file = log_dir / "empty.json"
        log_file.write_text("[]")

        sec.ALLOWED_MODEL_DIRS.append(str(model_dir))
        sec.ALLOWED_LOG_DIRS.append(str(log_dir))
        try:
            bp = BatchProcessor(
                model_dir=str(model_dir),
                log_dir=str(log_dir),
                output_dir=str(tmp_path / "out"),
            )
            bp.load_models()
            result = bp.process_file(log_file)
            assert result["status"] == "empty"
        finally:
            sec.ALLOWED_MODEL_DIRS.remove(str(model_dir))
            sec.ALLOWED_LOG_DIRS.remove(str(log_dir))


class TestBatchProcessBatchWithFiles:
    def test_process_batch_with_files(self, model_dir, tmp_path):
        from batch.batch_processor import BatchProcessor
        import common.security as sec

        log_dir = tmp_path / "testlogs"
        log_dir.mkdir()
        test_logs = Path(__file__).parent / "test_logs_normal.json"
        log_file = log_dir / "test_batch.json"
        log_file.write_text(test_logs.read_text())

        sec.ALLOWED_MODEL_DIRS.append(str(model_dir))
        sec.ALLOWED_LOG_DIRS.append(str(log_dir))
        try:
            bp = BatchProcessor(
                model_dir=str(model_dir),
                log_dir=str(log_dir),
                output_dir=str(tmp_path / "out"),
            )
            bp.load_models()
            result = bp.process_batch()
            assert result["processed"] >= 1
            assert "results" in result
        finally:
            sec.ALLOWED_MODEL_DIRS.remove(str(model_dir))
            sec.ALLOWED_LOG_DIRS.remove(str(log_dir))


class TestBatchLoadModelsFailure:
    def test_load_models_invalid_dir(self, tmp_path):
        from batch.batch_processor import BatchProcessor
        import common.security as sec

        log_dir = tmp_path / "testlogs"
        log_dir.mkdir()

        sec.ALLOWED_LOG_DIRS.append(str(log_dir))
        try:
            bp = BatchProcessor(
                model_dir="anomaly_outputs/nonexistent_xyz",
                log_dir=str(log_dir),
                output_dir=str(tmp_path / "out"),
            )
            result = bp.load_models()
            assert result is False
        finally:
            sec.ALLOWED_LOG_DIRS.remove(str(log_dir))
