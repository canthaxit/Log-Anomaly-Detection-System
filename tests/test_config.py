"""Tests for common/config.py — centralized constants."""


class TestConstants:
    def test_max_upload_size(self):
        from common.config import MAX_UPLOAD_SIZE
        assert MAX_UPLOAD_SIZE == 10 * 1024 * 1024

    def test_max_input_size_alias(self):
        from common.config import MAX_INPUT_SIZE, MAX_UPLOAD_SIZE
        assert MAX_INPUT_SIZE == MAX_UPLOAD_SIZE

    def test_max_log_events(self):
        from common.config import MAX_LOG_EVENTS
        assert MAX_LOG_EVENTS == 10_000

    def test_max_csv_columns(self):
        from common.config import MAX_CSV_COLUMNS
        assert MAX_CSV_COLUMNS == 50

    def test_rate_limits_are_strings(self):
        from common.config import RATE_LIMITS
        for key, value in RATE_LIMITS.items():
            assert isinstance(value, str)
            assert "/" in value

    def test_default_threshold(self):
        from common.config import DEFAULT_THRESHOLD
        assert 0.0 < DEFAULT_THRESHOLD < 1.0
