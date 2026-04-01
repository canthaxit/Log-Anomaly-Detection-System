"""Tests for common/sanitize.py — PII redaction in log messages."""


class TestSanitizeMessage:
    def test_redacts_password(self):
        from common.sanitize import sanitize_message
        assert "***REDACTED***" in sanitize_message("user login password=hunter2")

    def test_redacts_password_colon(self):
        from common.sanitize import sanitize_message
        assert "***REDACTED***" in sanitize_message("Password: mysecretpassword")

    def test_redacts_token(self):
        from common.sanitize import sanitize_message
        assert "***REDACTED***" in sanitize_message("auth token=abc123xyz")

    def test_redacts_key(self):
        from common.sanitize import sanitize_message
        assert "***REDACTED***" in sanitize_message("api key=AKIAIOSFODNN7")

    def test_redacts_secret(self):
        from common.sanitize import sanitize_message
        assert "***REDACTED***" in sanitize_message("secret=wJalrXUtnFEMI")

    def test_redacts_ssn(self):
        from common.sanitize import sanitize_message
        result = sanitize_message("SSN is 123-45-6789")
        assert "123-45-6789" not in result
        assert "***SSN***" in result

    def test_preserves_normal_message(self):
        from common.sanitize import sanitize_message
        msg = "User alice logged in from 10.0.0.1"
        assert sanitize_message(msg) == msg

    def test_truncates_long_message(self):
        from common.sanitize import sanitize_message, MAX_MESSAGE_LENGTH
        long_msg = "a" * (MAX_MESSAGE_LENGTH + 100)
        assert len(sanitize_message(long_msg)) == MAX_MESSAGE_LENGTH

    def test_custom_patterns(self):
        import re
        from common.sanitize import sanitize_message
        custom = [(re.compile(r"credit_card=\S+"), "credit_card=***REDACTED***")]
        result = sanitize_message("credit_card=4111111111111111", patterns=custom)
        assert "4111111111111111" not in result

    def test_multiple_redactions_in_one_message(self):
        from common.sanitize import sanitize_message
        msg = "password=abc token=xyz"
        result = sanitize_message(msg)
        assert "abc" not in result
        assert "xyz" not in result
