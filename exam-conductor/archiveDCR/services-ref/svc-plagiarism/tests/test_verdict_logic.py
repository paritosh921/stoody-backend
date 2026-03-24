"""Unit tests for teacher verdict validation.

Test IDs: U-PLAG-VRD-01 through U-PLAG-VRD-07
"""

import pytest

from src.domain.verdict_logic import (
    ValidationResult,
    Verdict,
    validate_verdict,
    MINIMUM_REASON_LENGTH,
)


class TestValidateVerdict:
    """Verdict validation: enum checks and mandatory reason."""

    @pytest.mark.unit
    def test_valid_confirmed_plagiarism(self) -> None:
        """U-PLAG-VRD-01: confirmed_plagiarism with reason is valid."""
        result = validate_verdict("confirmed_plagiarism", "Clear copying observed")
        assert result.valid is True
        assert result.errors == []

    @pytest.mark.unit
    def test_valid_dismissed(self) -> None:
        """U-PLAG-VRD-02: dismissed with reason is valid."""
        result = validate_verdict("dismissed", "Common phrasing, not plagiarism")
        assert result.valid is True
        assert result.errors == []

    @pytest.mark.unit
    def test_invalid_verdict_string(self) -> None:
        """U-PLAG-VRD-03: Unknown verdict string is rejected."""
        result = validate_verdict("auto_penalize", "some reason here")
        assert result.valid is False
        assert len(result.errors) == 1
        assert "auto_penalize" in result.errors[0]

    @pytest.mark.unit
    def test_reason_too_short(self) -> None:
        """U-PLAG-VRD-04: Reason below minimum length is rejected."""
        result = validate_verdict("dismissed", "ok")
        assert result.valid is False
        assert any("at least" in e for e in result.errors)

    @pytest.mark.unit
    def test_empty_reason(self) -> None:
        """U-PLAG-VRD-05: Empty reason is rejected."""
        result = validate_verdict("confirmed_plagiarism", "")
        assert result.valid is False

    @pytest.mark.unit
    def test_whitespace_only_reason(self) -> None:
        """U-PLAG-VRD-06: Whitespace-only reason is rejected (trimmed)."""
        result = validate_verdict("dismissed", "    ")
        assert result.valid is False
        assert any("at least" in e for e in result.errors)

    @pytest.mark.unit
    def test_both_invalid(self) -> None:
        """U-PLAG-VRD-07: Both bad verdict and bad reason produce two errors."""
        result = validate_verdict("unknown_verdict", "ab")
        assert result.valid is False
        assert len(result.errors) == 2

    @pytest.mark.unit
    def test_returns_validation_result_type(self) -> None:
        """U-PLAG-VRD-08: Return type is ValidationResult."""
        result = validate_verdict("dismissed", "Valid long reason text")
        assert isinstance(result, ValidationResult)

    @pytest.mark.unit
    def test_minimum_reason_exactly_at_boundary(self) -> None:
        """U-PLAG-VRD-09: Reason exactly at minimum length is accepted."""
        reason = "x" * MINIMUM_REASON_LENGTH
        result = validate_verdict("dismissed", reason)
        assert result.valid is True
