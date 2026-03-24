"""U-SCR-21 .. U-SCR-28: Override validation unit tests.

Domain-only -- no DB, no network.
"""

from datetime import datetime, timezone

import pytest

from src.domain.override_logic import (
    MIN_REASON_LENGTH,
    OverrideEvent,
    ValidationResult,
    build_override_event,
    validate_override,
)


class TestValidateOverride:
    """Core validation rules."""

    def test_valid_override(self) -> None:
        vr = validate_override(3.0, 4.0, "Student showed working in margin")
        assert vr.valid is True
        assert vr.errors == []

    def test_reason_too_short(self) -> None:
        vr = validate_override(3.0, 4.0, "ok")
        assert vr.valid is False
        assert any("at least" in e for e in vr.errors)

    def test_empty_reason(self) -> None:
        vr = validate_override(3.0, 4.0, "")
        assert vr.valid is False

    def test_whitespace_only_reason(self) -> None:
        vr = validate_override(3.0, 4.0, "     ")
        assert vr.valid is False

    def test_reason_at_boundary(self) -> None:
        """Exactly MIN_REASON_LENGTH chars should pass."""
        vr = validate_override(3.0, 4.0, "a" * MIN_REASON_LENGTH)
        assert vr.valid is True

    def test_negative_new_score(self) -> None:
        vr = validate_override(3.0, -1.0, "Valid reason here")
        assert vr.valid is False
        assert any(">= 0" in e for e in vr.errors)

    def test_exceeds_max_marks(self) -> None:
        vr = validate_override(3.0, 11.0, "Valid reason here", max_marks=10.0)
        assert vr.valid is False
        assert any("exceeds" in e for e in vr.errors)

    def test_same_value(self) -> None:
        vr = validate_override(5.0, 5.0, "Valid reason here")
        assert vr.valid is False
        assert any("same" in e.lower() for e in vr.errors)

    def test_multiple_errors(self) -> None:
        """Short reason + negative score = two errors."""
        vr = validate_override(5.0, -1.0, "hi")
        assert vr.valid is False
        assert len(vr.errors) >= 2

    def test_within_max_marks(self) -> None:
        vr = validate_override(3.0, 10.0, "Correct recheck", max_marks=10.0)
        assert vr.valid is True

    def test_no_max_marks_constraint(self) -> None:
        """Without max_marks, any positive value is fine."""
        vr = validate_override(3.0, 999.0, "Extra credit scenario")
        assert vr.valid is True


class TestBuildOverrideEvent:
    def test_creates_event(self) -> None:
        ts = datetime(2026, 3, 19, 12, 0, 0, tzinfo=timezone.utc)
        event = build_override_event(
            old_value=3.0,
            new_value=4.5,
            teacher_id="teacher_42",
            reason="  Partial credit for method  ",
            timestamp=ts,
        )
        assert isinstance(event, OverrideEvent)
        assert event.old_value == 3.0
        assert event.new_value == 4.5
        assert event.teacher_id == "teacher_42"
        assert event.reason == "Partial credit for method"  # stripped
        assert event.timestamp == ts

    def test_immutable(self) -> None:
        ts = datetime.now(timezone.utc)
        event = build_override_event(3.0, 4.0, "t1", "Valid reason", ts)
        with pytest.raises(AttributeError):
            event.new_value = 99.0  # type: ignore[misc]


class TestAuditTrail:
    """Verify that the OverrideEvent captures all audit-required fields."""

    def test_all_fields_present(self) -> None:
        ts = datetime.now(timezone.utc)
        event = build_override_event(2.0, 3.0, "teacher_7", "Recheck done", ts)
        assert event.old_value is not None
        assert event.new_value is not None
        assert event.teacher_id is not None
        assert event.reason is not None
        assert event.timestamp is not None
