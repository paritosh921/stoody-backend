"""Unit tests for pen-student binding validation — domain layer, no I/O.

Test IDs: U-ORCH-BND-01 through U-ORCH-BND-08.
"""

from __future__ import annotations

import pytest

from src.domain.binding_logic import (
    BindingValidationError,
    ExistingBinding,
    validate_binding_confirmation,
    validate_new_binding,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ROSTER = frozenset(["student-1", "student-2", "student-3"])


def _existing(
    pen_mac: str = "AA:BB:CC:DD:EE:FF",
    student_id: str = "student-1",
    status: str = "provisional",
) -> ExistingBinding:
    return ExistingBinding(pen_mac=pen_mac, student_id=student_id, status=status)


# ---------------------------------------------------------------------------
# U-ORCH-BND-01: Valid new binding
# ---------------------------------------------------------------------------


class TestValidNewBinding:
    def test_fresh_binding(self) -> None:
        validate_new_binding(
            pen_mac="AA:BB:CC:DD:EE:FF",
            student_id="student-1",
            roster_student_ids=_ROSTER,
            existing_bindings=[],
        )

    def test_binding_after_rejected(self) -> None:
        """Re-binding a pen after its previous binding was rejected."""
        existing = [_existing(status="rejected")]
        validate_new_binding(
            pen_mac="AA:BB:CC:DD:EE:FF",
            student_id="student-2",
            roster_student_ids=_ROSTER,
            existing_bindings=existing,
        )


# ---------------------------------------------------------------------------
# U-ORCH-BND-02: Student not on roster
# ---------------------------------------------------------------------------


class TestStudentNotOnRoster:
    def test_unknown_student_rejected(self) -> None:
        with pytest.raises(BindingValidationError, match="not on the exam roster"):
            validate_new_binding(
                pen_mac="AA:BB:CC:DD:EE:FF",
                student_id="unknown-student",
                roster_student_ids=_ROSTER,
                existing_bindings=[],
            )


# ---------------------------------------------------------------------------
# U-ORCH-BND-03: Duplicate pen MAC
# ---------------------------------------------------------------------------


class TestDuplicatePen:
    def test_same_pen_provisional_rejected(self) -> None:
        existing = [_existing(pen_mac="AA:BB:CC:DD:EE:FF", status="provisional")]
        with pytest.raises(BindingValidationError, match="already bound"):
            validate_new_binding(
                pen_mac="AA:BB:CC:DD:EE:FF",
                student_id="student-2",
                roster_student_ids=_ROSTER,
                existing_bindings=existing,
            )

    def test_same_pen_confirmed_rejected(self) -> None:
        existing = [_existing(pen_mac="AA:BB:CC:DD:EE:FF", status="confirmed")]
        with pytest.raises(BindingValidationError, match="already bound"):
            validate_new_binding(
                pen_mac="AA:BB:CC:DD:EE:FF",
                student_id="student-2",
                roster_student_ids=_ROSTER,
                existing_bindings=existing,
            )


# ---------------------------------------------------------------------------
# U-ORCH-BND-04: Duplicate student
# ---------------------------------------------------------------------------


class TestDuplicateStudent:
    def test_student_already_bound(self) -> None:
        existing = [_existing(
            pen_mac="11:22:33:44:55:66",
            student_id="student-1",
            status="confirmed",
        )]
        with pytest.raises(BindingValidationError, match="already has a pen"):
            validate_new_binding(
                pen_mac="AA:BB:CC:DD:EE:FF",
                student_id="student-1",
                roster_student_ids=_ROSTER,
                existing_bindings=existing,
            )


# ---------------------------------------------------------------------------
# U-ORCH-BND-05: Empty inputs
# ---------------------------------------------------------------------------


class TestEmptyInputs:
    def test_empty_pen_mac(self) -> None:
        with pytest.raises(BindingValidationError, match="pen_mac"):
            validate_new_binding(
                pen_mac="",
                student_id="student-1",
                roster_student_ids=_ROSTER,
                existing_bindings=[],
            )

    def test_empty_student_id(self) -> None:
        with pytest.raises(BindingValidationError, match="student_id"):
            validate_new_binding(
                pen_mac="AA:BB:CC:DD:EE:FF",
                student_id="",
                roster_student_ids=_ROSTER,
                existing_bindings=[],
            )


# ---------------------------------------------------------------------------
# U-ORCH-BND-06: Confirmation validation
# ---------------------------------------------------------------------------


class TestConfirmation:
    def test_confirm_provisional(self) -> None:
        validate_binding_confirmation("provisional", "confirmed")

    def test_reject_provisional(self) -> None:
        validate_binding_confirmation("provisional", "rejected")

    def test_confirm_already_confirmed(self) -> None:
        with pytest.raises(BindingValidationError, match="only provisional"):
            validate_binding_confirmation("confirmed", "confirmed")

    def test_confirm_rejected(self) -> None:
        with pytest.raises(BindingValidationError, match="only provisional"):
            validate_binding_confirmation("rejected", "confirmed")

    def test_invalid_target_status(self) -> None:
        with pytest.raises(BindingValidationError, match="Invalid target"):
            validate_binding_confirmation("provisional", "active")
