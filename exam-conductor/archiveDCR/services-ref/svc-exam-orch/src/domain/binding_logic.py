"""Pen-student binding validation — ZERO I/O, pure logic.

Binding lifecycle: discovered -> provisional -> confirmed | rejected

Invariants:
- No duplicate pen MAC within the same exam.
- Student must exist in the exam roster.
- Only provisional bindings can be confirmed or rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BindingStatus(StrEnum):
    """Pen binding lifecycle states."""

    PROVISIONAL = "provisional"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class BindingSource(StrEnum):
    """How the binding was initiated."""

    REGISTRATION_SCAN = "registration_scan"
    MANUAL_REGISTER = "manual_register"
    SERVER_SYNC = "server_sync"


class BindingValidationError(Exception):
    """Raised when a binding violates a domain invariant."""


# ---------------------------------------------------------------------------
# Existing-state representation (passed in from storage layer)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExistingBinding:
    """Lightweight view of an already-persisted binding."""

    pen_mac: str
    student_id: str
    status: str


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------


def validate_new_binding(
    pen_mac: str,
    student_id: str,
    roster_student_ids: frozenset[str],
    existing_bindings: list[ExistingBinding],
) -> None:
    """Check that a new binding request is valid.

    Parameters
    ----------
    pen_mac:
        BLE MAC address of the pen.
    student_id:
        Stoody student identifier.
    roster_student_ids:
        Set of student IDs on the exam roster.
    existing_bindings:
        All current bindings for this exam.

    Raises
    ------
    BindingValidationError
        If the pen MAC is already bound (non-rejected) or the student
        is not on the roster.
    """
    if not pen_mac or not pen_mac.strip():
        raise BindingValidationError("pen_mac must not be empty")

    if not student_id or not student_id.strip():
        raise BindingValidationError("student_id must not be empty")

    if student_id not in roster_student_ids:
        raise BindingValidationError(
            f"Student {student_id} is not on the exam roster"
        )

    # Check for duplicate MAC (only active — not rejected)
    for binding in existing_bindings:
        if binding.pen_mac == pen_mac and binding.status != BindingStatus.REJECTED:
            raise BindingValidationError(
                f"Pen {pen_mac} is already bound in this exam"
            )

    # Check for duplicate student (only active — not rejected)
    for binding in existing_bindings:
        if (
            binding.student_id == student_id
            and binding.status != BindingStatus.REJECTED
        ):
            raise BindingValidationError(
                f"Student {student_id} already has a pen bound"
            )


def validate_binding_confirmation(
    current_status: str,
    new_status: str,
) -> None:
    """Validate a binding status change (confirm or reject).

    Only ``provisional`` bindings can transition to ``confirmed`` or
    ``rejected``.

    Raises
    ------
    BindingValidationError
        If the transition is invalid.
    """
    if current_status != BindingStatus.PROVISIONAL:
        raise BindingValidationError(
            f"Cannot change binding from '{current_status}'; "
            f"only provisional bindings can be confirmed/rejected"
        )

    valid_targets = {BindingStatus.CONFIRMED, BindingStatus.REJECTED}
    if new_status not in valid_targets:
        raise BindingValidationError(
            f"Invalid target status '{new_status}'; "
            f"must be 'confirmed' or 'rejected'"
        )
