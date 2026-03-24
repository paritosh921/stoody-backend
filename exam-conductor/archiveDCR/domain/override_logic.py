"""Teacher override validation -- pure domain logic.

An override replaces the current score for a single question.  The
override is recorded as an event carrying the old value, new value,
the teacher who applied it, a mandatory human-readable reason, and
a timestamp.

This module is ZERO I/O -- no asyncio, no DB, no network imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


# ---- Data models -------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OverrideEvent:
    """Immutable record of a teacher score override."""
    old_value: float
    new_value: float
    teacher_id: str
    reason: str
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of override validation."""
    valid: bool
    errors: list[str]


# ---- Constants ---------------------------------------------------------------

MIN_REASON_LENGTH = 5


# ---- Validation logic --------------------------------------------------------

def validate_override(
    old_value: float,
    new_value: float,
    reason: str,
    *,
    max_marks: float | None = None,
) -> ValidationResult:
    """Validate a proposed score override.

    Rules
    -----
    * ``reason`` must be a non-empty string of at least
      ``MIN_REASON_LENGTH`` characters (after stripping whitespace).
    * ``new_value`` must be >= 0.
    * If ``max_marks`` is provided, ``new_value`` must not exceed it.
    * ``new_value`` must differ from ``old_value``.

    Returns a ``ValidationResult`` indicating success or listing errors.
    """
    errors: list[str] = []

    stripped = reason.strip() if reason else ""
    if len(stripped) < MIN_REASON_LENGTH:
        errors.append(
            f"Reason must be at least {MIN_REASON_LENGTH} characters "
            f"(got {len(stripped)})."
        )

    if new_value < 0:
        errors.append("New score must be >= 0.")

    if max_marks is not None and new_value > max_marks:
        errors.append(
            f"New score ({new_value}) exceeds max marks ({max_marks})."
        )

    if old_value == new_value:
        errors.append("New score is the same as the current score.")

    return ValidationResult(valid=len(errors) == 0, errors=errors)


def build_override_event(
    old_value: float,
    new_value: float,
    teacher_id: str,
    reason: str,
    timestamp: datetime,
) -> OverrideEvent:
    """Create an ``OverrideEvent`` after validation has passed.

    Callers MUST call ``validate_override`` first and check ``valid``.
    """
    return OverrideEvent(
        old_value=old_value,
        new_value=new_value,
        teacher_id=teacher_id,
        reason=reason.strip(),
        timestamp=timestamp,
    )
