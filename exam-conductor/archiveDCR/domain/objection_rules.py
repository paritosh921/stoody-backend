"""Objection business rules — ZERO I/O, pure domain logic.

Validates filing constraints, resolution payloads, and escalation
targets without touching any database or network.

Test IDs: U-REV-RULES-01 through U-REV-RULES-10
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


class Resolution(StrEnum):
    """Possible resolution outcomes."""

    APPROVED = "approved"
    REJECTED = "rejected"


class EscalationTarget(StrEnum):
    """Roles that may receive an escalated objection."""

    HOD = "hod"
    SENIOR_EVALUATOR = "senior_evaluator"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class FilingError(Exception):
    """Raised when objection filing preconditions are not met."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class ResolutionError(Exception):
    """Raised when a resolution payload is invalid."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class EscalationError(Exception):
    """Raised when escalation preconditions are not met."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


# ---------------------------------------------------------------------------
# Filing validation
# ---------------------------------------------------------------------------

_MIN_OBJECTION_TEXT_LENGTH = 10


@dataclass(frozen=True, slots=True)
class FilingContext:
    """Everything needed to validate an objection filing."""

    role: str
    objection_window_open: bool
    existing_objection_for_question: bool
    objection_text: str


def validate_filing(ctx: FilingContext) -> None:
    """Validate that the filing is permitted.

    Raises
    ------
    FilingError
        If any precondition fails.
    """
    if ctx.role != "student":
        raise FilingError(
            "not_student",
            "Only students may file objections",
        )

    if not ctx.objection_window_open:
        raise FilingError(
            "window_closed",
            "The objection window is not open for this exam",
        )

    if ctx.existing_objection_for_question:
        raise FilingError(
            "duplicate",
            "An objection has already been filed for this question",
        )

    if len(ctx.objection_text.strip()) < _MIN_OBJECTION_TEXT_LENGTH:
        raise FilingError(
            "text_too_short",
            f"Objection text must be at least {_MIN_OBJECTION_TEXT_LENGTH} characters",
        )


# ---------------------------------------------------------------------------
# Resolution validation
# ---------------------------------------------------------------------------

_MIN_REASON_LENGTH = 5


@dataclass(frozen=True, slots=True)
class ResolutionPayload:
    """Data supplied when resolving an objection."""

    resolution: Resolution
    reason: str
    new_score: float | None = None


def validate_resolution(payload: ResolutionPayload) -> None:
    """Validate a resolution payload.

    Rules
    -----
    * Approval requires ``new_score`` to be present and non-negative.
    * Rejection requires a mandatory ``reason`` (>= 5 chars).
    * Both resolutions require a reason.

    Raises
    ------
    ResolutionError
        If any rule is violated.
    """
    if len(payload.reason.strip()) < _MIN_REASON_LENGTH:
        raise ResolutionError(
            "reason_too_short",
            f"Reason must be at least {_MIN_REASON_LENGTH} characters",
        )

    if payload.resolution == Resolution.APPROVED:
        if payload.new_score is None:
            raise ResolutionError(
                "missing_score",
                "Approved resolution must include a new_score",
            )
        if payload.new_score < 0:
            raise ResolutionError(
                "negative_score",
                "new_score must be non-negative",
            )

    # Rejection: reason is already validated above.  No extra fields needed.


# ---------------------------------------------------------------------------
# Escalation validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EscalationPayload:
    """Data supplied when escalating an objection."""

    escalated_to: str
    reason: str


_VALID_ESCALATION_TARGETS = {t.value for t in EscalationTarget}


def validate_escalation(payload: EscalationPayload) -> None:
    """Validate an escalation payload.

    Rules
    -----
    * Target must be ``hod`` or ``senior_evaluator``.
    * Reason is mandatory (>= 5 chars).

    Raises
    ------
    EscalationError
        If any rule is violated.
    """
    if payload.escalated_to not in _VALID_ESCALATION_TARGETS:
        raise EscalationError(
            "invalid_target",
            f"Escalation target must be one of: {', '.join(sorted(_VALID_ESCALATION_TARGETS))}",
        )

    if len(payload.reason.strip()) < _MIN_REASON_LENGTH:
        raise EscalationError(
            "reason_too_short",
            f"Escalation reason must be at least {_MIN_REASON_LENGTH} characters",
        )
