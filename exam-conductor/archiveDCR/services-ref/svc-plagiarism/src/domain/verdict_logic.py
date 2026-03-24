"""Teacher verdict validation logic.

ZERO I/O -- pure computation only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ---- enums & dataclasses -------------------------------------------------- #


class Verdict(str, Enum):
    """Allowed teacher verdict values."""

    CONFIRMED_PLAGIARISM = "confirmed_plagiarism"
    DISMISSED = "dismissed"


class VerdictStatus(str, Enum):
    """Verdict record status including the initial pending state."""

    PENDING = "pending"
    CONFIRMED_PLAGIARISM = "confirmed_plagiarism"
    DISMISSED = "dismissed"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of verdict validation."""

    valid: bool
    errors: list[str]


# ---- constants ------------------------------------------------------------ #

MINIMUM_REASON_LENGTH = 5
ALLOWED_VERDICTS = frozenset({Verdict.CONFIRMED_PLAGIARISM, Verdict.DISMISSED})


# ---- public API ----------------------------------------------------------- #


def validate_verdict(
    verdict: str,
    reason: str,
) -> ValidationResult:
    """Validate a teacher verdict submission.

    Rules
    -----
    - ``verdict`` must be one of the allowed enum values.
    - ``reason`` is mandatory for ALL verdicts (minimum 5 characters).
    - Leading/trailing whitespace in ``reason`` is stripped before
      length check.

    Returns
    -------
    ValidationResult with ``valid=True`` if all checks pass, or
    ``valid=False`` with a list of human-readable error messages.
    """
    errors: list[str] = []

    # -- verdict enum check ---
    try:
        Verdict(verdict)
    except ValueError:
        allowed = ", ".join(v.value for v in Verdict)
        errors.append(
            f"Invalid verdict '{verdict}'. Must be one of: {allowed}."
        )

    # -- reason check ---
    trimmed = reason.strip() if reason else ""
    if len(trimmed) < MINIMUM_REASON_LENGTH:
        errors.append(
            f"Reason is required and must be at least "
            f"{MINIMUM_REASON_LENGTH} characters (got {len(trimmed)})."
        )

    return ValidationResult(valid=len(errors) == 0, errors=errors)
