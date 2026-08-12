"""Shared marks semantics for authored and immutable exam questions.

Negative marking belongs to deterministic objective scoring. Subjective and
unclassified questions use criterion-based scores whose lower bound is zero,
so carrying a JEE-style penalty into those records is both misleading and an
invalid grading contract.
"""

from __future__ import annotations

import math
from typing import Any


OBJECTIVE_QUESTION_TYPES = frozenset({"mcq", "integer", "objective"})
MAX_NEGATIVE_MARKS = 50.0


def parse_question_penalty(
    value: Any,
    *,
    allow_missing: bool = True,
    maximum: float = MAX_NEGATIVE_MARKS,
) -> float | None:
    """Strictly validate an author-provided objective penalty.

    Read paths use :func:`normalize_question_penalty` to tolerate historical
    records. Write/finalization paths use this parser so NaN, infinity,
    booleans, and out-of-policy values cannot enter a frozen exam contract.
    """

    if value in (None, ""):
        if allow_missing:
            return None
        raise ValueError("Penalty is required")
    if isinstance(value, bool):
        raise ValueError("Penalty must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Penalty must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError("Penalty must be a finite number")
    if parsed < 0:
        raise ValueError("Penalty must be greater than or equal to zero")
    if parsed > maximum:
        raise ValueError(f"Penalty cannot exceed {maximum:g} points")
    return round(parsed, 2)


def canonical_question_type(
    question_type: Any,
    *,
    document_question_type: Any = None,
) -> str:
    """Resolve a question's explicit type without guessing mixed questions."""

    resolved = str(question_type or "").strip().lower()
    if not resolved:
        resolved = str(document_question_type or "").strip().lower()
    return resolved


def supports_negative_marking(
    question_type: Any,
    *,
    document_question_type: Any = None,
) -> bool:
    """Return whether a wrong attempted answer may receive a penalty."""

    return canonical_question_type(
        question_type,
        document_question_type=document_question_type,
    ) in OBJECTIVE_QUESTION_TYPES


def normalize_question_penalty(
    value: Any,
    *,
    question_type: Any,
    document_question_type: Any = None,
    objective_default: float = 1.0,
) -> float:
    """Return the canonical non-negative penalty for one question.

    Objective content retains the historical +4/-1 default when no explicit
    value exists. Subjective, mixed-unclassified, and unknown types always
    resolve to zero because their scoring engines never award negative marks.
    """

    if not supports_negative_marking(
        question_type,
        document_question_type=document_question_type,
    ):
        return 0.0

    candidate = objective_default if value in (None, "") or isinstance(value, bool) else value
    try:
        parsed = float(candidate)
    except (TypeError, ValueError):
        parsed = float(objective_default)
    if not math.isfinite(parsed):
        parsed = float(objective_default)
    return round(max(0.0, parsed), 2)
