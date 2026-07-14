"""Immutable PCR marking-policy and criterion helpers.

The AI is allowed to *evaluate evidence*, not invent the teacher's marking
scheme.  Authoring stores a small structured rubric per question and the
document stores the strictness/temperature policy.  Finalisation validates and
freezes both into the paper snapshot; conducted sessions only read that
snapshot.
"""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional


POLICY_VERSION = "criterion-rubric-v1"
STRUCTURED_RUBRIC_MODE = "criterion_rubric_v1"
LEGACY_AI_MODE = "legacy_ai_v1"
MAX_AI_TEMPERATURE = 0.20


# These are marking standards, not model-temperature aliases.  Temperature is
# deliberately kept low so a repeated assessment is consistent and auditable.
STRICTNESS_PROFILES: Dict[str, str] = {
    "lenient": (
        "Accept scientifically or academically valid alternative methods and "
        "minor OCR/notation defects when the intended work is clear. Award "
        "available method credit where the criterion is substantially met."
    ),
    "balanced": (
        "Apply ordinary teacher partial credit. Accept valid alternative "
        "methods, but require the criterion's stated outcome or evidence "
        "before awarding its marks."
    ),
    "strict": (
        "Apply the criterion exactly. Do not infer missing method, units, "
        "working, or conclusion. If handwriting/OCR makes evidence ambiguous, "
        "mark it for teacher review instead of guessing a deduction."
    ),
}

_CRITERION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_MARK_TOLERANCE = 0.01


def default_structured_marking_policy() -> Dict[str, Any]:
    """Return the policy used for newly authored PCR papers."""

    return {
        "version": POLICY_VERSION,
        "mode": STRUCTURED_RUBRIC_MODE,
        "strictness": "balanced",
        "temperature": 0.10,
    }


def legacy_marking_policy() -> Dict[str, Any]:
    """Return a compatibility policy for already-finalized legacy papers."""

    return {
        "version": "legacy-ai-v1",
        "mode": LEGACY_AI_MODE,
        "strictness": "balanced",
        "temperature": 0.10,
    }


def normalize_marking_policy(
    value: Optional[Mapping[str, Any]],
    *,
    default_structured: bool = False,
) -> Dict[str, Any]:
    """Validate and canonicalise one PCR document/session marking policy.

    ``default_structured=False`` is intentional: documents created before this
    feature remain readable and are not suddenly made invalid.  New documents
    and explicit policy saves use ``default_structured=True``.
    """

    if value is None:
        return (
            default_structured_marking_policy()
            if default_structured
            else legacy_marking_policy()
        )
    if not isinstance(value, Mapping):
        raise ValueError("PCR marking policy must be an object")

    fallback = default_structured_marking_policy() if default_structured else legacy_marking_policy()
    raw_mode = str(value.get("mode") or fallback["mode"]).strip().lower()
    aliases = {
        "structured": STRUCTURED_RUBRIC_MODE,
        "criteria": STRUCTURED_RUBRIC_MODE,
        "criterion_rubric": STRUCTURED_RUBRIC_MODE,
        "criterion-rubric-v1": STRUCTURED_RUBRIC_MODE,
        "legacy": LEGACY_AI_MODE,
    }
    mode = aliases.get(raw_mode, raw_mode)
    if mode not in {STRUCTURED_RUBRIC_MODE, LEGACY_AI_MODE}:
        raise ValueError("Unsupported PCR marking policy mode")

    strictness = str(value.get("strictness") or fallback["strictness"]).strip().lower()
    if strictness not in STRICTNESS_PROFILES:
        raise ValueError("Strictness must be lenient, balanced, or strict")

    raw_temperature = value.get("temperature", fallback["temperature"])
    try:
        temperature = float(raw_temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("AI temperature must be a number between 0 and 0.20") from exc
    if not math.isfinite(temperature) or not 0.0 <= temperature <= MAX_AI_TEMPERATURE:
        raise ValueError("AI temperature must be between 0 and 0.20")

    return {
        "version": POLICY_VERSION if mode == STRUCTURED_RUBRIC_MODE else "legacy-ai-v1",
        "mode": mode,
        "strictness": strictness,
        "temperature": round(temperature, 2),
    }


def is_structured_rubric_policy(policy: Optional[Mapping[str, Any]]) -> bool:
    """Return whether a policy requires locked criterion-by-criterion marks."""

    return normalize_marking_policy(policy).get("mode") == STRUCTURED_RUBRIC_MODE


def strictness_instruction(strictness: str) -> str:
    """Return the server-owned instruction for a marking standard."""

    return STRICTNESS_PROFILES.get(str(strictness).strip().lower(), STRICTNESS_PROFILES["balanced"])


def _as_json_list(value: Any) -> List[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Marking criteria must be valid JSON") from exc
    if not isinstance(value, list):
        raise ValueError("Marking criteria must be a list")
    return value


def normalize_marking_criteria(
    value: Any,
    *,
    assign_missing_ids: bool = True,
) -> List[Dict[str, Any]]:
    """Return a safe, serialisable list of teacher-authored criteria.

    Draft rows are allowed to be incomplete while a teacher is editing.  The
    stronger :func:`validate_marking_criteria` check is used only at the paper
    finalisation boundary.
    """

    raw_items = _as_json_list(value)
    criteria: List[Dict[str, Any]] = []
    used_ids: set[str] = set()
    for position, item in enumerate(raw_items, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Criterion {position} must be an object")

        raw_id = str(item.get("criterion_id") or item.get("id") or "").strip()
        if not raw_id and assign_missing_ids:
            raw_id = f"criterion_{position}"
        if raw_id and not _CRITERION_ID_RE.fullmatch(raw_id):
            raise ValueError(
                f"Criterion {position} has an invalid id; use letters, numbers, _ or -"
            )
        if raw_id in used_ids:
            raise ValueError(f"Criterion ids must be unique ({raw_id})")
        if raw_id:
            used_ids.add(raw_id)

        raw_marks = item.get("max_marks", item.get("marks", 0))
        try:
            max_marks = float(raw_marks or 0)
        except (TypeError, ValueError):
            max_marks = 0.0
        if not math.isfinite(max_marks):
            raise ValueError(f"Criterion {position} has an invalid mark value")

        description = str(
            item.get("description") or item.get("criterion") or item.get("step") or ""
        ).strip()
        acceptable_evidence = str(
            item.get("acceptable_evidence")
            or item.get("evidence")
            or item.get("expected_evidence")
            or ""
        ).strip()
        if len(description) > 1600 or len(acceptable_evidence) > 1600:
            raise ValueError(f"Criterion {position} is too long")

        criteria.append(
            {
                "criterion_id": raw_id,
                "description": description,
                "max_marks": round(max_marks, 2),
                "acceptable_evidence": acceptable_evidence,
            }
        )
    return criteria


def validate_marking_criteria(
    criteria: Iterable[Mapping[str, Any]],
    question_max_marks: Any,
) -> List[str]:
    """Return finalisation errors for a locked teacher marking rubric."""

    normalized = normalize_marking_criteria(list(criteria), assign_missing_ids=False)
    if not normalized:
        return ["add at least one criterion with marks"]

    try:
        max_marks = float(question_max_marks)
    except (TypeError, ValueError):
        max_marks = 0.0
    if not math.isfinite(max_marks) or max_marks <= 0:
        return ["assign question marks greater than zero before setting criteria"]

    errors: List[str] = []
    total = 0.0
    seen: set[str] = set()
    for position, criterion in enumerate(normalized, start=1):
        criterion_id = str(criterion.get("criterion_id") or "").strip()
        if not criterion_id:
            errors.append(f"criterion {position}: missing criterion id")
        elif criterion_id in seen:
            errors.append(f"criterion {position}: duplicate criterion id {criterion_id}")
        seen.add(criterion_id)

        if not str(criterion.get("description") or "").strip():
            errors.append(f"criterion {position}: describe what earns the marks")
        try:
            marks = float(criterion.get("max_marks") or 0)
        except (TypeError, ValueError):
            marks = 0.0
        if not math.isfinite(marks) or marks <= 0:
            errors.append(f"criterion {position}: assign marks greater than zero")
        total += marks

    if abs(total - max_marks) > _MARK_TOLERANCE:
        errors.append(
            f"criterion marks total {total:g}, but this question is worth {max_marks:g}"
        )
    return errors


def snapshot_criteria(value: Any) -> List[Dict[str, Any]]:
    """Make an immutable copy of criteria after finalisation validation."""

    return deepcopy(normalize_marking_criteria(value, assign_missing_ids=False))

