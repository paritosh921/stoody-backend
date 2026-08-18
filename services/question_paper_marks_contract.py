"""Question-paper mark ownership and validation.

The historical PDF pipeline stored ``4`` whenever OCR could not read a mark.
That made an unknown value indistinguishable from a real four-mark question.
This module provides a migration-free compatibility boundary:

* readable printed marks remain authoritative;
* teacher edits are an explicit, authoritative confirmation;
* missing, unclear, and legacy implicit marks use a provisional one-mark
  authoring budget instead of blocking the whole paper; and
* provenance is retained so provisional marks remain visible for review.

No function in this module mutates a persisted document.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional


VERIFIED_MARK_STATUSES = {"verified", "teacher_confirmed"}
UNRESOLVED_MARK_STATUS = "unresolved"
PROVISIONAL_MARK_STATUS = "provisional_default"
PROVISIONAL_DEFAULT_MARKS = 1.0
_ARITHMETIC_MARK_RE = re.compile(
    r"(?P<count>\d+(?:\.\d+)?)\s*[xX×*]\s*"
    r"(?P<each>\d+(?:\.\d+)?)\s*=\s*(?P<total>\d+(?:\.\d+)?)"
)
_NUMBER_RE = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])")


def positive_marks(value: Any) -> Optional[float]:
    """Return a finite positive mark value, otherwise ``None``."""

    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def question_marks_status(question: Dict[str, Any]) -> str:
    """Resolve a question's marks state without changing legacy data."""

    metadata = question.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    explicit_status = str(metadata.get("marks_status") or "").strip().lower()
    source = str(metadata.get("marks_source") or "").strip().lower()
    has_positive_marks = any(
        positive_marks(question.get(key)) is not None
        for key in ("points", "max_marks", "marks", "total_points")
    )

    if explicit_status == "teacher_confirmed" or source == "teacher":
        return "teacher_confirmed" if has_positive_marks else PROVISIONAL_MARK_STATUS
    if explicit_status == "verified":
        return "verified" if has_positive_marks else PROVISIONAL_MARK_STATUS
    if explicit_status in {UNRESOLVED_MARK_STATUS, PROVISIONAL_MARK_STATUS}:
        return PROVISIONAL_MARK_STATUS

    # This is the exact signature of the old fabricated-four fallback.
    if metadata.get("max_marks_extracted") is False:
        return PROVISIONAL_MARK_STATUS

    # Manually-created questions and older successfully-extracted questions do
    # not have the new status field. Preserve those existing valid contracts.
    return "legacy_verified" if has_positive_marks else PROVISIONAL_MARK_STATUS


def effective_question_marks(question: Dict[str, Any]) -> Optional[float]:
    """Return the authoritative or provisional runtime mark budget."""

    if question_marks_status(question) == PROVISIONAL_MARK_STATUS:
        return PROVISIONAL_DEFAULT_MARKS
    for key in ("points", "max_marks", "marks", "total_points"):
        value = positive_marks(question.get(key))
        if value is not None:
            return value
    return None


def project_question_marks_for_authoring(question: Dict[str, Any]) -> Dict[str, Any]:
    """Return a read-time authoring projection for old and new questions.

    The stored legacy value remains untouched. Clients receive a provisional
    one-mark value and explicit review metadata, so authoring can continue
    without presenting the fallback as printed or teacher-confirmed evidence.
    """

    projected = dict(question)
    metadata = dict(projected.get("metadata") or {})
    status = question_marks_status(projected)
    metadata["marks_status"] = status
    metadata["marks_review_required"] = status == PROVISIONAL_MARK_STATUS
    if status == PROVISIONAL_MARK_STATUS:
        projected["points"] = PROVISIONAL_DEFAULT_MARKS
        metadata["marks_source"] = "system_default"
        metadata["max_marks_extracted"] = False
        metadata.setdefault(
            "marks_review_reason",
            "Printed marks were not verified, so this question provisionally uses 1 mark.",
        )
    projected["metadata"] = metadata
    return projected


def teacher_confirmed_marks_metadata(
    existing_metadata: Any,
    *,
    actor_id: Any,
    confirmed_at: Any,
) -> Dict[str, Any]:
    """Build metadata written when a teacher explicitly saves marks."""

    metadata = dict(existing_metadata or {}) if isinstance(existing_metadata, dict) else {}
    metadata.update(
        {
            "max_marks_extracted": True,
            "marks_status": "teacher_confirmed",
            "marks_source": "teacher",
            "marks_review_required": False,
            "marks_review_reason": None,
            "marks_confirmed_by": actor_id,
            "marks_confirmed_at": confirmed_at,
        }
    )
    return metadata


def _normalized_bbox(value: Any) -> Optional[Dict[str, float]]:
    if not isinstance(value, dict):
        return None
    aliases = {
        "x0": ("x0", "left", "x", "top_left_x"),
        "y0": ("y0", "top", "y", "top_left_y"),
        "x1": ("x1", "right", "bottom_right_x"),
        "y1": ("y1", "bottom", "bottom_right_y"),
    }
    normalized: Dict[str, float] = {}
    for target, keys in aliases.items():
        raw = next((value.get(key) for key in keys if value.get(key) is not None), None)
        try:
            number = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        normalized[target] = min(1000.0, max(0.0, number))
    if normalized["x1"] <= normalized["x0"] or normalized["y1"] <= normalized["y0"]:
        return None
    return normalized


def validate_visual_marks_evidence(raw: Any) -> Dict[str, Any]:
    """Validate one model-returned mark against its printed visual evidence."""

    evidence = raw if isinstance(raw, dict) else {}
    value = positive_marks(evidence.get("value", evidence.get("max_marks")))
    printed_text = str(
        evidence.get("printed_text") or evidence.get("text") or evidence.get("raw_text") or ""
    ).strip()
    try:
        confidence = float(evidence.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0

    result: Dict[str, Any] = {
        "value": value,
        "printed_text": printed_text,
        "page": evidence.get("page"),
        "bbox": _normalized_bbox(evidence.get("bbox")),
        "coordinate_space": "normalized_1000",
        "confidence": confidence,
        "verified": False,
        "reason": None,
    }
    if value is None:
        result["reason"] = "no positive printed mark was identified"
        return result
    if not printed_text:
        result["reason"] = "printed mark text is missing"
        return result
    if confidence < 0.70:
        result["reason"] = "printed mark confidence is below 0.70"
        return result

    formula = _ARITHMETIC_MARK_RE.search(printed_text.replace(",", ""))
    if formula:
        count = float(formula.group("count"))
        each = float(formula.group("each"))
        total = float(formula.group("total"))
        if not math.isclose(count * each, total, abs_tol=0.01):
            result["reason"] = "printed marks formula is arithmetically inconsistent"
            return result
        if not math.isclose(value, total, abs_tol=0.01):
            result["reason"] = "returned mark disagrees with the printed formula"
            return result
    else:
        visible_numbers = [float(item) for item in _NUMBER_RE.findall(printed_text)]
        if not any(math.isclose(value, item, abs_tol=0.01) for item in visible_numbers):
            result["reason"] = "returned mark is not present in the printed evidence"
            return result

    result["verified"] = True
    return result


def extracted_marks_metadata(raw_question: Dict[str, Any], *, visual_source: bool) -> Dict[str, Any]:
    """Produce marks metadata and a nullable point value for one extraction."""

    if visual_source:
        raw_evidence = raw_question.get("marks_evidence") or raw_question.get("marks")
        evidence = validate_visual_marks_evidence(raw_evidence)
        verified = bool(evidence.get("verified"))
        return {
            "points": evidence.get("value") if verified else PROVISIONAL_DEFAULT_MARKS,
            "max_marks_extracted": verified,
            "marks_status": "verified" if verified else PROVISIONAL_MARK_STATUS,
            "marks_source": "visual_printed_evidence" if verified else "system_default",
            "marks_evidence": evidence,
            "marks_review_required": not verified,
            "marks_review_reason": None if verified else evidence.get("reason"),
        }

    value = None
    for key in ("max_marks", "marks", "points"):
        value = positive_marks(raw_question.get(key))
        if value is not None:
            break
    return {
        "points": value if value is not None else PROVISIONAL_DEFAULT_MARKS,
        "max_marks_extracted": value is not None,
        "marks_status": "verified" if value is not None else PROVISIONAL_MARK_STATUS,
        "marks_source": "ocr_text" if value is not None else "system_default",
        "marks_evidence": None,
        "marks_review_required": value is None,
        "marks_review_reason": (
            None
            if value is not None
            else "Marks were not explicit in the paper, so this question provisionally uses 1 mark."
        ),
    }


def visual_paper_total(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    raw = raw_data.get("paper_total_marks") or raw_data.get("paper_total_evidence")
    if isinstance(raw, (int, float, str)):
        raw = {"value": raw, "printed_text": str(raw), "confidence": 0.0}
    return validate_visual_marks_evidence(raw)


def summarize_question_marks(
    questions: Iterable[Dict[str, Any]],
    *,
    expected_total: Any = None,
) -> Dict[str, Any]:
    question_list = list(questions)
    values: List[float] = []
    unresolved_ids: List[str] = []
    provisional_ids: List[str] = []
    authoritative_count = 0
    for index, question in enumerate(question_list, start=1):
        question_id = str(question.get("id") or question.get("question_number") or index)
        mark_status = question_marks_status(question)
        marks = effective_question_marks(question)
        if marks is None:
            unresolved_ids.append(question_id)
        else:
            values.append(marks)
            if mark_status == PROVISIONAL_MARK_STATUS:
                provisional_ids.append(question_id)
            else:
                authoritative_count += 1

    expected = positive_marks(expected_total)
    calculated = sum(values)
    reconciled: Optional[bool]
    if unresolved_ids:
        reconciled = False
    elif expected is None:
        reconciled = None
    else:
        reconciled = math.isclose(calculated, expected, abs_tol=0.01)
    return {
        "question_count": len(question_list),
        "resolved_count": len(values),
        "authoritative_count": authoritative_count,
        "unresolved_count": len(unresolved_ids),
        "unresolved_question_ids": unresolved_ids,
        "provisional_count": len(provisional_ids),
        "provisional_question_ids": provisional_ids,
        "review_required_count": len(provisional_ids) + len(unresolved_ids),
        "calculated_total": calculated,
        "expected_total": expected,
        "reconciled": reconciled,
    }
