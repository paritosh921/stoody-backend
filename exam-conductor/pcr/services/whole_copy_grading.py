"""Compact holistic grading contract for PCR visual answer copies.

The model sees the immutable paper, solution, locked catalog, and every student
page in one request.  It returns marks and physical page references only.  The
server owns score totals and turns those page references into stable full-page
evidence records during validation.

This module intentionally contains no evidence-mapping, crop, recursive split,
or retry orchestration.  The caller may make one primary request and at most
one targeted recovery request.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

from ..language_assessment import language_feedback_schema


PROMPT_VERSION = "pcr-full-document-visual-v16"
PIPELINE_VERSION = 7
MAPPING_PIPELINE_VERSION = "whole-copy-rubric-v7"
REQUIRED_PROCESSING_PATH = "full_document_visual"

_CONTENT_TYPES = ("TEXT_ONLY", "DIAGRAM_HEAVY", "TABLE_PRESENT", "MIXED")
_ATTEMPT_STATUSES = ("attempted", "not_attempted", "unresolved")
_CREDIT_BASES = ("direct_evidence", "error_carried_forward", "no_credit")


def system_instructions() -> str:
    """Return the single operational prompt used by both bounded calls."""

    return (
        "You are grading one complete handwritten student answer copy. Read the "
        "original question paper, teacher solution or marking scheme, locked marking "
        "catalog, and every labelled student page directly. Pages may be dark, faint, "
        "angled, sideways, photographed poorly, marked by a teacher, or written in "
        "Hindi or another language. Alternate orientation images with the same page "
        "number are duplicate views of one physical page, never additional work.\n\n"
        "Locate each answer across the complete copy before grading. Several answers "
        "may share a page; one answer may be continued, jumbled, or written on distant "
        "pages. Associate work using visible question numbers, wording, topic, variables, "
        "answer structure, diagrams, tables, and continuity together. Incorrect or "
        "incomplete work is still an attempt. Use not_attempted only after checking every "
        "physical page and finding no relevant work. Use unresolved only when meaning or "
        "ownership genuinely prevents a reliable award after using the readable views.\n\n"
        "Grade the student's visible work, not teacher ticks, crosses, circles, written "
        "marks, or corrections. Apply the locked criteria and maximums exactly. Award "
        "credit for each correct visible step or diagram component and deduct only for "
        "missing or incorrect requirements. Do not assume correctness from fluent prose, "
        "matching keywords, or a teacher annotation. For diagrams, inspect labels, "
        "relationships, direction, construction, and requested components visually. "
        "Equivalent correct wording and methods receive credit. Before awarding a full "
        "score, actively check every criterion for a missing or incorrect detail and "
        "award full marks only when the cited visible evidence satisfies all of them.\n\n"
        "Return every requested catalog question exactly once. source_pages contains "
        "only physical answer-copy page numbers. student_answer is a concise faithful "
        "transcription or visual description of the student's work. Criterion evidence "
        "must state what is actually visible and why it earns or loses marks. Keep "
        "feedback short. When a catalog question includes language_feedback_profile, "
        "also return its seven diagnostic writing dimensions. This developmental profile "
        "never changes marks: criterion_marks remain the only score source. Follow the "
        "profile's applicability exactly, use not_applicable for excluded dimensions, "
        "and use not_assessed with empty evidence and feedback when the answer is not "
        "attempted or genuinely unresolved. Write feedback in the language used by the "
        "student where practical. For each "
        "applicable dimension, cite a short visible phrase or feature and give one concrete "
        "next step. Do not penalize one defect twice across dimensions, do not call uncertain "
        "handwriting a spelling or grammar error, and do not fabricate a correction. "
        "For every question without an enabled language_feedback_profile, assess only the "
        "locked subject criteria. Do not reduce marks or criticize spelling, grammar, "
        "punctuation, tone, style, or handwriting quality unless a locked criterion "
        "explicitly requires that skill. The language in which a Physics, Chemistry, "
        "Mathematics, or Biology answer is written never turns it into a language task. "
        "When response_selection is present, the question contains alternatives. Identify "
        "the assessment units the student actually attempted from visible labels and answer "
        "content, and return their unit IDs in visible answer order. Grade only the first "
        "required_count attempted units; do not emit criterion rows for unattempted alternatives. "
        "If more alternatives were attempted than permitted, keep the first required_count for "
        "the score and set needs_review=true so the teacher can confirm the intended selection. "
        "Set all_student_work_accounted=true only after every visible "
        "student answer block has been associated or explicitly represented as unresolved. "
        "Do not return coordinates, crops, OCR commentary, confidence thresholds, or "
        "implementation details."
    )


def whole_copy_schema(
    question_contracts: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Return a strict but compact ledger schema for the locked catalog."""

    variants = [
        _question_schema(contract, fallback_number=index)
        for index, contract in enumerate(question_contracts, start=1)
    ]
    if not variants:
        raise ValueError("Cannot build a whole-copy schema without questions")
    items: Dict[str, Any] = variants[0] if len(variants) == 1 else {"anyOf": variants}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "all_student_work_accounted": {"type": "boolean"},
            "questions": {
                "type": "array",
                "items": items,
                "minItems": len(variants),
                "maxItems": len(variants),
            },
        },
        "required": ["all_student_work_accounted", "questions"],
    }


def output_limit(
    question_contracts: Sequence[Mapping[str, Any]],
    *,
    reasoning_effort: str,
    recovery: bool = False,
) -> int:
    """Size one response ceiling without creating output-exhaustion retries.

    The Responses API charges actual generated tokens, not this ceiling.  A
    sufficient ceiling avoids discarding a nearly complete structured ledger.
    """

    question_count = max(1, len(question_contracts))
    criterion_count = sum(
        len(contract.get("marking_criteria") or [])
        for contract in question_contracts
        if isinstance(contract, Mapping)
    )
    language_dimension_count = sum(
        len((contract.get("language_feedback_profile") or {}).get("dimensions") or [])
        for contract in question_contracts
        if isinstance(contract, Mapping)
        and isinstance(contract.get("language_feedback_profile"), Mapping)
        and (contract.get("language_feedback_profile") or {}).get("enabled")
    )
    reserve = {
        "none": 1_000,
        "minimal": 2_000,
        "low": 4_000,
        "medium": 7_000,
        "high": 10_000,
    }.get(str(reasoning_effort or "").strip().lower(), 7_000)
    visible = (
        1_500
        + question_count * 650
        + max(1, criterion_count) * 650
        + language_dimension_count * 180
    )
    if recovery:
        visible = (
            1_000
            + question_count * 550
            + max(1, criterion_count) * 550
            + language_dimension_count * 160
        )
    rounded = int(math.ceil((reserve + visible) / 2_000.0) * 2_000)
    lower = 12_000 if recovery else 24_000
    upper = 24_000 if recovery else 32_000
    return min(upper, max(lower, rounded))


def normalize_payload(payload: Any) -> Dict[str, Any]:
    """Adapt the compact page ledger to the shared materialization validator.

    No semantic mark is changed.  Physical page numbers become the validator's
    legacy full-page evidence records, and criterion rows cite those server-
    generated stable IDs.
    """

    raw = payload if isinstance(payload, Mapping) else {}
    normalized_questions = []
    for raw_question in raw.get("questions") or []:
        if not isinstance(raw_question, Mapping):
            continue
        question = dict(raw_question)
        try:
            number = int(question.get("question_number") or 0)
        except (TypeError, ValueError):
            number = 0
        pages = []
        for value in question.get("source_pages") or []:
            try:
                page = int(value)
            except (TypeError, ValueError):
                continue
            if page > 0 and page not in pages:
                pages.append(page)
        question["source_pages"] = pages
        evidence_ids = [f"q{number}-legacy-page-{page}" for page in pages]
        marks = []
        for raw_mark in question.get("criterion_marks") or []:
            if not isinstance(raw_mark, Mapping):
                continue
            mark = dict(raw_mark)
            mark["evidence_region_ids"] = list(evidence_ids)
            marks.append(mark)
        question["criterion_marks"] = marks
        normalized_questions.append(question)
    accounted = bool(raw.get("all_student_work_accounted"))
    return {
        "document_review": {
            "all_student_work_accounted": accounted,
            "warnings": [] if accounted else [
                "The holistic grader did not account for every visible student answer block"
            ],
        },
        "questions": normalized_questions,
    }


def merge_recovery_payload(
    primary: Mapping[str, Any],
    recovery: Mapping[str, Any],
    *,
    recovered_question_numbers: Sequence[int],
) -> Dict[str, Any]:
    """Replace only requested question rows returned exactly once by recovery."""

    requested = {int(number) for number in recovered_question_numbers if int(number) > 0}
    primary_rows = [
        dict(row) for row in primary.get("questions") or []
        if isinstance(row, Mapping)
    ]
    recovery_rows = [
        dict(row) for row in recovery.get("questions") or []
        if isinstance(row, Mapping)
    ]
    recovery_by_number: Dict[int, list[Dict[str, Any]]] = {}
    for row in recovery_rows:
        try:
            number = int(row.get("question_number") or 0)
        except (TypeError, ValueError):
            continue
        if number in requested:
            recovery_by_number.setdefault(number, []).append(row)
    replacements = {
        number: rows[0]
        for number, rows in recovery_by_number.items()
        if len(rows) == 1
    }
    merged = []
    seen = set()
    for row in primary_rows:
        try:
            number = int(row.get("question_number") or 0)
        except (TypeError, ValueError):
            number = 0
        if number in replacements:
            merged.append(replacements[number])
            seen.add(number)
        else:
            merged.append(row)
            if number:
                seen.add(number)
    for number in sorted(replacements):
        if number not in seen:
            merged.append(replacements[number])
    return {
        "all_student_work_accounted": bool(
            primary.get("all_student_work_accounted")
            or recovery.get("all_student_work_accounted")
        ),
        "questions": merged,
    }


def _question_schema(
    contract: Mapping[str, Any],
    *,
    fallback_number: int,
) -> Dict[str, Any]:
    number = _positive_int(contract.get("question_number")) or fallback_number
    objective = str(contract.get("grading_mode") or "").strip().lower() == "objective"
    criteria = [] if objective else [
        item for item in (contract.get("marking_criteria") or [])
        if isinstance(item, Mapping)
    ]
    criterion_variants = [_criterion_schema(item) for item in criteria]
    criterion_items: Dict[str, Any]
    if not criterion_variants:
        criterion_items = _criterion_schema(None)
    elif len(criterion_variants) == 1:
        criterion_items = criterion_variants[0]
    else:
        criterion_items = {"anyOf": criterion_variants}
    maximum = _finite_nonnegative(contract.get("max_marks"))
    properties: Dict[str, Any] = {
        "question_number": {"type": "integer", "enum": [number]},
        "attempt_status": {"type": "string", "enum": list(_ATTEMPT_STATUSES)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "student_answer": {"type": "string", "maxLength": 5000},
        "content_type": {"type": "string", "enum": list(_CONTENT_TYPES)},
        "source_pages": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "maxItems": 50,
        },
        "criterion_marks": {
            "type": "array",
            "items": criterion_items,
            "minItems": 0,
            "maxItems": len(criteria),
        },
        "total_score": {"type": "number", "minimum": 0, "maximum": maximum},
        "overall_feedback": {"type": "string", "maxLength": 1000},
        "needs_review": {"type": "boolean"},
        "review_reason": {"type": "string", "maxLength": 1000},
    }
    response_selection = contract.get("response_selection")
    if isinstance(response_selection, Mapping):
        available_ids = [
            str(value or "").strip()
            for value in response_selection.get("available_unit_ids") or []
            if str(value or "").strip()
        ]
        if available_ids:
            properties["attempted_unit_ids"] = {
                "type": "array",
                "items": {"type": "string", "enum": available_ids},
                "minItems": 0,
                "maxItems": len(available_ids),
            }
    required = list(properties)
    profile = contract.get("language_feedback_profile")
    feedback_schema = (
        language_feedback_schema(profile)
        if isinstance(profile, Mapping)
        else None
    )
    if feedback_schema is not None:
        properties["language_feedback"] = feedback_schema
        required.append("language_feedback")
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _criterion_schema(criterion: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    criterion_id = str((criterion or {}).get("criterion_id") or "")
    maximum = _finite_nonnegative((criterion or {}).get("max_marks"))
    properties: Dict[str, Any] = {
        "criterion_id": (
            {"type": "string", "enum": [criterion_id]}
            if criterion_id else {"type": "string"}
        ),
        "marks_awarded": {"type": "number", "minimum": 0, "maximum": maximum},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string", "maxLength": 1200},
        "evidence": {"type": "string", "maxLength": 1200},
        "credit_basis": {"type": "string", "enum": list(_CREDIT_BASES)},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _finite_nonnegative(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    if not math.isfinite(parsed):
        parsed = 0.0
    return max(0.0, parsed)
