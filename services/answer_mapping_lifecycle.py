"""Backward-compatible lifecycle helpers for uploaded answer mapping."""

from __future__ import annotations

from typing import Any, Dict, Optional


ANSWER_MAPPING_ACTIVE_STATUSES = frozenset({"queued", "extracting", "waiting_for_questions", "mapping"})
ANSWER_MAPPING_TERMINAL_STATUSES = frozenset({"completed", "needs_review", "error", "not_expected"})
ANSWER_MAPPING_STATUSES = ANSWER_MAPPING_ACTIVE_STATUSES | ANSWER_MAPPING_TERMINAL_STATUSES | {"not_started"}


def completed_answer_mapping_status(
    *,
    mapped_count: int,
    question_count: int,
    manual_review_count: int = 0,
) -> str:
    """Return the terminal state after a mapping attempt."""

    return (
        "completed"
        if question_count > 0
        and mapped_count >= question_count
        and manual_review_count <= 0
        else "needs_review"
    )


def effective_answer_mapping_status(
    document: Dict[str, Any],
    *,
    question_count: Optional[int] = None,
) -> str:
    """Read new lifecycle fields while deriving safe states for legacy records."""

    explicit = str(document.get("answer_mapping_status") or "").strip().lower()
    if explicit in ANSWER_MAPPING_STATUSES:
        return explicit

    mode = str(document.get("answer_solution_mode") or "").strip().lower()
    has_upload = bool(document.get("answer_sheet_path"))
    if mode != "upload" and not has_upload:
        return "not_expected"

    answer_ocr_status = str(document.get("answer_sheet_ocr_status") or "not_processed").strip().lower()
    if answer_ocr_status == "error":
        return "error"
    if answer_ocr_status == "processing":
        return "extracting"
    if answer_ocr_status in {"", "not_processed"}:
        return "not_started"

    question_ocr_status = str(document.get("ocr_status") or "not_processed").strip().lower()
    if question_ocr_status != "completed":
        return "waiting_for_questions"

    mapped_count = int(document.get("answer_sheet_mapped_answers_count") or 0)
    expected_count = int(question_count or document.get("extracted_questions_count") or 0)
    summary = document.get("answer_sheet_mapping_summary") or {}
    manual_review_count = int(summary.get("manual_review_count") or 0)
    if expected_count > 0 and mapped_count >= expected_count and manual_review_count <= 0:
        return "completed"

    # A legacy record with terminal OCR and incomplete mappings is reviewable;
    # never pretend an untracked background job is still running forever.
    return "needs_review"

