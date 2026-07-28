"""Shared mapped-answer contract for PCR authoring.

An uploaded answer document may contain either a worked solution or only an
answer key.  Both are valid teacher answers.  Persistent
``answer_question_mappings`` remain the source for worked solutions, while an
accepted answer-key candidate is exposed as a read-only virtual mapping.  This
keeps one contract for readiness, authoring APIs, and marking-plan preparation
without copying synthetic mapping rows into MongoDB.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


_VALID_ANSWER_LABELS = {"A", "B", "C", "D", "E", "F"}


def normalize_answer_label(value: Any) -> str:
    """Return a canonical single-letter answer label when one is present."""

    text = str(value or "").strip().upper()
    if text in _VALID_ANSWER_LABELS:
        return text
    match = re.search(
        r"\b(?:ANSWER|ANS|OPTION|CHOICE)?\s*(?:IS|:|-)?\s*\(?([A-F])\)?\b",
        text,
    )
    return match.group(1) if match else ""


def mapping_question_id(mapping: Dict[str, Any]) -> str:
    return str(
        mapping.get("question_id") or mapping.get("question_region_id") or ""
    ).strip()


def question_id(question: Dict[str, Any]) -> str:
    return str(question.get("id") or question.get("question_id") or "").strip()


def question_text(question: Dict[str, Any]) -> str:
    return str(question.get("question_text") or question.get("text") or "").strip()


def _candidate_for_question(
    document: Dict[str, Any],
    current_question_id: str,
) -> Optional[Dict[str, Any]]:
    for candidate in document.get("answer_key_candidates") or []:
        if str(candidate.get("question_id") or "").strip() == current_question_id:
            return candidate
    return None


def _option_text(question: Dict[str, Any], label: str) -> str:
    option_index = ord(label) - ord("A")
    enhanced_options = question.get("enhanced_options") or []
    if isinstance(enhanced_options, list):
        for index, option in enumerate(enhanced_options):
            if not isinstance(option, dict):
                continue
            option_label = normalize_answer_label(
                option.get("label") or option.get("key") or option.get("id")
            )
            if option_label != label and not (not option_label and index == option_index):
                continue
            content = str(
                option.get("content")
                or option.get("text")
                or option.get("value")
                or ""
            ).strip()
            if content:
                return content

    options = question.get("options") or []
    if isinstance(options, list) and 0 <= option_index < len(options):
        option = options[option_index]
        if isinstance(option, dict):
            return str(
                option.get("content") or option.get("text") or option.get("value") or ""
            ).strip()
        return str(option or "").strip()
    return ""


def _format_answer_text(label: str, option_text: str) -> str:
    if not option_text:
        return f"Option {label}"
    if re.match(rf"^\s*\(?{re.escape(label)}\)?\s*[\).:-]\s*", option_text, re.I):
        return option_text
    return f"{label}. {option_text}"


def build_answer_key_mapping(
    document: Dict[str, Any],
    question: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build a virtual mapped answer from the accepted canonical answer key."""

    current_question_id = question_id(question)
    if not current_question_id:
        return None

    candidate = _candidate_for_question(document, current_question_id)
    stored_label = normalize_answer_label(
        question.get("correct_answer") or question.get("correctAnswer")
    )
    candidate_label = normalize_answer_label((candidate or {}).get("correct_answer"))
    label = stored_label or candidate_label
    if not label:
        return None

    conflict = bool(stored_label and candidate_label and stored_label != candidate_label)
    candidate_needs_review = bool(
        (candidate or {}).get("needs_review")
        or (candidate or {}).get("manual_review_required")
    )
    # A saved canonical answer is trusted unless the uploaded key explicitly
    # disagrees with it. Candidate-only answers retain their OCR review state.
    manual_review_required = conflict or (not stored_label and candidate_needs_review)
    review_status = "needs_review" if manual_review_required else "accepted"
    try:
        confidence = float(
            (candidate or {}).get("confidence")
            if (candidate or {}).get("confidence") is not None
            else question.get("correct_answer_confidence", 1.0)
        )
    except (TypeError, ValueError):
        confidence = 1.0 if stored_label else 0.0
    confidence = max(0.0, min(1.0, confidence))

    option_text = _option_text(question, label)
    answer_only = _format_answer_text(label, option_text)
    mapped_question_text = question_text(question)
    evidence = str((candidate or {}).get("evidence") or "").strip()
    reasons: List[str] = []
    if conflict:
        reasons.append("answer_key_conflicts_with_saved_answer")
    elif stored_label and candidate_label:
        reasons.append("answer_key_matches_saved_answer")
    elif stored_label:
        reasons.append("saved_correct_answer")
    else:
        reasons.append("extracted_answer_key")

    document_id = str(document.get("document_id") or "").strip()
    return {
        "mapping_id": f"{document_id}:{current_question_id}:answer-key",
        "document_id": document_id,
        "question_id": current_question_id,
        "question_region_id": current_question_id,
        "answer_region_id": "",
        # An answer-key mapping is an objective answer, not a worked solution.
        # Keep the question text in its dedicated field and never duplicate it
        # into the answer consumed by either the UI or the grader.
        "answer_text": answer_only,
        "final_answer_text": answer_only,
        "mapped_question_text": mapped_question_text,
        "answer_kind": "answer_key",
        "virtual": True,
        "editable": False,
        "mapping_strategy": "answer_key",
        "source": "answer_key",
        "confidence": confidence,
        "manual_review_required": manual_review_required,
        "review_status": review_status,
        "correct_answer_candidate": label,
        "correct_answer_confidence": confidence,
        "correct_option_verified": not manual_review_required,
        "mapping_reasons": reasons,
        "mapping_evidence": evidence,
        "mapping_notes": (
            f"Uploaded key {candidate_label} conflicts with saved answer {stored_label}."
            if conflict
            else "Accepted answer key mapped to the question."
        ),
        "solution_images": [],
    }


def answer_mapping_rank(mapping: Dict[str, Any]) -> tuple:
    """Rank accepted evidence first, then prefer richer worked solutions."""

    review_status = str(mapping.get("review_status") or "").strip().lower()
    if review_status in {"accepted", "trusted"} and not mapping.get(
        "manual_review_required"
    ):
        review_rank = 30
    elif review_status == "rejected":
        review_rank = -20
    else:
        review_rank = 0

    source = str(mapping.get("source") or "").strip().lower()
    strategy = str(mapping.get("mapping_strategy") or "").strip().lower()
    if source == "manual_answer_segmentation":
        source_rank = 50
    elif source == "answer_sheet_full_ocr":
        source_rank = 40
    elif source in {"answer_sheet", "uploaded_answer_sheet", "upload"}:
        source_rank = 35
    elif source == "answer_key" or strategy == "answer_key":
        source_rank = 25
    elif source == "ai_generated" or strategy == "ai_generated_solution":
        source_rank = 10
    else:
        source_rank = 0

    try:
        confidence_rank = float(mapping.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence_rank = 0.0
    return (review_rank, source_rank, confidence_rank)


def select_effective_answer_mapping(
    document: Dict[str, Any],
    question: Dict[str, Any],
    mappings: Iterable[Dict[str, Any]],
    *,
    include_answer_key: bool = True,
) -> Optional[Dict[str, Any]]:
    """Select the best usable answer across worked and key-only evidence."""

    current_question_id = question_id(question)
    candidates = [
        mapping
        for mapping in mappings or []
        if mapping_question_id(mapping) == current_question_id
        and str(
            mapping.get("final_answer_text") or mapping.get("answer_text") or ""
        ).strip()
        and str(mapping.get("review_status") or "").strip().lower() != "rejected"
    ]
    if include_answer_key:
        answer_key_mapping = build_answer_key_mapping(document, question)
        if answer_key_mapping is not None:
            candidates.append(answer_key_mapping)
    return max(candidates, key=answer_mapping_rank) if candidates else None


def effective_answer_mappings(
    document: Dict[str, Any],
    questions: Iterable[Dict[str, Any]],
    mappings: Iterable[Dict[str, Any]],
    *,
    include_answer_key: bool = True,
) -> List[Dict[str, Any]]:
    """Return at most one effective mapped answer for every current question."""

    raw_mappings = list(mappings or [])
    selected: List[Dict[str, Any]] = []
    for question in questions or []:
        mapping = select_effective_answer_mapping(
            document,
            question,
            raw_mappings,
            include_answer_key=include_answer_key,
        )
        if mapping is not None:
            selected.append(mapping)
    return selected
