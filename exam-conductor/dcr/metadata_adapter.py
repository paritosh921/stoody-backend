"""
DCR Metadata Adapter
====================

Translates existing tutor-side question/exam data from the Stoody backend
into the ``AnswerKey`` format expected by DCR's ``TemplateMatcher``
(DUAL_MODE_ARCHITECTURE.md Section 4).

This module bridges the tutor's question-paper workflow with ExamPen's
DCR evaluation engine.  It does NOT rebuild the question upload path; it
reuses existing tutor/backend data and maps it to DCR shapes.

Ownership Declaration
---------------------
- Writes: nothing directly (produces ``AnswerKey`` instances)
- Reads from: tutor-side question/paper documents (passed as dicts)
- Never writes to: exampen_dcr_results, evalpen_submissions,
  evalpen_answer_pages

References
----------
- AnswerKey model: dcr/models.py
- DCR engine contract: new-docs/architecture/DUAL_MODE_ARCHITECTURE.md
  Section 4.2, 4.5
- Constraint C2: archiveDCR is backup only
- Constraint C5: Reuse existing tutor/backend question-paper path
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .models import AnswerKey

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Match mode inference
# ---------------------------------------------------------------------------

# Question types that strongly suggest a particular match mode.
_QUESTION_TYPE_MATCH_MODES: Dict[str, str] = {
    "mcq": "case_insensitive",
    "true_false": "case_insensitive",
    "fill_in_blank": "case_insensitive",
    "integer": "numeric",
    "numerical": "numeric",
}


def _infer_match_mode(
    question_type: Optional[str],
    expected_text: str,
    explicit_mode: Optional[str] = None,
) -> Optional[str]:
    """Infer the DCR match mode hint for the ``TemplateMatcher``.

    Priority:
    1. Explicit ``match_mode`` if already provided and valid.
    2. ``question_type`` mapping.
    3. Numeric detection on ``expected_text``.
    4. ``None`` — let the matcher apply all heuristics.

    Valid modes: ``exact``, ``numeric``, ``case_insensitive``.
    """
    valid_modes = {"exact", "numeric", "case_insensitive"}

    if explicit_mode and explicit_mode in valid_modes:
        return explicit_mode

    if question_type:
        qt_lower = question_type.lower().replace(" ", "_").replace("-", "_")
        mapped = _QUESTION_TYPE_MATCH_MODES.get(qt_lower)
        if mapped:
            return mapped

    # Attempt numeric detection on the expected text
    if expected_text:
        stripped = expected_text.strip()
        try:
            float(stripped)
            return "numeric"
        except ValueError:
            pass
        # Handle common numeric patterns like "42.0 cm", "-3/2"
        numeric_chars = sum(1 for c in stripped if c.isdigit() or c in ".,-/")
        if len(stripped) > 0 and numeric_chars / len(stripped) > 0.6:
            return "numeric"

    return None


# ---------------------------------------------------------------------------
# Numeric tolerance defaults
# ---------------------------------------------------------------------------

def _infer_numeric_tolerance(
    match_mode: Optional[str],
    explicit_tolerance: Optional[float] = None,
) -> Optional[float]:
    """Infer numeric matching tolerance (DCR-02 mitigation).

    Returns ``None`` for non-numeric match modes.  Default tolerance
    is ``0.01`` for numeric matching unless explicitly overridden.
    """
    if explicit_tolerance is not None and explicit_tolerance >= 0:
        return explicit_tolerance

    if match_mode == "numeric":
        return 0.01

    return None


# ---------------------------------------------------------------------------
# Public adapters
# ---------------------------------------------------------------------------

def adapt_question_to_dcr(
    question_doc: Dict[str, Any],
    *,
    page_number: Optional[int] = None,
) -> Optional[AnswerKey]:
    """Translate an existing backend question document to a DCR ``AnswerKey``.

    Parameters
    ----------
    question_doc : dict
        A question document from the ``questions`` or ``question_papers``
        collections.  Expected fields (graceful defaults for missing ones):

        - ``id`` or ``question_id`` — unique question identifier
        - ``correctAnswer`` or ``correct_answer`` or ``expected_text``
          — the expected answer text
        - ``marks`` or ``max_marks`` or ``max_score`` — maximum marks
        - ``question_type`` — e.g. "mcq", "integer", "fill_in_blank"
        - ``match_mode`` — explicit DCR match mode override
        - ``numeric_tolerance`` — explicit numeric tolerance override
        - ``page_number`` — expected page number for the answer

    page_number : int, optional
        Override page number if not present in the question doc.

    Returns
    -------
    AnswerKey or None
        ``None`` if the question has no usable expected answer (e.g.
        subjective questions that DCR cannot evaluate).
    """
    question_id = (
        question_doc.get("question_id")
        or question_doc.get("id")
        or ""
    )
    if not question_id:
        logger.warning(
            "adapt_question_to_dcr: question_doc has no id field; skipping."
        )
        return None

    # Resolve expected text from multiple possible field names
    expected_text = (
        question_doc.get("expected_text")
        or question_doc.get("correctAnswer")
        or question_doc.get("correct_answer")
        or ""
    ).strip()

    if not expected_text:
        # DCR cannot evaluate questions without an expected answer
        logger.debug(
            "adapt_question_to_dcr: question %s has no expected answer; "
            "skipping for DCR.",
            question_id,
        )
        return None

    # Resolve max score from multiple possible field names
    max_score_raw = (
        question_doc.get("max_score")
        or question_doc.get("max_marks")
        or question_doc.get("marks")
        or 1
    )
    try:
        max_score = float(max_score_raw)
    except (TypeError, ValueError):
        max_score = 1.0

    question_type = question_doc.get("question_type")
    explicit_mode = question_doc.get("match_mode")
    explicit_tolerance = question_doc.get("numeric_tolerance")

    match_mode = _infer_match_mode(question_type, expected_text, explicit_mode)
    numeric_tolerance = _infer_numeric_tolerance(match_mode, explicit_tolerance)

    resolved_page = (
        page_number
        or question_doc.get("page_number")
    )

    return AnswerKey(
        question_id=question_id,
        expected_text=expected_text,
        max_score=max_score,
        match_mode=match_mode,
        numeric_tolerance=numeric_tolerance,
        page_number=resolved_page,
    )


def adapt_exam_to_answer_keys(
    exam_doc: Dict[str, Any],
    questions: List[Dict[str, Any]],
) -> List[AnswerKey]:
    """Batch-convert questions for a given exam into DCR ``AnswerKey`` list.

    Parameters
    ----------
    exam_doc : dict
        The exam/paper document (used for context like subject, but not
        required to carry question data itself).
    questions : list[dict]
        List of question documents belonging to this exam.

    Returns
    -------
    list[AnswerKey]
        Answer keys for all questions that have a usable expected answer.
        Questions without expected answers (e.g. subjective-only) are
        silently skipped.
    """
    answer_keys: List[AnswerKey] = []

    for idx, question_doc in enumerate(questions):
        ak = adapt_question_to_dcr(question_doc)
        if ak is not None:
            answer_keys.append(ak)

    if not answer_keys:
        logger.info(
            "adapt_exam_to_answer_keys: no DCR-compatible questions found "
            "for exam %s (all questions may be subjective).",
            exam_doc.get("id") or exam_doc.get("exam_id", "unknown"),
        )

    return answer_keys


def adapt_paper_to_answer_keys(
    paper_doc: Dict[str, Any],
) -> List[AnswerKey]:
    """Translate all questions in a question-paper document to DCR answer keys.

    Parameters
    ----------
    paper_doc : dict
        A ``question_papers`` document from the paper-builder.  Expected
        to have ``sections[].questions[]`` structure per
        ``paper_builder_async.py``.

    Returns
    -------
    list[AnswerKey]
        Answer keys for all DCR-compatible questions in the paper.
    """
    all_questions: List[Dict[str, Any]] = []

    for section in paper_doc.get("sections") or []:
        for question in section.get("questions") or []:
            all_questions.append(question)

    return adapt_exam_to_answer_keys(paper_doc, all_questions)
