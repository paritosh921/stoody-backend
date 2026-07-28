"""
PCR Metadata Adapter
====================

Translates existing tutor-side question/exam data from the Stoody backend
into the format expected by ``evalpen_questions`` (PCR_EVAL_ENGINE_SPEC
Section 7.4).

This module bridges the tutor's question-paper workflow with ExamPen's
PCR evaluation engine.  It does NOT rebuild the question upload path; it
reuses existing tutor/backend data and maps it to PCR shapes.

Ownership Declaration
---------------------
- Writes: nothing directly (produces dicts for ``QuestionRepository``)
- Reads from: tutor-side question/paper documents (passed as dicts)
- Never writes to: evalpen_submissions, evalpen_answer_pages,
  evalpen_detected_responses, practice persistence

References
----------
- PCR questions collection: new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md
  Section 7.4 (evalpen_questions)
- Complexity router: PCR_EVAL_ENGINE_SPEC Section 5.2
- Template families: PCR_EVAL_ENGINE_SPEC Section 5.3
- Constraint C5: Reuse existing tutor/backend question-paper path
"""

from __future__ import annotations

import logging
import copy
from typing import Any, Dict, List, Optional

from services.answer_mapping_contract import normalize_answer_label

from .marking_policy import normalize_marking_criteria, normalize_method_policy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Complexity inference (PCR_EVAL_ENGINE_SPEC Section 5.2)
# ---------------------------------------------------------------------------

# Mapping from question_type to default complexity when not explicitly set.
_QUESTION_TYPE_COMPLEXITY: Dict[str, str] = {
    "mcq": "L1",
    "fill_in_blank": "L1",
    "true_false": "L1",
    "integer": "L1",
    "short_answer": "L1",
    "factual_recall": "L1",
    "definition": "L2",
    "numerical": "L2",
    "derivation": "L2",
    "proof": "L3",
    "essay": "L3",
    "subjective": "L2",        # default for generic subjective
    "long_answer": "L3",
}

# Marks-based fallback when question_type is unknown or absent.
_MARKS_COMPLEXITY_THRESHOLDS = [
    (2, "L1"),   # up to 2 marks -> L1
    (5, "L2"),   # 3-5 marks -> L2
]
# Anything above 5 -> L3


def _infer_complexity(
    question_type: Optional[str],
    max_marks: Optional[int],
    explicit_complexity: Optional[str] = None,
) -> str:
    """Infer PCR complexity tier (L1/L2/L3).

    Priority:
    1. Explicit ``complexity`` if already set and valid.
    2. ``question_type`` mapping table.
    3. ``max_marks`` threshold fallback.
    4. Default: ``L2``.
    """
    valid_complexities = {"L1", "L2", "L3"}

    if explicit_complexity and explicit_complexity.upper() in valid_complexities:
        return explicit_complexity.upper()

    if question_type:
        qt_lower = question_type.lower().replace(" ", "_").replace("-", "_")
        mapped = _QUESTION_TYPE_COMPLEXITY.get(qt_lower)
        if mapped:
            return mapped

    if max_marks is not None:
        for threshold, tier in _MARKS_COMPLEXITY_THRESHOLDS:
            if max_marks <= threshold:
                return tier
        return "L3"

    return "L2"


# ---------------------------------------------------------------------------
# Eval template inference (PCR_EVAL_ENGINE_SPEC Section 5.3)
# ---------------------------------------------------------------------------

# Subject-level defaults when question_type does not provide enough signal.
_SUBJECT_TEMPLATE_DEFAULTS: Dict[str, str] = {
    "mathematics": "stepwise_numerical",
    "math": "stepwise_numerical",
    "maths": "stepwise_numerical",
    "physics": "stepwise_numerical",
    "chemistry": "stepwise_numerical",
    "accountancy": "ledger_tabular",
    "accounts": "ledger_tabular",
    "english": "essay_rubric",
    "hindi": "essay_rubric",
    "history": "essay_rubric",
    "geography": "keyword_coverage",
    "biology": "keyword_coverage",
    "science": "keyword_coverage",
    "economics": "keyword_coverage",
    "political_science": "essay_rubric",
    "civics": "essay_rubric",
}

_QUESTION_TYPE_TEMPLATES: Dict[str, str] = {
    "mcq": "factual_recall",
    "fill_in_blank": "factual_recall",
    "true_false": "factual_recall",
    "integer": "stepwise_numerical",
    "numerical": "stepwise_numerical",
    "derivation": "proof_derivation",
    "proof": "proof_derivation",
    "definition": "keyword_coverage",
    "short_answer": "factual_recall",
    "essay": "essay_rubric",
    "long_answer": "essay_rubric",
    "subjective": "keyword_coverage",
}


def _infer_eval_template(
    question_type: Optional[str],
    subject: Optional[str],
    explicit_template: Optional[str] = None,
) -> str:
    """Infer the PCR eval template family.

    Priority:
    1. Explicit ``eval_template`` if already set.
    2. ``question_type`` mapping.
    3. ``subject`` default.
    4. Fallback: ``keyword_coverage``.
    """
    valid_templates = {
        "stepwise_numerical",
        "essay_rubric",
        "factual_recall",
        "keyword_coverage",
        "ledger_tabular",
        "proof_derivation",
    }

    if explicit_template and explicit_template in valid_templates:
        return explicit_template

    if question_type:
        qt_lower = question_type.lower().replace(" ", "_").replace("-", "_")
        mapped = _QUESTION_TYPE_TEMPLATES.get(qt_lower)
        if mapped:
            return mapped

    if subject:
        subj_lower = subject.lower().replace(" ", "_").replace("-", "_")
        mapped = _SUBJECT_TEMPLATE_DEFAULTS.get(subj_lower)
        if mapped:
            return mapped

    return "keyword_coverage"


# ---------------------------------------------------------------------------
# Diagram weight defaults
# ---------------------------------------------------------------------------

def _infer_diagram_weight(
    question_type: Optional[str],
    has_diagram: Optional[bool],
    explicit_weight: Optional[float] = None,
) -> float:
    """Infer diagram prorating weight.

    ``diagram_weight`` is used in:
        scoreable_marks = max_marks * (1 - diagram_weight)

    Returns 0.0 if no diagram expected, otherwise a default of 0.3
    (30% of marks attributed to diagram).
    """
    if explicit_weight is not None and 0.0 <= explicit_weight <= 1.0:
        return explicit_weight

    if has_diagram:
        return 0.3

    return 0.0


# ---------------------------------------------------------------------------
# Public adapters
# ---------------------------------------------------------------------------

def adapt_question_to_pcr(
    question_doc: Dict[str, Any],
    *,
    exam_id: Optional[str] = None,
    default_subject: Optional[str] = None,
) -> Dict[str, Any]:
    """Translate an existing backend question document to PCR ``evalpen_questions`` shape.

    Parameters
    ----------
    question_doc : dict
        A question document from the ``questions`` or ``question_papers``
        collections.  Expected fields (all optional for graceful degradation):

        - ``id`` or ``question_id`` — unique question identifier
        - ``text`` or ``question_text`` — question body
        - ``subject`` — subject name
        - ``question_type`` — e.g. "mcq", "subjective", "numerical"
        - ``difficulty`` — "easy" / "medium" / "hard"
        - ``marks`` or ``max_marks`` — marks for this question
        - ``has_diagram`` — whether a diagram is expected
        - ``diagram_weight`` — explicit diagram prorating weight
        - ``complexity`` — explicit PCR complexity (L1/L2/L3)
        - ``eval_template`` — explicit PCR eval template
        - ``correctAnswer`` or ``correct_answer`` — expected answer text
        - ``explanation`` — answer explanation (used as rubric hint)

    exam_id : str, optional
        Override exam_id if the question doc does not carry it.

    default_subject : str, optional
        Fallback subject when the question doc does not specify one.

    Returns
    -------
    dict
        A document matching the ``evalpen_questions`` collection shape
        (PCR_EVAL_ENGINE_SPEC Section 7.4).
    """
    question_id = (
        question_doc.get("question_id")
        or question_doc.get("id")
        or ""
    )
    if not question_id:
        logger.warning(
            "adapt_question_to_pcr: question_doc has no id field; "
            "generated doc will have empty question_id."
        )

    subject = question_doc.get("subject") or default_subject or ""
    question_type = question_doc.get("question_type", "subjective")

    # Marks: try various field names
    max_marks_raw = (
        question_doc.get("max_marks")
        or question_doc.get("marks")
        or question_doc.get("points")
        or question_doc.get("total_points")
        or 0
    )
    try:
        max_marks = float(max_marks_raw)
    except (TypeError, ValueError):
        max_marks = 0

    # Explicit overrides
    explicit_complexity = question_doc.get("complexity")
    explicit_template = question_doc.get("eval_template")
    has_diagram = question_doc.get("has_diagram", False)
    explicit_diagram_weight = question_doc.get("diagram_weight")

    complexity = _infer_complexity(question_type, max_marks, explicit_complexity)
    eval_template = _infer_eval_template(question_type, subject, explicit_template)
    diagram_weight = _infer_diagram_weight(
        question_type, has_diagram, explicit_diagram_weight
    )

    # Build rubric from explanation if available.
    # ``question_text`` and ``reference_solution`` are deliberately carried
    # into ExamPen metadata as well.  EvalCore uses them to build the marking
    # prompt; dropping them here meant a real conducted PCR paper could be
    # evaluated against a placeholder question and an LLM-generated key even
    # when the teacher had already supplied the approved marking material.
    rubric = question_doc.get("rubric")
    if not rubric and question_doc.get("explanation"):
        rubric = question_doc["explanation"]

    question_text = (
        question_doc.get("question_text")
        or question_doc.get("text")
        or question_doc.get("question")
        or ""
    )
    reference_solution = (
        question_doc.get("reference_solution")
        or question_doc.get("solution")
        or question_doc.get("answer")
        or question_doc.get("correct_answer")
        or question_doc.get("correctAnswer")
        or question_doc.get("final_answer_text")
        or ""
    )
    if not reference_solution and isinstance(question_doc.get("metadata"), dict):
        reference_solution = (
            question_doc["metadata"].get("reference_solution")
            or question_doc["metadata"].get("solution")
            or question_doc["metadata"].get("answer")
            or ""
        )

    objective_options: List[Dict[str, str]] = []
    enhanced_options = question_doc.get("enhanced_options")
    if isinstance(enhanced_options, list):
        for index, option in enumerate(enhanced_options):
            if not isinstance(option, dict):
                continue
            content = str(
                option.get("content")
                or option.get("text")
                or option.get("value")
                or ""
            ).strip()
            if not content:
                continue
            label = normalize_answer_label(
                option.get("label") or option.get("key") or option.get("id")
            ) or chr(ord("A") + index)
            objective_options.append({"label": label, "text": content})
    if not objective_options:
        options = question_doc.get("options")
        if isinstance(options, list):
            for index, option in enumerate(options):
                content = str(
                    (
                        option.get("content")
                        or option.get("text")
                        or option.get("value")
                        or ""
                    )
                    if isinstance(option, dict)
                    else option or ""
                ).strip()
                if content:
                    objective_options.append(
                        {"label": chr(ord("A") + index), "text": content}
                    )
    correct_answer = normalize_answer_label(
        question_doc.get("correct_answer") or question_doc.get("correctAnswer")
    )
    try:
        penalty_marks = float(
            question_doc.get(
                "penalty",
                question_doc.get("penalty_marks", 1),
            )
        )
    except (TypeError, ValueError):
        penalty_marks = 1.0

    raw_criteria = question_doc.get("marking_criteria")
    if raw_criteria is None and isinstance(question_doc.get("metadata"), dict):
        raw_criteria = question_doc["metadata"].get("marking_criteria")
    try:
        marking_criteria = normalize_marking_criteria(raw_criteria)
    except ValueError:
        # Draft/legacy content is validated at finalisation.  The adapter must
        # remain tolerant for existing read-only records.
        marking_criteria = []

    raw_method_policy = question_doc.get("method_policy")
    if raw_method_policy is None and isinstance(question_doc.get("metadata"), dict):
        raw_method_policy = question_doc["metadata"].get("method_policy")
    try:
        method_policy = normalize_method_policy(raw_method_policy)
    except ValueError:
        method_policy = normalize_method_policy(None)

    # Expected word range heuristic based on complexity
    expected_word_range: Optional[Dict[str, int]] = None
    if complexity == "L1":
        expected_word_range = {"min": 1, "max": 50}
    elif complexity == "L2":
        expected_word_range = {"min": 20, "max": 200}
    elif complexity == "L3":
        expected_word_range = {"min": 50, "max": 500}

    resolved_exam_id = (
        exam_id
        or question_doc.get("exam_id")
        or question_doc.get("document_id")
        or ""
    )

    # Preserve the authoring order used by student answer labels (Q1, Q2,
    # ...).  Session snapshots may override this with their immutable
    # position, but carrying it through the generic adapter keeps direct PCR
    # syncs and legacy papers addressable as well.
    question_number: Optional[int] = None
    for number_field in ("question_number", "extraction_order", "position"):
        raw_number = question_doc.get(number_field)
        try:
            parsed_number = int(raw_number)
        except (TypeError, ValueError):
            continue
        if parsed_number > 0:
            question_number = parsed_number
            break

    return {
        "question_id": question_id,
        "exam_id": resolved_exam_id,
        "question_number": question_number,
        "subject": subject,
        "question_type": question_type,
        "grading_mode": (
            "objective"
            if str(question_type or "").strip().lower()
            in {"mcq", "objective", "integer"}
            else "subjective"
        ),
        "options": copy.deepcopy(objective_options),
        "correct_answer": correct_answer or None,
        "penalty_marks": max(0.0, penalty_marks),
        "complexity": complexity,
        "eval_template": eval_template,
        "max_marks": max_marks,
        "question_text": question_text,
        "reference_solution": reference_solution or None,
        "rubric": rubric,
        "marking_criteria": copy.deepcopy(marking_criteria),
        "method_policy": copy.deepcopy(method_policy),
        "marking_policy": copy.deepcopy(question_doc.get("marking_policy"))
        if isinstance(question_doc.get("marking_policy"), dict)
        else None,
        "expects_diagram": bool(has_diagram),
        "diagram_weight": diagram_weight,
        "expected_word_range": expected_word_range,
    }


def adapt_paper_questions_to_pcr(
    paper_doc: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Translate all questions in a question-paper document to PCR format.

    Parameters
    ----------
    paper_doc : dict
        A ``question_papers`` document from the paper-builder.  Expected
        to have ``sections[].questions[]`` structure per
        ``paper_builder_async.py``.

    Returns
    -------
    list[dict]
        List of ``evalpen_questions``-shaped documents.
    """
    paper_id = paper_doc.get("id") or paper_doc.get("_id", "")
    subject = paper_doc.get("subject", "")
    results: List[Dict[str, Any]] = []

    for section in paper_doc.get("sections") or []:
        for question in section.get("questions") or []:
            pcr_doc = adapt_question_to_pcr(
                question,
                exam_id=str(paper_id),
                default_subject=subject,
            )
            results.append(pcr_doc)

    return results
