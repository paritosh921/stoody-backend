"""Compact image-to-answer contract for objective PCR submissions.

The multimodal model only transcribes the student's marks.  It receives the
conducted question numbers and permitted answer labels, never the answer key or
marks.  Deterministic code validates the compact response and the existing
objective scorer applies the frozen key and marking policy.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from services.answer_mapping_contract import normalize_answer_label
from services.objective_scoring_service import is_integer_question

OBJECTIVE_PAPER_CONTEXT_VERSION = "objective-answer-ledger-v3"
OBJECTIVE_PROMPT_VERSION = "pcr-objective-answer-ledger-v3"
OBJECTIVE_LEDGER_VERSION = "pcr-objective-page-observations-v3"

LEGACY_OBJECTIVE_PAPER_CONTEXT_VERSIONS = frozenset(
    {"objective-answer-ledger-v1", "objective-answer-ledger-v2"}
)
LEGACY_OBJECTIVE_PROMPT_VERSIONS = frozenset(
    {"pcr-objective-answer-ledger-v1", "pcr-objective-answer-ledger-v2"}
)
LEGACY_OBJECTIVE_LEDGER_VERSIONS = frozenset(
    {
        "pcr-objective-page-observations-v1",
        "pcr-objective-page-observations-v2",
    }
)

_ANSWER_STATES = {
    "selected",
    "blank",
    "multiple_selected",
    "ambiguous",
    "unreadable",
}
_SHEET_FORMATS = {
    "omr_grid",
    "numbered_answer_list",
    "mixed",
    "unrecognized",
}
_MIN_AUTOMATIC_READING_CONFIDENCE = 0.55


def is_objective_question(question: Mapping[str, Any]) -> bool:
    return str(
        question.get("grading_mode")
        or question.get("question_type")
        or ""
    ).strip().lower() in {"objective", "mcq", "integer"}


def all_questions_are_objective(
    questions: Iterable[Mapping[str, Any]],
) -> bool:
    items = list(questions)
    return bool(items) and all(is_objective_question(item) for item in items)


def objective_option_labels(question: Mapping[str, Any]) -> List[str]:
    if is_integer_question(dict(question)):
        return []
    labels: List[str] = []
    options = question.get("options")
    if not isinstance(options, list) or not options:
        options = question.get("enhanced_options")
    if not isinstance(options, list):
        return labels
    for index, option in enumerate(options):
        if isinstance(option, Mapping):
            label = normalize_answer_label(
                option.get("label") or option.get("key") or option.get("id")
            )
        else:
            label = ""
        label = label or chr(ord("A") + index)
        if label not in labels:
            labels.append(label)
    return labels


def objective_extraction_catalog(
    questions: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Return only the values needed to transcribe a submitted answer sheet."""

    catalog: List[Dict[str, Any]] = []
    for position, question in enumerate(questions, start=1):
        number = _positive_int(question.get("question_number")) or position
        catalog.append(
            {
                "question_number": number,
                "answer_format": (
                    "integer_or_numeric_text"
                    if is_integer_question(dict(question))
                    else "option_label"
                ),
                "allowed_option_labels": objective_option_labels(question),
            }
        )
    return catalog


def objective_page_observation_schema(
    question_numbers: Sequence[int] = (),
) -> Dict[str, Any]:
    """Return a strict, compact schema limited to the conducted examination."""

    allowed_numbers = sorted(
        {
            number
            for raw in question_numbers
            if (number := _positive_int(raw)) is not None
        }
    )
    question_number_schema: Dict[str, Any] = {"type": "integer", "minimum": 1}
    if allowed_numbers:
        question_number_schema["enum"] = allowed_numbers
    observation = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "question_number": question_number_schema,
            "state": {"type": "string", "enum": sorted(_ANSWER_STATES)},
            "selected_answer": {"type": "string"},
            "alternative_answers": {
                "type": "array",
                "items": {"type": "string"},
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": [
            "question_number",
            "state",
            "selected_answer",
            "alternative_answers",
            "confidence",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ledger_version": {
                "type": "string",
                "enum": [OBJECTIVE_LEDGER_VERSION],
            },
            "page_number": {"type": "integer", "minimum": 1},
            "sheet_format": {
                "type": "string",
                "enum": sorted(_SHEET_FORMATS),
            },
            "page_fully_reviewed": {"type": "boolean"},
            "observations": {"type": "array", "items": observation},
        },
        "required": [
            "ledger_version",
            "page_number",
            "sheet_format",
            "page_fully_reviewed",
            "observations",
        ],
    }


def objective_reader_instructions() -> str:
    return (
        "Read one original-resolution photographed objective answer page and "
        "transcribe only the student's marks. Do not solve questions, grade, infer "
        "marks, or compare with a correct answer; no answer key is provided. The "
        "catalog lists the only conducted question numbers and permitted labels. "
        "The page may be an OMR grid, a handwritten list such as '11 D', or a mix. "
        "For an OMR grid, report every visible catalog question row, including blank "
        "rows. Ignore pre-printed row numbers that are outside the catalog. For a "
        "handwritten numbered list, report every visible written question-answer "
        "entry and do not invent entries that were not written. For a mixed page, "
        "report the union of both. Use state=selected when exactly one option is "
        "clearly marked. First distinguish a clean filled bubble from an option that "
        "has been crossed out, struck through, or heavily scribbled as a correction. "
        "When exactly one clean filled bubble remains and other options are crossed "
        "out, the clean filled bubble is the final selected answer; crossed-out "
        "options are cancelled alternatives and must not cause multiple_selected. "
        "When there is no clean filled bubble, a single tick, cross, circle, or "
        "handwritten label may itself be the selected answer. Use multiple_selected "
        "only when two or more non-cancelled options remain selected, such as two "
        "clean filled bubbles. Use ambiguous only when the final intended option "
        "cannot be determined after applying this correction rule. Use unreadable "
        "only when the relevant pixels cannot be read. "
        "For selected, selected_answer must be one permitted label such as C, or the "
        "exact numeric text for an integer question. For every other state leave "
        "selected_answer empty. alternative_answers is only for plausible or multiple "
        "options. Set page_fully_reviewed=true only after inspecting the entire page. "
        "Return the requested JSON and nothing else."
    )


def merge_objective_page_ledgers(
    page_payloads: Sequence[Mapping[str, Any]],
    *,
    questions: Sequence[Mapping[str, Any]],
    page_count: int,
) -> Tuple[Dict[str, Any], List[str]]:
    """Validate page readings and build the existing materializer ledger shape."""

    question_by_number = {
        _positive_int(question.get("question_number")) or position: dict(question)
        for position, question in enumerate(questions, start=1)
    }
    records_by_number: Dict[int, List[Dict[str, Any]]] = {
        number: [] for number in question_by_number
    }
    errors: List[str] = []
    warnings: List[str] = []
    seen_pages: set[int] = set()
    coverage_complete = True

    for raw_page in page_payloads:
        if not isinstance(raw_page, Mapping):
            errors.append("Objective reader returned a non-object page ledger")
            coverage_complete = False
            continue
        page_number = _positive_int(raw_page.get("page_number"))
        if not page_number or page_number > page_count:
            errors.append("Objective reader returned an invalid page number")
            coverage_complete = False
            continue
        if page_number in seen_pages:
            errors.append(f"Objective reader returned page {page_number} more than once")
            coverage_complete = False
            continue
        seen_pages.add(page_number)

        if str(raw_page.get("ledger_version") or "") != OBJECTIVE_LEDGER_VERSION:
            errors.append(f"Page {page_number} used the wrong objective ledger version")
            coverage_complete = False
        sheet_format = str(raw_page.get("sheet_format") or "").strip().lower()
        if sheet_format not in _SHEET_FORMATS:
            errors.append(f"Page {page_number} has an invalid sheet format")
            coverage_complete = False
        elif sheet_format == "unrecognized":
            warnings.append(
                f"Page {page_number} answer-sheet format could not be recognized"
            )
            coverage_complete = False
        if not bool(raw_page.get("page_fully_reviewed")):
            warnings.append(f"Page {page_number} was not fully reviewed")
            coverage_complete = False

        observations = raw_page.get("observations")
        if not isinstance(observations, list):
            errors.append(f"Page {page_number} observations are not an array")
            coverage_complete = False
            continue
        if sheet_format == "omr_grid" and not observations:
            errors.append(
                f"Page {page_number} was identified as OMR but has no catalog rows"
            )
            coverage_complete = False

        page_numbers: set[int] = set()
        for index, raw_observation in enumerate(observations, start=1):
            observation, observation_errors = _validate_observation(
                raw_observation,
                page_number=page_number,
                index=index,
                question_by_number=question_by_number,
            )
            if observation_errors:
                errors.extend(observation_errors)
                coverage_complete = False
            if observation is None:
                continue
            number = observation["question_number"]
            if number in page_numbers:
                errors.append(
                    f"Page {page_number} reports question {number} more than once"
                )
                coverage_complete = False
                continue
            page_numbers.add(number)
            records_by_number[number].append(observation)

    missing_pages = sorted(set(range(1, page_count + 1)) - seen_pages)
    if missing_pages:
        errors.append(
            "Objective reader omitted submitted page(s): "
            + ", ".join(str(page) for page in missing_pages)
        )
        coverage_complete = False

    coverage_confidence = 1.0 if coverage_complete else 0.0
    questions_payload = [
        _merge_question_records(
            number,
            question,
            records_by_number.get(number) or [],
            coverage_complete=coverage_complete,
            coverage_confidence=coverage_confidence,
        )
        for number, question in question_by_number.items()
    ]
    payload: Dict[str, Any] = {
        "evidence_graph_version": OBJECTIVE_LEDGER_VERSION,
        "document_review": {
            "all_student_work_accounted": coverage_complete,
            "confidence": coverage_confidence,
            "warnings": list(dict.fromkeys(warnings)),
        },
        "questions": questions_payload,
    }
    if errors:
        payload["evidence_graph_validation_errors"] = list(dict.fromkeys(errors))
    return payload, list(dict.fromkeys(errors))


def _validate_observation(
    raw: Any,
    *,
    page_number: int,
    index: int,
    question_by_number: Mapping[int, Mapping[str, Any]],
) -> Tuple[Dict[str, Any] | None, List[str]]:
    prefix = f"Page {page_number} observation {index}"
    if not isinstance(raw, Mapping):
        return None, [f"{prefix} is not an object"]
    number = _positive_int(raw.get("question_number"))
    if not number or number not in question_by_number:
        return None, [f"{prefix} refers to a non-catalog question"]
    state = str(raw.get("state") or "").strip().lower()
    errors: List[str] = []
    if state not in _ANSWER_STATES:
        errors.append(f"{prefix} has an invalid answer state")
        state = "unreadable"
    confidence = _confidence(raw.get("confidence"))
    selected_answer = str(raw.get("selected_answer") or "").strip()
    raw_alternatives = raw.get("alternative_answers")
    if not isinstance(raw_alternatives, list):
        errors.append(f"{prefix} alternatives are not an array")
        raw_alternatives = []
    question = question_by_number[number]
    if state == "selected":
        selected_answer = _normalize_selected_answer(question, selected_answer)
        if not selected_answer:
            errors.append(f"{prefix} has no permitted selected answer")
            state = "ambiguous"
    elif selected_answer:
        errors.append(f"{prefix} supplied a selected answer for state {state}")
        selected_answer = ""
    alternatives = list(
        dict.fromkeys(
            value
            for value in (
                _normalize_selected_answer(question, item)
                for item in raw_alternatives
            )
            if value
        )
    )
    if state == "multiple_selected" and len(alternatives) < 2:
        errors.append(f"{prefix} did not identify multiple permitted answers")
    return (
        {
            "question_number": number,
            "page_number": page_number,
            "state": state,
            "selected_answer": selected_answer,
            "alternative_answers": alternatives,
            "confidence": confidence,
            "index": index,
        },
        errors,
    )


def _merge_question_records(
    number: int,
    question: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    coverage_complete: bool,
    coverage_confidence: float,
) -> Dict[str, Any]:
    base = {
        "question_number": number,
        "content_type": "TEXT_ONLY",
        "criterion_marks": [],
        "method_analysis": {},
        "overall_feedback": "",
        "needs_review": False,
        "review_reason": "",
        "interpretation_hypotheses": [],
        "visual_semantics": {},
        "evidence_regions": [],
        "source_page_numbers": sorted(
            {
                int(record.get("page_number") or 0)
                for record in records
                if int(record.get("page_number") or 0) > 0
            }
        ),
        "total_score": 0.0,
    }
    if not records:
        if coverage_complete:
            return {
                **base,
                "attempt_status": "not_attempted",
                "confidence": coverage_confidence,
                "student_answer": "",
            }
        return {
            **base,
            "attempt_status": "unresolved",
            "confidence": 0.0,
            "student_answer": "",
            "needs_review": True,
            "review_reason": "The submitted pages were not completely read",
        }

    confidence = min(_confidence(item.get("confidence")) for item in records)
    selected = [
        item for item in records if str(item.get("state") or "") == "selected"
    ]
    blanks = [
        item for item in records if str(item.get("state") or "") == "blank"
    ]
    selected_answers = {
        str(item.get("selected_answer") or "") for item in selected
    }

    if (
        len(selected) == len(records)
        and len(selected_answers) == 1
        and confidence >= _MIN_AUTOMATIC_READING_CONFIDENCE
    ):
        return {
            **base,
            "attempt_status": "attempted",
            "confidence": confidence,
            "student_answer": next(iter(selected_answers)),
        }
    if (
        len(blanks) == len(records)
        and coverage_complete
        and confidence >= _MIN_AUTOMATIC_READING_CONFIDENCE
    ):
        return {
            **base,
            "attempt_status": "not_attempted",
            "confidence": confidence,
            "student_answer": "",
        }

    return {
        **base,
        "attempt_status": "unresolved",
        "confidence": confidence,
        "student_answer": (
            next(iter(selected_answers))
            if len(selected_answers) == 1
            else ""
        ),
        "needs_review": True,
        "review_reason": _unresolved_reason(records, question, confidence),
    }


def _unresolved_reason(
    records: Sequence[Mapping[str, Any]],
    question: Mapping[str, Any],
    confidence: float,
) -> str:
    states = {str(record.get("state") or "") for record in records}
    if confidence < _MIN_AUTOMATIC_READING_CONFIDENCE:
        return "The selected option could not be read with sufficient confidence"
    if len(records) > 1:
        answers = {
            str(record.get("selected_answer") or "")
            for record in records
            if str(record.get("selected_answer") or "")
        }
        if len(answers) > 1:
            return "Conflicting answers were found for this question"
        return "Conflicting answer states were found for this question"
    state = next(iter(states), "unreadable")
    if state == "multiple_selected":
        return "More than one option is selected"
    if state == "blank":
        return "The blank answer row could not be verified"
    if state == "ambiguous":
        return "The student's final intended option is visually ambiguous"
    if state == "unreadable":
        return "The answer area is unreadable"
    return (
        "The selected answer is not one of the permitted options"
        if objective_option_labels(question)
        else "The objective answer could not be verified"
    )


def _normalize_selected_answer(
    question: Mapping[str, Any],
    raw_answer: Any,
) -> str:
    value = str(raw_answer or "").strip()
    if not value:
        return ""
    if is_integer_question(dict(question)):
        return value[:100]
    label = normalize_answer_label(value)
    return label if label in objective_option_labels(question) else ""


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _confidence(value: Any) -> float:
    parsed = _finite_float(value)
    if parsed is None:
        return 0.0
    return max(0.0, min(1.0, parsed))
