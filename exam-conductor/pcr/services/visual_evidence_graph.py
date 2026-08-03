"""Multistage visual evidence contracts for handwritten PCR submissions.

The evidence graph deliberately separates two model responsibilities:

1. locate and associate every visible student-work region across the complete
   answer copy; and
2. grade question-specific high-resolution evidence against the immutable
   marking catalog.

OCR text is never the authority in this path.  Coordinates remain normalized
to the original student page so every criterion can be audited against pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence


# The external prompt version remains stable for already-finalized cohorts.
# This internal evidence version identifies the lean rubric-grading response
# introduced without invalidating an in-progress exam's frozen model contract.
EVIDENCE_GRAPH_VERSION = "pcr-multimodal-evidence-graph-v2"
PROMPT_VERSION = "pcr-full-document-visual-v5"

_CONTENT_TYPES = {"TEXT_ONLY", "MIXED", "DIAGRAM_HEAVY", "TABLE_PRESENT"}
_EVIDENCE_KINDS = {
    "handwriting",
    "mathematics",
    "diagram",
    "table",
    "graph",
    "label",
    "mixed",
}
_ATTEMPT_STATES = {"attempted", "not_attempted", "unresolved"}


@dataclass
class MappingValidationResult:
    document_review: Dict[str, Any]
    questions: Dict[int, Dict[str, Any]]
    unassigned_regions: List[Dict[str, Any]]
    errors: List[str] = field(default_factory=list)

    def as_payload(self) -> Dict[str, Any]:
        return {
            "evidence_graph_version": EVIDENCE_GRAPH_VERSION,
            "document_review": self.document_review,
            "questions": [
                self.questions[number] for number in sorted(self.questions)
            ],
            "unassigned_regions": list(self.unassigned_regions),
        }


def evidence_region_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "region_id": {"type": "string"},
            "page_number": {"type": "integer", "minimum": 1},
            "x_start": {"type": "number", "minimum": 0, "maximum": 1000},
            "y_start": {"type": "number", "minimum": 0, "maximum": 1000},
            "x_end": {"type": "number", "minimum": 0, "maximum": 1000},
            "y_end": {"type": "number", "minimum": 0, "maximum": 1000},
            "evidence_kind": {
                "type": "string",
                "enum": sorted(_EVIDENCE_KINDS),
            },
            "continuation_group": {"type": "string"},
            "evidence": {"type": "string"},
            "mapping_confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
        },
        "required": [
            "region_id",
            "page_number",
            "x_start",
            "y_start",
            "x_end",
            "y_end",
            "evidence_kind",
            "continuation_group",
            "evidence",
            "mapping_confidence",
        ],
    }


def evidence_mapping_schema() -> Dict[str, Any]:
    region = evidence_region_schema()
    question = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "question_number": {"type": "integer", "minimum": 1},
            "attempt_status": {
                "type": "string",
                "enum": sorted(_ATTEMPT_STATES),
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "content_type": {
                "type": "string",
                "enum": sorted(_CONTENT_TYPES),
            },
            "evidence_regions": {"type": "array", "items": region},
            "mapping_reason": {"type": "string"},
            "needs_review": {"type": "boolean"},
            "review_reason": {"type": "string"},
        },
        "required": [
            "question_number",
            "attempt_status",
            "confidence",
            "content_type",
            "evidence_regions",
            "mapping_reason",
            "needs_review",
            "review_reason",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "evidence_graph_version": {
                "type": "string",
                "enum": [EVIDENCE_GRAPH_VERSION],
            },
            "document_review": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "all_student_work_accounted": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "warnings": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "all_student_work_accounted",
                    "confidence",
                    "warnings",
                ],
            },
            "questions": {"type": "array", "items": question},
            "unassigned_regions": {"type": "array", "items": region},
        },
        "required": [
            "evidence_graph_version",
            "document_review",
            "questions",
            "unassigned_regions",
        ],
    }


def question_grading_schema() -> Dict[str, Any]:
    method_analysis = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "detected_method": {"type": "string"},
            "method_classification": {
                "type": "string",
                "enum": [
                    "reference_method",
                    "alternative_method",
                    "specified_method",
                    "no_method_visible",
                    "not_applicable",
                    "unresolved",
                ],
            },
            "method_validity": {
                "type": "string",
                "enum": [
                    "valid",
                    "partially_valid",
                    "invalid",
                    "not_applicable",
                    "unresolved",
                ],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "explanation": {"type": "string"},
            "error_carried_forward": {
                "type": "string",
                "enum": [
                    "applied",
                    "not_applied",
                    "not_applicable",
                    "unresolved",
                ],
            },
            "error_carried_forward_reason": {"type": "string"},
        },
        "required": [
            "detected_method",
            "method_classification",
            "method_validity",
            "confidence",
            "explanation",
            "error_carried_forward",
            "error_carried_forward_reason",
        ],
    }
    criterion = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "criterion_id": {"type": "string"},
            "decision": {
                "type": "string",
                            "enum": ["met", "partially_met", "not_met", "unresolved"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "marks_awarded": {"type": "number", "minimum": 0},
            "rationale": {"type": "string"},
            "evidence": {"type": "string"},
            "evidence_region_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "missing_evidence": {"type": "string"},
            "credit_basis": {
                "type": "string",
                "enum": [
                                "direct_evidence",
                                "error_carried_forward",
                                "no_credit",
                                "unresolved",
                ],
            },
        },
        "required": [
            "criterion_id",
            "decision",
            "confidence",
            "marks_awarded",
            "rationale",
            "evidence",
            "evidence_region_ids",
            "missing_evidence",
            "credit_basis",
        ],
    }
    question = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "question_number": {"type": "integer", "minimum": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "student_answer": {"type": "string"},
            "method_analysis": method_analysis,
            "criterion_marks": {"type": "array", "items": criterion},
            "total_score": {"type": "number", "minimum": 0},
            "overall_feedback": {"type": "string"},
            "needs_review": {"type": "boolean"},
            "review_reason": {"type": "string"},
        },
        "required": [
            "question_number",
            "confidence",
            "student_answer",
            "method_analysis",
            "criterion_marks",
            "total_score",
            "overall_feedback",
            "needs_review",
            "review_reason",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "evidence_graph_version": {
                "type": "string",
                "enum": [EVIDENCE_GRAPH_VERSION],
            },
            "questions": {"type": "array", "items": question},
        },
        "required": ["evidence_graph_version", "questions"],
    }


def mapping_system_instructions() -> str:
    return (
        "You are the visual evidence mapper for a high-stakes handwritten exam. "
        "This stage does not award marks. Inspect the original question paper, "
        "teacher solution, immutable question catalog, and every student page. "
        "Locate all student work using two-dimensional regions. A question may use "
        "many disconnected regions on distant pages; one page may contain several "
        "questions. Do not assume document order or trust a written question number "
        "without checking values, variables, terminology, requested result, method, "
        "diagram semantics, and continuation context. Preserve diagrams, graphs, "
        "tables, arrows, labels, superscripts, fractions, crossed-out work, and text "
        "around a visual. Printed question-paper or teacher-solution content is never "
        "student evidence. Use x/y coordinates from 0 to 1000 relative to the full "
        "original page. Use stable unique region_id values. Give continuation regions "
        "the same non-empty continuation_group. Set not_attempted only after inspecting "
        "every page. Put any visible work that cannot be assigned safely into "
        "unassigned_regions and set all_student_work_accounted=false. Objective "
        "answers may be written as a compact numbered list such as '1 B, 2 C'. Map "
        "each visible label to its catalog question and keep its region independently "
        "auditable. Names, roll numbers, class/section details, dates, signatures, "
        "page numbers, copy labels, invigilator marks, and other administrative header "
        "content are not answer work: ignore them and never put them in "
        "unassigned_regions. Make each answer region generous enough to include the "
        "complete line, final answer, nearby working, and immediately adjacent labels. "
        "When an answer continues elsewhere, return every continuation region rather "
        "than one representative crop. Never invent or grade evidence."
    )


def grading_system_instructions() -> str:
    return (
        "You are the question-level visual examiner for a high-stakes handwritten "
        "exam. The evidence mapper has already fixed question ownership. Grade only "
        "the supplied high-resolution evidence crops, their surrounding page context, "
        "and their stated region IDs against the supplied immutable question catalog, "
        "reference solution, and locked criteria. Full-page context is supplied to recover clipped symbols, "
        "final answers, and immediately adjacent continuation work belonging to the "
        "mapped answer; never borrow unrelated work from another question. "
        "Never use OCR text as the authority and never move evidence to another "
        "question. Read mathematical layout spatially: distinguish superscripts, "
        "fractions, roots, signs, matrices, and overwritten work. Evaluate diagrams "
        "visually. For a diagram criterion, describe the relevant label-arrow endpoint, "
        "structure, circuit connection, vector direction, graph axis, chemical bond, "
        "or other subject-specific relationship directly in that criterion's evidence "
        "and rationale. The reference solution "
        "is a correctness anchor, not the only acceptable method. Equivalent methods "
        "and equivalent representations receive the same criterion decisions. Award "
        "step marks for valid visible work and apply error-carried-forward only when "
        "the locked policy permits it. Cite one or more supplied evidence_region_ids "
        "for every criterion decision, including zero. Return exactly one result for "
        "each requested question and never exceed locked criterion maxima. Make the "
        "best evidence-supported provisional criterion decision even when handwriting "
        "is imperfect; express genuine uncertainty with confidence, needs_review, and "
        "review_reason instead of omitting the question or inventing extra metadata. "
        "Use the frozen paper context, subject, class or standard, board or course, response genre, and teacher guidance when applying the locked criteria. Open-ended answers must be judged by meaning and the stated writing criteria, not by exact phrase matching. Return one criterion result for every locked criterion in the exact order supplied; internal criterion ids are server-owned and must not be invented or reordered. "
        "Use student_answer as a faithful, concise Work shown transcription that "
        "preserves meaningful steps, values, equations, and the final answer in reading "
        "order; do not correct it there. Keep criterion evidence literal and each "
        "rationale to one short sentence. Write overall_feedback like a teacher's "
        "correction: one or two short, direct sentences saying what is correct and the "
        "specific correction needed. For a fully correct response, say 'Correct method "
        "and answer.' Do not mention AI, OCR, confidence, rubrics, evidence regions, "
        "uploaded images, or refer to 'the student' in the correction. "
        "For a "
        "catalog question with grading_mode=objective, only read the selected option "
        "label from the fixed evidence crop and put that single label in "
        "student_answer. Return empty criterion_marks, total_score 0, and "
        "not_applicable method analysis. The server alone compares the label with the "
        "answer key and applies positive or negative marks. If the label is ambiguous, "
        "set needs_review instead of guessing."
    )


def validate_mapping_payload(
    payload: Any,
    *,
    question_numbers: Sequence[int],
    page_count: int,
    absence_confidence: float,
) -> MappingValidationResult:
    errors: List[str] = []
    raw = payload if isinstance(payload, dict) else {}
    if raw.get("evidence_graph_version") != EVIDENCE_GRAPH_VERSION:
        errors.append("Evidence graph version does not match the required contract")
    review_raw = raw.get("document_review")
    if not isinstance(review_raw, dict):
        review_raw = {}
        errors.append("Document evidence review is missing")
    review = {
        "all_student_work_accounted": bool(
            review_raw.get("all_student_work_accounted")
        ),
        "confidence": _confidence(review_raw.get("confidence")),
        "warnings": [
            str(item)[:500]
            for item in (review_raw.get("warnings") or [])
            if str(item).strip()
        ],
    }

    expected = {int(number) for number in question_numbers}
    by_number: Dict[int, Dict[str, Any]] = {}
    raw_questions = raw.get("questions")
    if not isinstance(raw_questions, list):
        raw_questions = []
        errors.append("Question evidence mappings are missing")
    for item in raw_questions:
        if not isinstance(item, dict):
            errors.append("Question evidence mapping is not an object")
            continue
        number = _positive_int(item.get("question_number"))
        if number not in expected:
            errors.append("Evidence mapping refers to an unknown question")
            continue
        if number in by_number:
            errors.append(f"Q{number} has duplicate evidence mappings")
            by_number[number] = _unresolved_mapping(
                number,
                "Duplicate question evidence mappings require review",
            )
            continue
        by_number[number] = _validate_question_mapping(
            item,
            question_number=number,
            page_count=page_count,
            coverage_complete=review["all_student_work_accounted"],
            coverage_confidence=review["confidence"],
            absence_confidence=absence_confidence,
            errors=errors,
        )

    for number in sorted(expected - set(by_number)):
        reason = "No visual evidence-mapping result was returned for this question"
        errors.append(f"Q{number}: {reason}")
        by_number[number] = _unresolved_mapping(number, reason)

    unassigned: List[Dict[str, Any]] = []
    raw_unassigned = raw.get("unassigned_regions")
    if isinstance(raw_unassigned, list):
        for index, item in enumerate(raw_unassigned, start=1):
            region, region_errors = _validate_region(
                item,
                page_count=page_count,
                fallback_region_id=f"unassigned-{index}",
            )
            errors.extend(f"Unassigned region: {error}" for error in region_errors)
            if region:
                unassigned.append(region)
    if unassigned:
        review["all_student_work_accounted"] = False
        review["warnings"].append(
            f"{len(unassigned)} visible student-work region(s) remain unassigned"
        )
    if errors:
        review["all_student_work_accounted"] = False
        review["warnings"].append(
            "The visual evidence graph has structural validation errors"
        )
    return MappingValidationResult(
        document_review=review,
        questions=by_number,
        unassigned_regions=unassigned,
        errors=errors,
    )


def merge_mapping_and_grading(
    mapping: MappingValidationResult,
    grading_payloads: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    grades: Dict[int, Dict[str, Any]] = {}
    for payload in grading_payloads:
        raw_questions = payload.get("questions")
        if not isinstance(raw_questions, list):
            continue
        for item in raw_questions:
            if not isinstance(item, dict):
                continue
            number = _positive_int(item.get("question_number"))
            if number and number not in grades:
                grades[number] = dict(item)

    questions: List[Dict[str, Any]] = []
    for number in sorted(mapping.questions):
        mapped = mapping.questions[number]
        status = mapped["attempt_status"]
        if status == "not_attempted":
            questions.append(
                {
                    **mapped,
                    "student_answer": "",
                    "interpretation_hypotheses": [],
                    "visual_semantics": _empty_visual_semantics(),
                    "criterion_marks": [],
                    "total_score": 0,
                    "overall_feedback": "Question not attempted.",
                    "needs_review": False,
                    "review_reason": "",
                }
            )
            continue
        if status == "unresolved":
            questions.append(
                {
                    **mapped,
                    "student_answer": "",
                    "interpretation_hypotheses": [],
                    "visual_semantics": _empty_visual_semantics(),
                    "criterion_marks": [],
                    "total_score": 0,
                    "overall_feedback": "Question evidence requires review.",
                    "needs_review": True,
                    "review_reason": mapped.get("review_reason")
                    or "Question evidence ownership is unresolved",
                }
            )
            continue
        grade = grades.get(number)
        if not grade:
            questions.append(
                {
                    **mapped,
                    "attempt_status": "unresolved",
                    "student_answer": "",
                    "interpretation_hypotheses": [],
                    "visual_semantics": _empty_visual_semantics(),
                    "criterion_marks": [],
                    "total_score": 0,
                    "overall_feedback": "Question-specific visual grading did not complete.",
                    "needs_review": True,
                    "review_reason": (
                        "Question-specific high-resolution grading result is missing"
                    ),
                }
            )
            continue
        questions.append(
            {
                **mapped,
                "confidence": min(
                    _confidence(mapped.get("confidence")),
                    _confidence(grade.get("confidence")),
                ),
                "student_answer": str(grade.get("student_answer") or "").strip(),
                # Retain the legacy materialized fields for API compatibility.
                # They are no longer model-generated scoring requirements.
                "interpretation_hypotheses": [],
                "visual_semantics": _empty_visual_semantics(),
                "method_analysis": dict(grade.get("method_analysis") or {}),
                "criterion_marks": list(grade.get("criterion_marks") or []),
                "total_score": grade.get("total_score", 0),
                "overall_feedback": str(
                    grade.get("overall_feedback") or ""
                ).strip(),
                "needs_review": bool(
                    mapped.get("needs_review") or grade.get("needs_review")
                ),
                "review_reason": str(
                    mapped.get("review_reason")
                    or grade.get("review_reason")
                    or ""
                ).strip(),
            }
        )
    return {
        "evidence_graph_version": EVIDENCE_GRAPH_VERSION,
        "document_review": dict(mapping.document_review),
        "questions": questions,
        "unassigned_regions": list(mapping.unassigned_regions),
    }


def _validate_question_mapping(
    raw: Mapping[str, Any],
    *,
    question_number: int,
    page_count: int,
    coverage_complete: bool,
    coverage_confidence: float,
    absence_confidence: float,
    errors: List[str],
) -> Dict[str, Any]:
    status = str(raw.get("attempt_status") or "unresolved").strip().lower()
    if status not in _ATTEMPT_STATES:
        status = "unresolved"
    regions: List[Dict[str, Any]] = []
    seen: set[str] = set()
    raw_regions = raw.get("evidence_regions")
    if not isinstance(raw_regions, list):
        raw_regions = []
    for index, item in enumerate(raw_regions, start=1):
        region, region_errors = _validate_region(
            item,
            page_count=page_count,
            fallback_region_id=f"q{question_number}-region-{index}",
        )
        errors.extend(f"Q{question_number}: {error}" for error in region_errors)
        if not region:
            continue
        if region["region_id"] in seen:
            errors.append(
                f"Q{question_number}: duplicate evidence region ID "
                f"{region['region_id']}"
            )
            continue
        seen.add(region["region_id"])
        regions.append(region)

    reason = str(raw.get("review_reason") or "").strip()
    needs_review = bool(raw.get("needs_review"))
    confidence = _confidence(raw.get("confidence"))
    if status == "attempted" and not regions:
        status = "unresolved"
        needs_review = True
        reason = reason or "Attempted answer has no visible evidence region"
    if status == "not_attempted":
        if regions:
            status = "unresolved"
            needs_review = True
            reason = reason or "Not-attempted decision contradicts visible evidence"
        elif not coverage_complete or coverage_confidence < absence_confidence:
            status = "unresolved"
            needs_review = True
            reason = reason or "Full-copy mapping did not prove this question absent"
    if status == "unresolved":
        needs_review = True
        reason = reason or "Question evidence ownership is unresolved"
    content_type = str(raw.get("content_type") or "MIXED").strip().upper()
    if content_type not in _CONTENT_TYPES:
        content_type = "MIXED"
    return {
        "question_number": question_number,
        "attempt_status": status,
        "confidence": confidence,
        "content_type": content_type,
        "evidence_regions": regions,
        "mapping_reason": str(raw.get("mapping_reason") or "").strip(),
        "needs_review": needs_review,
        "review_reason": reason,
    }


def _validate_region(
    raw: Any,
    *,
    page_count: int,
    fallback_region_id: str,
) -> tuple[Dict[str, Any] | None, List[str]]:
    if not isinstance(raw, dict):
        return None, ["Evidence region is not an object"]
    errors: List[str] = []
    page_number = _positive_int(raw.get("page_number"))
    if not page_number or page_number > page_count:
        return None, ["Evidence refers to a non-submitted page"]
    coords = {
        key: _finite_float(raw.get(key))
        for key in ("x_start", "y_start", "x_end", "y_end")
    }
    if any(value is None for value in coords.values()):
        return None, ["Evidence region is missing two-dimensional coordinates"]
    x_start = float(coords["x_start"])
    y_start = float(coords["y_start"])
    x_end = float(coords["x_end"])
    y_end = float(coords["y_end"])
    if (
        x_start < 0
        or y_start < 0
        or x_end > 1000
        or y_end > 1000
        or x_end <= x_start
        or y_end <= y_start
    ):
        return None, ["Evidence has an invalid two-dimensional page region"]
    region_id = str(raw.get("region_id") or fallback_region_id).strip()
    if not region_id:
        region_id = fallback_region_id
    kind = str(raw.get("evidence_kind") or "mixed").strip().lower()
    if kind not in _EVIDENCE_KINDS:
        kind = "mixed"
        errors.append("Evidence kind was normalized to mixed")
    return (
        {
            "region_id": region_id[:120],
            "page_number": page_number,
            "x_start": round(x_start, 3),
            "y_start": round(y_start, 3),
            "x_end": round(x_end, 3),
            "y_end": round(y_end, 3),
            "evidence_kind": kind,
            "continuation_group": str(
                raw.get("continuation_group") or ""
            ).strip()[:120],
            "evidence": str(raw.get("evidence") or "").strip()[:1000],
            "mapping_confidence": _confidence(raw.get("mapping_confidence")),
        },
        errors,
    )


def _unresolved_mapping(question_number: int, reason: str) -> Dict[str, Any]:
    return {
        "question_number": question_number,
        "attempt_status": "unresolved",
        "confidence": 0.0,
        "content_type": "MIXED",
        "evidence_regions": [],
        "mapping_reason": "",
        "needs_review": True,
        "review_reason": reason,
    }


def _empty_visual_semantics() -> Dict[str, Any]:
    return {
        "summary": "",
        "elements": [],
        "relationships": [],
        "confidence": 0.0,
    }


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return parsed


def _confidence(value: Any) -> float:
    parsed = _finite_float(value)
    if parsed is None:
        return 0.0
    return max(0.0, min(1.0, parsed))
