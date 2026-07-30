"""Deterministic scoring for objective questions.

Online Test Series and camera-based PCR use different answer-capture
mechanisms, but an objective response must have one scoring contract.  This
module owns that contract so an AI model may transcribe a photographed option
without ever deciding the marks.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from services.answer_mapping_contract import normalize_answer_label


class ObjectiveScoringContractError(ValueError):
    """Raised when a frozen objective question cannot be scored safely."""


def _finite_number(value: Any, *, default: float) -> float:
    if value in (None, "") or isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def objective_points(question: Dict[str, Any]) -> float:
    for key in ("points", "max_marks", "marks", "total_points"):
        parsed = _finite_number(question.get(key), default=0.0)
        if parsed > 0:
            return round(parsed, 2)
    return 4.0


def objective_penalty(question: Dict[str, Any]) -> float:
    value = question.get("penalty")
    if value in (None, ""):
        value = question.get("penalty_marks")
    return round(max(0.0, _finite_number(value, default=1.0)), 2)


def _is_numeric_answer(value: Any) -> bool:
    try:
        float(str(value or "").strip().replace("+", ""))
        return True
    except (TypeError, ValueError):
        return False


def is_integer_question(question: Dict[str, Any]) -> bool:
    question_type = str(question.get("question_type") or "mcq").strip().lower()
    correct_answer = str(
        question.get("correct_answer") or question.get("correctAnswer") or ""
    ).strip()
    options = question.get("options") or question.get("enhanced_options") or []
    return question_type == "integer" or (
        not options
        and _is_numeric_answer(correct_answer)
        and not normalize_answer_label(correct_answer)
    )


def _option_labels(question: Dict[str, Any]) -> list[str]:
    options = question.get("options")
    if not isinstance(options, list) or not options:
        options = question.get("enhanced_options")
    if not isinstance(options, list):
        return []
    labels: list[str] = []
    for index, option in enumerate(options):
        raw_label = (
            option.get("label") or option.get("key") or option.get("id")
            if isinstance(option, dict)
            else None
        )
        label = normalize_answer_label(raw_label) or chr(ord("A") + index)
        if label not in labels:
            labels.append(label)
    return labels


def score_objective_response(
    question: Dict[str, Any],
    student_answer: Any,
) -> Dict[str, Any]:
    """Score one response using the immutable key and marking values.

    Blank/SKIPPED answers receive zero. MCQ labels are canonicalized from
    common camera transcriptions such as ``"(C)"`` or ``"Option C"``. A
    non-empty response that cannot be reduced to an option label is rejected
    rather than guessed.
    """

    raw_student_answer = str(student_answer or "").strip()
    raw_correct_answer = str(
        question.get("correct_answer") or question.get("correctAnswer") or ""
    ).strip()
    points = objective_points(question)
    penalty = objective_penalty(question)
    attempted = bool(raw_student_answer) and raw_student_answer.upper() != "SKIPPED"

    if not raw_correct_answer:
        raise ObjectiveScoringContractError("Objective question has no correct answer")

    if not attempted:
        return {
            "attempted": False,
            "is_correct": False,
            "selected_answer": "",
            "correct_answer": (
                raw_correct_answer
                if is_integer_question(question)
                else normalize_answer_label(raw_correct_answer)
            ),
            "points": points,
            "penalty_marks": penalty,
            "points_earned": 0.0,
        }

    if is_integer_question(question):
        try:
            selected_number = float(raw_student_answer.replace("+", ""))
            correct_number = float(raw_correct_answer.replace("+", ""))
            is_correct = abs(selected_number - correct_number) < 1e-9
        except (TypeError, ValueError):
            is_correct = raw_student_answer.casefold() == raw_correct_answer.casefold()
        selected = raw_student_answer
        correct = raw_correct_answer
    else:
        selected = normalize_answer_label(raw_student_answer)
        correct = normalize_answer_label(raw_correct_answer)
        if not correct:
            raise ObjectiveScoringContractError(
                "Objective question has an invalid correct-answer label"
            )
        if not selected:
            raise ObjectiveScoringContractError(
                "Student response is not a recognizable objective option"
            )
        allowed_labels = _option_labels(question)
        if allowed_labels and selected not in allowed_labels:
            raise ObjectiveScoringContractError(
                "Student response is not one of the frozen objective options"
            )
        is_correct = selected == correct

    return {
        "attempted": True,
        "is_correct": is_correct,
        "selected_answer": selected,
        "correct_answer": correct,
        "points": points,
        "penalty_marks": penalty,
        "points_earned": points if is_correct else -penalty,
    }
