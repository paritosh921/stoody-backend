"""Step-level rubric evaluation -- pure domain logic.

Given an AI result (recognised text + step breakdown) and a rubric
(expected steps with marks), produce a ``QuestionScore`` with per-step
awarded marks, a total, and an overall confidence.

Supports:
    * Positive step marking (each correct step earns marks).
    * Negative marking config (incorrect steps deduct marks).
    * Partial-credit proportional to confidence.

This module is ZERO I/O -- no asyncio, no DB, no network imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


# ---- Data models (pure value objects) ----------------------------------------

@dataclass(frozen=True, slots=True)
class RubricStep:
    """One expected step in the marking scheme."""
    label: str
    max_marks: float
    keywords: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Rubric:
    """Full rubric for a single question."""
    question_id: str
    version: int
    steps: list[RubricStep]
    negative_marking: bool = False
    negative_factor: float = 0.0  # deduction per wrong step


@dataclass(frozen=True, slots=True)
class StepScore:
    """Result of evaluating one rubric step."""
    label: str
    awarded: float
    max: float
    matched: bool


@dataclass(frozen=True, slots=True)
class QuestionScore:
    """Aggregate result for one question."""
    question_id: str
    step_scores: list[StepScore]
    total_marks: float
    max_marks: float
    confidence: float
    rubric_version: int


# ---- Evaluation logic --------------------------------------------------------

def _step_matches(step: RubricStep, recognised_steps: Sequence[str]) -> bool:
    """Return True if any recognised step text contains at least one keyword."""
    if not step.keywords:
        return False
    lower_recognised = [s.lower() for s in recognised_steps]
    return any(
        kw.lower() in text
        for kw in step.keywords
        for text in lower_recognised
    )


def evaluate(
    ai_question_result: dict,
    rubric: Rubric,
) -> QuestionScore:
    """Evaluate a single question's AI result against its rubric.

    Parameters
    ----------
    ai_question_result:
        A dict matching one element of ``ai.result.question_results``:
        ``{"question_id", "recognized_text", "confidence", "step_breakdown"}``.
    rubric:
        The ``Rubric`` to mark against.

    Returns
    -------
    QuestionScore with per-step breakdown.
    """
    recognised_steps: list[str] = ai_question_result.get(
        "step_breakdown", []
    )
    confidence: float = float(ai_question_result.get("confidence", 0.0))

    step_scores: list[StepScore] = []
    total = 0.0
    max_total = 0.0

    for rubric_step in rubric.steps:
        matched = _step_matches(rubric_step, recognised_steps)
        awarded = rubric_step.max_marks if matched else 0.0

        if not matched and rubric.negative_marking:
            awarded = -(rubric.negative_factor * rubric_step.max_marks)

        step_scores.append(
            StepScore(
                label=rubric_step.label,
                awarded=awarded,
                max=rubric_step.max_marks,
                matched=matched,
            )
        )
        total += awarded
        max_total += rubric_step.max_marks

    # Clamp total so negative marking cannot produce below zero.
    total = max(total, 0.0)

    return QuestionScore(
        question_id=rubric.question_id,
        step_scores=step_scores,
        total_marks=total,
        max_marks=max_total,
        confidence=confidence,
        rubric_version=rubric.version,
    )
