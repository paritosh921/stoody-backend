"""Composite plagiarism scorer -- combines all signal dimensions.

ZERO I/O -- pure computation only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .text_similarity import compute_tfidf_similarity
from .structural_similarity import compute_structural_similarity


# ---- enums & dataclasses -------------------------------------------------- #


class Severity(str, Enum):
    """Plagiarism flag severity levels."""

    REVIEW_RECOMMENDED = "review_recommended"
    STRONG_MATCH = "strong_match"


class QuestionType(str, Enum):
    """Broad question-type categories for scoring adjustment."""

    MCQ = "mcq"
    OBJECTIVE = "objective"  # fill-in, true/false, one-word
    SUBJECTIVE = "subjective"


@dataclass(frozen=True, slots=True)
class CompositeScore:
    """Immutable composite plagiarism score for a student pair."""

    text_sim: float
    structural_sim: float
    temporal_corr: float
    proximity_score: float
    composite: float
    severity: Severity | None


# ---- thresholds & weights ------------------------------------------------- #

REVIEW_THRESHOLD = 0.75
STRONG_THRESHOLD = 0.90

# Default weights for subjective questions (all four signals)
_W_TEXT = 0.35
_W_STRUCT = 0.30
_W_TEMPORAL = 0.20
_W_PROXIMITY = 0.15

# For MCQ/objective: identical correct answers are NOT plagiarism.
# Only temporal correlation and seating proximity matter.
_W_MCQ_TEMPORAL = 0.60
_W_MCQ_PROXIMITY = 0.40


# ---- helpers -------------------------------------------------------------- #


def _classify_severity(composite: float) -> Severity | None:
    """Map a composite score to a severity level (or None if below
    the review threshold)."""
    if composite >= STRONG_THRESHOLD:
        return Severity.STRONG_MATCH
    if composite >= REVIEW_THRESHOLD:
        return Severity.REVIEW_RECOMMENDED
    return None


def _is_objective_type(question_type: QuestionType) -> bool:
    return question_type in (QuestionType.MCQ, QuestionType.OBJECTIVE)


# ---- public API ----------------------------------------------------------- #


def score_pair(
    text_a: str,
    text_b: str,
    question_type: QuestionType,
    temporal_corr: float = 0.0,
    proximity_score: float = 0.0,
    correct_answer: str | None = None,
) -> CompositeScore:
    """Compute a composite plagiarism score for one student pair on one
    question.

    For MCQ/objective questions text similarity and structural similarity
    are excluded -- identical correct answers are expected, not plagiarism.

    Parameters
    ----------
    text_a, text_b:
        Recognized answer texts.
    question_type:
        Determines which signals contribute.
    temporal_corr:
        Pre-computed temporal correlation (0.0-1.0).
    proximity_score:
        Pre-computed seating proximity (0.0-1.0).
    correct_answer:
        Optional correct answer for same-error boost in structural
        similarity.

    Returns
    -------
    CompositeScore with all dimensions, composite, and severity.
    """
    temporal_corr = max(0.0, min(1.0, temporal_corr))
    proximity_score = max(0.0, min(1.0, proximity_score))

    if _is_objective_type(question_type):
        # MCQ/objective: skip text-based signals entirely
        text_sim = 0.0
        structural_sim = 0.0
        composite = (
            temporal_corr * _W_MCQ_TEMPORAL
            + proximity_score * _W_MCQ_PROXIMITY
        )
    else:
        text_sim = compute_tfidf_similarity(text_a, text_b)
        structural_sim = compute_structural_similarity(
            text_a, text_b, correct_answer,
        )
        composite = (
            text_sim * _W_TEXT
            + structural_sim * _W_STRUCT
            + temporal_corr * _W_TEMPORAL
            + proximity_score * _W_PROXIMITY
        )

    severity = _classify_severity(composite)

    return CompositeScore(
        text_sim=round(text_sim, 4),
        structural_sim=round(structural_sim, 4),
        temporal_corr=round(temporal_corr, 4),
        proximity_score=round(proximity_score, 4),
        composite=round(composite, 4),
        severity=severity,
    )
