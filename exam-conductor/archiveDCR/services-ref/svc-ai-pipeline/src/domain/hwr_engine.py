"""Handwriting recognition engine — pure domain logic, ZERO I/O.

Delegates actual inference to a model adapter passed in as a callable.
Handles confidence scoring, per-character thresholding, and flagging.
"""

from dataclasses import dataclass, field

# Default per-character confidence threshold (from FAILURE_MITIGATION_REGISTER A4.6).
DEFAULT_CONFIDENCE_THRESHOLD = 0.85

# If more than this fraction of characters are below threshold,
# the entire answer is flagged for teacher review.
ANSWER_FLAG_RATIO = 0.30


@dataclass(frozen=True)
class CharConfidence:
    """Confidence score for a single recognized character."""

    char: str
    confidence: float
    flagged: bool


@dataclass(frozen=True)
class HWRResult:
    """Result of handwriting recognition on a single image region."""

    recognized_text: str
    confidence: float
    per_character_confidence: list[CharConfidence] = field(default_factory=list)
    language: str = "en"
    flagged_for_review: bool = False
    flagged_characters: list[int] = field(default_factory=list)


def _compute_aggregate_confidence(
    per_char: list[CharConfidence],
) -> float:
    """Return mean confidence across characters, or 0.0 if empty."""
    if not per_char:
        return 0.0
    return sum(c.confidence for c in per_char) / len(per_char)


def _flag_characters(
    per_char: list[CharConfidence],
    threshold: float,
) -> list[int]:
    """Return indices of characters whose confidence is below *threshold*."""
    return [i for i, c in enumerate(per_char) if c.confidence < threshold]


def _should_flag_answer(
    per_char: list[CharConfidence],
    threshold: float,
) -> bool:
    """Flag the entire answer if >ANSWER_FLAG_RATIO characters are below threshold."""
    if not per_char:
        return False
    below = sum(1 for c in per_char if c.confidence < threshold)
    return (below / len(per_char)) > ANSWER_FLAG_RATIO


def build_per_char_confidence(
    chars: list[str],
    confidences: list[float],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[CharConfidence]:
    """Build CharConfidence list from parallel char and confidence arrays."""
    if len(chars) != len(confidences):
        raise ValueError(
            f"chars length ({len(chars)}) != confidences length ({len(confidences)})"
        )
    return [
        CharConfidence(char=ch, confidence=conf, flagged=conf < threshold)
        for ch, conf in zip(chars, confidences)
    ]


def recognize_text(
    image_data: bytes,
    language: str,
    run_inference_fn: callable,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> HWRResult:
    """Run HWR on *image_data* via the provided inference callable.

    *run_inference_fn* must accept (image_data: bytes) and return a dict:
        {"chars": list[str], "confidences": list[float]}

    This function performs ZERO I/O itself — all I/O is in the callable.
    """
    raw = run_inference_fn(image_data)
    chars: list[str] = raw.get("chars", [])
    confidences: list[float] = raw.get("confidences", [])

    per_char = build_per_char_confidence(chars, confidences, threshold)
    flagged_indices = _flag_characters(per_char, threshold)
    aggregate = _compute_aggregate_confidence(per_char)
    answer_flagged = _should_flag_answer(per_char, threshold)

    recognized_text = "".join(chars)

    return HWRResult(
        recognized_text=recognized_text,
        confidence=aggregate,
        per_character_confidence=per_char,
        language=language,
        flagged_for_review=answer_flagged,
        flagged_characters=flagged_indices,
    )
