"""Content type classification — pure domain logic, ZERO I/O.

Classifies page content as text, formula, diagram, or mixed
based on extracted features from HWR and image analysis.
"""

from dataclasses import dataclass
from enum import Enum


class ContentType(str, Enum):
    """Recognized content types on a page region."""

    TEXT = "text"
    FORMULA = "formula"
    DIAGRAM = "diagram"
    MIXED = "mixed"
    EMPTY = "empty"


@dataclass(frozen=True)
class ClassificationResult:
    """Output of content classification."""

    content_type: ContentType
    type_confidence: float
    features_used: list[str]


# Feature keys expected in the features dict.
_KEY_HAS_MATH_SYMBOLS = "has_math_symbols"
_KEY_HAS_DRAWN_SHAPES = "has_drawn_shapes"
_KEY_TEXT_LINE_COUNT = "text_line_count"
_KEY_SYMBOL_RATIO = "symbol_ratio"
_KEY_STROKE_DENSITY = "stroke_density"
_KEY_ASPECT_RATIO = "aspect_ratio"

# Thresholds for classification heuristics.
_MATH_SYMBOL_RATIO_THRESHOLD = 0.15
_DIAGRAM_STROKE_DENSITY_THRESHOLD = 0.6
_TEXT_LINE_MIN = 2


def classify_content(features: dict) -> ClassificationResult:
    """Classify content type from extracted features.

    Parameters
    ----------
    features:
        Dictionary with keys like ``has_math_symbols``, ``has_drawn_shapes``,
        ``text_line_count``, ``symbol_ratio``, ``stroke_density``.

    Returns
    -------
    ClassificationResult with the predicted content type.
    """
    has_math = features.get(_KEY_HAS_MATH_SYMBOLS, False)
    has_shapes = features.get(_KEY_HAS_DRAWN_SHAPES, False)
    text_lines = features.get(_KEY_TEXT_LINE_COUNT, 0)
    symbol_ratio = features.get(_KEY_SYMBOL_RATIO, 0.0)
    stroke_density = features.get(_KEY_STROKE_DENSITY, 0.0)

    used: list[str] = []
    scores: dict[ContentType, float] = {
        ContentType.TEXT: 0.0,
        ContentType.FORMULA: 0.0,
        ContentType.DIAGRAM: 0.0,
    }

    # No content at all
    if text_lines == 0 and not has_math and not has_shapes and stroke_density < 0.01:
        return ClassificationResult(
            content_type=ContentType.EMPTY,
            type_confidence=0.95,
            features_used=["text_line_count", "stroke_density"],
        )

    # Text signals
    if text_lines >= _TEXT_LINE_MIN:
        scores[ContentType.TEXT] += 0.4
        used.append(_KEY_TEXT_LINE_COUNT)

    # Math signals
    if has_math:
        scores[ContentType.FORMULA] += 0.3
        used.append(_KEY_HAS_MATH_SYMBOLS)
    if symbol_ratio >= _MATH_SYMBOL_RATIO_THRESHOLD:
        scores[ContentType.FORMULA] += 0.3
        used.append(_KEY_SYMBOL_RATIO)

    # Diagram signals
    if has_shapes:
        scores[ContentType.DIAGRAM] += 0.4
        used.append(_KEY_HAS_DRAWN_SHAPES)
    if stroke_density >= _DIAGRAM_STROKE_DENSITY_THRESHOLD:
        scores[ContentType.DIAGRAM] += 0.3
        used.append(_KEY_STROKE_DENSITY)

    # Determine winner
    top_type = max(scores, key=scores.get)  # type: ignore[arg-type]
    top_score = scores[top_type]

    # If top two types are close, classify as mixed
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) >= 2 and sorted_scores[0] - sorted_scores[1] < 0.25:
        return ClassificationResult(
            content_type=ContentType.MIXED,
            type_confidence=round(top_score, 3),
            features_used=used,
        )

    confidence = min(round(top_score + 0.3, 3), 1.0)
    return ClassificationResult(
        content_type=top_type,
        type_confidence=confidence,
        features_used=used,
    )
