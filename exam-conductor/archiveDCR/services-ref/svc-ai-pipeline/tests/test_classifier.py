"""Unit tests for content classifier — content type classification.

Test IDs: U-AI-CLS-01 through U-AI-CLS-07
Validation level: L3 (unit, domain, no I/O)
"""

import pytest

from src.domain.classifier import (
    ClassificationResult,
    ContentType,
    classify_content,
)


# ---------------------------------------------------------------------------
# U-AI-CLS-01: Pure text features classify as TEXT
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pure_text():
    """U-AI-CLS-01: high text_line_count with no math/diagram -> TEXT."""
    features = {
        "has_math_symbols": False,
        "has_drawn_shapes": False,
        "text_line_count": 5,
        "symbol_ratio": 0.02,
        "stroke_density": 0.1,
    }
    result = classify_content(features)
    assert isinstance(result, ClassificationResult)
    assert result.content_type == ContentType.TEXT


# ---------------------------------------------------------------------------
# U-AI-CLS-02: Math symbols classify as FORMULA
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_formula_detection():
    """U-AI-CLS-02: math symbols + high symbol ratio -> FORMULA."""
    features = {
        "has_math_symbols": True,
        "has_drawn_shapes": False,
        "text_line_count": 1,
        "symbol_ratio": 0.40,
        "stroke_density": 0.2,
    }
    result = classify_content(features)
    assert result.content_type == ContentType.FORMULA


# ---------------------------------------------------------------------------
# U-AI-CLS-03: Drawn shapes classify as DIAGRAM
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_diagram_detection():
    """U-AI-CLS-03: drawn shapes + high stroke density -> DIAGRAM."""
    features = {
        "has_math_symbols": False,
        "has_drawn_shapes": True,
        "text_line_count": 0,
        "symbol_ratio": 0.0,
        "stroke_density": 0.8,
    }
    result = classify_content(features)
    assert result.content_type == ContentType.DIAGRAM


# ---------------------------------------------------------------------------
# U-AI-CLS-04: Mixed signals classify as MIXED
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_mixed_content():
    """U-AI-CLS-04: competing text and formula signals -> MIXED."""
    features = {
        "has_math_symbols": True,
        "has_drawn_shapes": False,
        "text_line_count": 4,
        "symbol_ratio": 0.20,
        "stroke_density": 0.1,
    }
    result = classify_content(features)
    assert result.content_type == ContentType.MIXED


# ---------------------------------------------------------------------------
# U-AI-CLS-05: Empty page classified as EMPTY
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_empty_page():
    """U-AI-CLS-05: no content -> EMPTY."""
    features = {
        "has_math_symbols": False,
        "has_drawn_shapes": False,
        "text_line_count": 0,
        "symbol_ratio": 0.0,
        "stroke_density": 0.0,
    }
    result = classify_content(features)
    assert result.content_type == ContentType.EMPTY
    assert result.type_confidence >= 0.9


# ---------------------------------------------------------------------------
# U-AI-CLS-06: features_used tracks which features contributed
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_features_used_tracked():
    """U-AI-CLS-06: result reports which feature keys were used."""
    features = {
        "has_math_symbols": True,
        "has_drawn_shapes": False,
        "text_line_count": 0,
        "symbol_ratio": 0.30,
        "stroke_density": 0.0,
    }
    result = classify_content(features)
    assert "has_math_symbols" in result.features_used
    assert "symbol_ratio" in result.features_used


# ---------------------------------------------------------------------------
# U-AI-CLS-07: Missing feature keys treated as defaults
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_missing_features_default():
    """U-AI-CLS-07: missing keys default to falsy values without error."""
    features = {}  # all keys missing
    result = classify_content(features)
    assert result.content_type == ContentType.EMPTY


# ---------------------------------------------------------------------------
# U-AI-CLS-08: Confidence is bounded to [0, 1]
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_confidence_bounded():
    """U-AI-CLS-08: type_confidence never exceeds 1.0."""
    features = {
        "has_math_symbols": True,
        "has_drawn_shapes": True,
        "text_line_count": 10,
        "symbol_ratio": 0.50,
        "stroke_density": 0.9,
    }
    result = classify_content(features)
    assert 0.0 <= result.type_confidence <= 1.0
