"""Unit tests for step detector — math step detection, text handling.

Test IDs: U-AI-STEP-01 through U-AI-STEP-08
Validation level: L3 (unit, domain, no I/O)
"""

import pytest

from src.domain.step_detector import (
    StepResult,
    detect_steps,
    detect_steps_diagram,
    detect_steps_formula,
    detect_steps_text,
)


# ---------------------------------------------------------------------------
# U-AI-STEP-01: Formula answer with explicit step markers
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_formula_explicit_steps():
    """U-AI-STEP-01: explicit 'Step N:' markers split into steps."""
    text = (
        "Step 1: Given x = 5\n"
        "Step 2: Substituting x into equation\n"
        "Step 3: Therefore y = 10"
    )
    result = detect_steps(text, "formula")
    assert isinstance(result, StepResult)
    assert result.step_count == 3
    assert "Given x = 5" in result.steps[0]


# ---------------------------------------------------------------------------
# U-AI-STEP-02: Formula answer with arrow markers
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_formula_arrow_markers():
    """U-AI-STEP-02: '=>' markers create step boundaries."""
    text = (
        "x^2 + 2x + 1 = 0\n"
        "=> (x + 1)^2 = 0\n"
        "=> x = -1"
    )
    result = detect_steps(text, "formula")
    assert result.step_count >= 2


# ---------------------------------------------------------------------------
# U-AI-STEP-03: Text answer splits by paragraphs
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_text_paragraph_split():
    """U-AI-STEP-03: text answers split into paragraphs as steps."""
    text = "First paragraph about photosynthesis.\n\nSecond paragraph about light."
    result = detect_steps(text, "text")
    assert result.step_count == 2
    assert "photosynthesis" in result.steps[0]
    assert "light" in result.steps[1]


# ---------------------------------------------------------------------------
# U-AI-STEP-04: Diagram is a single step
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_diagram_single_step():
    """U-AI-STEP-04: diagram content becomes exactly one step."""
    text = "Circle with radius r, center at O"
    result = detect_steps(text, "diagram")
    assert result.step_count == 1
    assert result.step_boundaries[0].label == "diagram"


# ---------------------------------------------------------------------------
# U-AI-STEP-05: Empty text returns zero steps
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_empty_text():
    """U-AI-STEP-05: empty input returns step_count=0."""
    result = detect_steps("", "formula")
    assert result.step_count == 0
    assert result.steps == []


# ---------------------------------------------------------------------------
# U-AI-STEP-06: Mixed type falls back to text if no formula steps
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_mixed_type_fallback():
    """U-AI-STEP-06: mixed type with plain text falls back to text splitting."""
    text = "Some explanation.\n\nAnother paragraph."
    result = detect_steps(text, "mixed")
    assert result.step_count == 2


# ---------------------------------------------------------------------------
# U-AI-STEP-07: Step boundaries have correct offsets
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_step_boundaries_offsets():
    """U-AI-STEP-07: step boundaries have non-negative start/end."""
    text = (
        "Step 1: Formula a = b + c\n"
        "Step 2: Answer a = 5"
    )
    result = detect_steps_formula(text)
    for boundary in result.step_boundaries:
        assert boundary.start >= 0
        assert boundary.end >= boundary.start
        assert boundary.label in ("formula", "substitution", "simplification", "answer", "text")


# ---------------------------------------------------------------------------
# U-AI-STEP-08: Whitespace-only text returns zero steps
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_whitespace_only():
    """U-AI-STEP-08: whitespace-only input returns empty result."""
    result = detect_steps("   \n\n   ", "text")
    assert result.step_count == 0


# ---------------------------------------------------------------------------
# U-AI-STEP-09: Substitution keyword detected
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_substitution_keyword():
    """U-AI-STEP-09: 'substituting' keyword is recognized as a step boundary."""
    text = (
        "Given: v = u + at\n"
        "Substituting u = 0, a = 10, t = 5\n"
        "Therefore v = 50"
    )
    result = detect_steps_formula(text)
    assert result.step_count >= 2
