"""Mathematical step detection — pure domain logic, ZERO I/O.

Splits recognized answer text into logical solution steps
(formula, substitution, simplification, answer) based on question type.
"""

from dataclasses import dataclass, field
from enum import Enum
import re


class QuestionType(str, Enum):
    """Supported answer types for step detection."""

    FORMULA = "formula"
    TEXT = "text"
    DIAGRAM = "diagram"
    MIXED = "mixed"


@dataclass(frozen=True)
class StepBoundary:
    """Character-offset boundary of a detected step."""

    start: int
    end: int
    label: str


@dataclass(frozen=True)
class StepResult:
    """Result of step detection on recognized text."""

    steps: list[str] = field(default_factory=list)
    step_count: int = 0
    step_boundaries: list[StepBoundary] = field(default_factory=list)


# Patterns that signal a new mathematical step.
_MATH_STEP_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)^\s*(?:step\s*\d+|given|find|formula|solution|answer)\s*[:.]"),
    re.compile(r"^\s*(?:=>|->|∴|therefore|hence|thus)\s", re.IGNORECASE),
    re.compile(r"^\s*(?:substituting|putting|replacing)\b", re.IGNORECASE),
]

# Heuristic: lines containing '=' are likely formula/computation lines.
_EQUALS_LINE = re.compile(r"=")


def _split_into_lines(text: str) -> list[str]:
    """Split text on newlines, preserving non-empty lines."""
    return [line for line in text.split("\n") if line.strip()]


def _is_step_boundary(line: str) -> bool:
    """Return True if the line signals the start of a new step."""
    return any(p.search(line) for p in _MATH_STEP_PATTERNS)


def _classify_line(line: str) -> str:
    """Return a label for the line: formula, substitution, simplification, answer, or text."""
    stripped = line.strip().lower()
    if stripped.startswith(("answer", "ans", "∴", "therefore", "hence")):
        return "answer"
    if re.search(r"(?i)substitut|putting|replacing", stripped):
        return "substitution"
    if _EQUALS_LINE.search(stripped):
        return "simplification"
    return "text"


def detect_steps_formula(text: str) -> StepResult:
    """Detect steps in a mathematical/formula answer."""
    lines = _split_into_lines(text)
    if not lines:
        return StepResult()

    steps: list[str] = []
    boundaries: list[StepBoundary] = []
    current_step_lines: list[str] = []
    current_start = 0
    offset = 0

    for line in lines:
        if _is_step_boundary(line) and current_step_lines:
            step_text = "\n".join(current_step_lines)
            label = _classify_line(current_step_lines[0])
            steps.append(step_text)
            boundaries.append(StepBoundary(
                start=current_start, end=offset - 1, label=label,
            ))
            current_step_lines = []
            current_start = offset

        current_step_lines.append(line)
        offset += len(line) + 1  # +1 for the newline

    # Flush remaining
    if current_step_lines:
        step_text = "\n".join(current_step_lines)
        label = _classify_line(current_step_lines[0])
        steps.append(step_text)
        boundaries.append(StepBoundary(
            start=current_start, end=offset - 1, label=label,
        ))

    return StepResult(steps=steps, step_count=len(steps), step_boundaries=boundaries)


def detect_steps_text(text: str) -> StepResult:
    """For text answers, each paragraph is a step."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return StepResult()

    boundaries: list[StepBoundary] = []
    offset = 0
    for para in paragraphs:
        boundaries.append(StepBoundary(
            start=offset, end=offset + len(para), label="text",
        ))
        offset += len(para) + 2  # +2 for double newline

    return StepResult(
        steps=paragraphs,
        step_count=len(paragraphs),
        step_boundaries=boundaries,
    )


def detect_steps_diagram(text: str) -> StepResult:
    """Diagrams are treated as a single step."""
    if not text.strip():
        return StepResult()
    return StepResult(
        steps=[text.strip()],
        step_count=1,
        step_boundaries=[StepBoundary(start=0, end=len(text), label="diagram")],
    )


def detect_steps(recognized_text: str, question_type: str) -> StepResult:
    """Route to the appropriate step detector based on question type.

    Parameters
    ----------
    recognized_text:
        The HWR-recognized answer text.
    question_type:
        One of "formula", "text", "diagram", "mixed".
    """
    qtype = question_type.lower()
    if qtype == QuestionType.FORMULA:
        return detect_steps_formula(recognized_text)
    if qtype == QuestionType.DIAGRAM:
        return detect_steps_diagram(recognized_text)
    if qtype == QuestionType.TEXT:
        return detect_steps_text(recognized_text)
    # mixed — try formula detection first, fall back to text
    result = detect_steps_formula(recognized_text)
    if result.step_count <= 1:
        return detect_steps_text(recognized_text)
    return result
