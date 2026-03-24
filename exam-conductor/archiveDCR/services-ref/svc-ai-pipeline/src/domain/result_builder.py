"""AI result assembly — pure domain logic, ZERO I/O.

Combines HWR, step detection, and classification outputs into a single
AIResult that matches the ai.result.schema.json event contract.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from src.domain.classifier import ContentType
from src.domain.hwr_engine import HWRResult
from src.domain.step_detector import StepResult


@dataclass(frozen=True)
class QuestionResult:
    """AI result for a single question, matching the event schema."""

    question_id: str
    recognized_text: str
    confidence: float
    step_breakdown: list[str] = field(default_factory=list)
    content_type: str = "text"
    flagged_for_review: bool = False


@dataclass(frozen=True)
class AIResult:
    """Complete AI result for a page, matching ai.result.schema.json."""

    event_id: str
    event_type: str = "ai.result"
    event_version: str = "1.0.0"
    occurred_at: str = ""
    exam_id: str = ""
    student_id: str = ""
    model_version: str = ""
    source_type: str = "strokes"
    question_results: list[QuestionResult] = field(default_factory=list)


def build_question_result(
    hwr: HWRResult,
    steps: StepResult,
    content_type: ContentType,
    question_id: str,
) -> QuestionResult:
    """Assemble a single question result from pipeline outputs."""
    return QuestionResult(
        question_id=question_id,
        recognized_text=hwr.recognized_text,
        confidence=round(hwr.confidence, 4),
        step_breakdown=list(steps.steps),
        content_type=content_type.value,
        flagged_for_review=hwr.flagged_for_review,
    )


def build_result(
    question_results: list[QuestionResult],
    exam_id: str,
    student_id: str,
    model_version: str,
    source_type: str = "strokes",
) -> AIResult:
    """Assemble the full AIResult event payload.

    Parameters
    ----------
    question_results:
        Per-question results from the pipeline.
    exam_id, student_id:
        Identifiers from the page.ready event.
    model_version:
        Version string of the model(s) used.
    source_type:
        ``"strokes"`` or ``"copy_image"``.
    """
    return AIResult(
        event_id=str(uuid4()),
        event_type="ai.result",
        event_version="1.0.0",
        occurred_at=datetime.now(timezone.utc).isoformat(),
        exam_id=exam_id,
        student_id=student_id,
        model_version=model_version,
        source_type=source_type,
        question_results=question_results,
    )
