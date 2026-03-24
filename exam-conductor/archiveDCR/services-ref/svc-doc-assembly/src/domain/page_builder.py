"""Page document assembly. ZERO I/O — pure composition logic.

Combines rendered SVG + question metadata + miss indicators into a
PageDocument dataclass ready for storage and event publishing.
"""

from __future__ import annotations

from src.domain.miss_detector import detect_all_regions
from src.domain.models import (
    PageDocument,
    QuestionRegion,
    QuestionResult,
    Stroke,
    SyncMetadata,
)
from src.domain.renderer import render_page_svg


def build_page(
    strokes: list[Stroke],
    question_regions: list[QuestionRegion],
    sync_metadata: SyncMetadata | None,
    exam_id: str,
    student_id: str,
    page_number: int,
    page_width: float = 210.0,
    page_height: float = 297.0,
) -> PageDocument:
    """Assemble a complete page document.

    1. Render strokes to SVG.
    2. Detect miss indicators for each question region.
    3. Package into PageDocument.
    """
    svg_content = render_page_svg(strokes, page_width, page_height)

    miss_states = detect_all_regions(strokes, question_regions, sync_metadata)

    question_results = [
        QuestionResult(
            question_id=region.question_id,
            auto_state=miss_states[region.question_id],
        )
        for region in question_regions
    ]

    return PageDocument(
        exam_id=exam_id,
        student_id=student_id,
        page_number=page_number,
        svg_content=svg_content,
        question_results=question_results,
        page_width_mm=page_width,
        page_height_mm=page_height,
    )
