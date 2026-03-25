"""
ExamPen DCR (Direct Character Recognition) Engine
===================================================

Template-overlay + LLM Vision OCR recognition and scoring over canonical
conducted-exam artifacts.

Ownership Declaration (per STATE_OWNERSHIP_MAP.md):
  - Writes: exampen_dcr_results
  - Reads from: evalpen_submissions, evalpen_answer_pages (ingest substrate), exam metadata
  - Never writes to: canonical conducted-exam artifacts, practice persistence, gate config
  - Transactional boundaries: recognized text + match result + score (atomic DCR result creation)

Architecture reference: DUAL_MODE_ARCHITECTURE.md §4
Failure modes: DCR-01, DCR-02, DCR-03
Test IDs: U-DCR-01, U-DCR-02, U-DCR-03, I-DCR-01, I-DCR-02, E2E-DCR-01
"""

from .models import (
    MatchType,
    PageRef,
    DCRSubmission,
    DCRSubmissionPage,
    RecognitionOutput,
    MatchOutput,
    DCRResult,
    DCRAuditEntry,
    DCREvaluateRequest,
    DCREvaluateResponse,
    AnswerKey,
)
from .recognizer import HWRRecognizer
from .matcher import TemplateMatcher
from .repository import DCRRepository
from .service import DCRService
from .template_overlay import (
    render_strokes_to_image,
    overlay_on_template,
    build_dcr_extraction_prompt,
    parse_dcr_extraction_response,
)

__all__ = [
    # Models
    "MatchType",
    "PageRef",
    "DCRSubmission",
    "DCRSubmissionPage",
    "RecognitionOutput",
    "MatchOutput",
    "DCRResult",
    "DCRAuditEntry",
    "DCREvaluateRequest",
    "DCREvaluateResponse",
    "AnswerKey",
    # Components
    "HWRRecognizer",
    "TemplateMatcher",
    "DCRRepository",
    "DCRService",
    # Template overlay utilities
    "render_strokes_to_image",
    "overlay_on_template",
    "build_dcr_extraction_prompt",
    "parse_dcr_extraction_response",
]
