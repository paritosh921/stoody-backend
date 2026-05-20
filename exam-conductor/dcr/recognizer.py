"""
DCR Vision Recognizer — LLM-based handwriting recognition via template overlay.

Implements a template-overlay + LLM Vision OCR
pipeline:

    1. Render canonical stroke vectors to a rasterised image.
    2. Overlay the stroke image onto the answer-sheet template grid.
    3. Send the merged image to the LLM gate (``dcr_ai`` caller) for OCR.
    4. Parse the LLM response into per-question recognised text.

Architecture: DUAL_MODE_ARCHITECTURE.md S4.3 (template overlay variant)
Test ID: U-DCR-01 — Vision OCR output normalised into DCR recognition input
Failure mode: DCR-01 — LLM confidence heuristic too low -> route to fallback

Hard constraints:
  - C4: All LLM calls go through the gate (``dcr_ai`` caller).
  - C5: Reads canonical artifacts from the ingest substrate; does not own them.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from .models import (
    DCRSubmissionPage,
    RecognitionOutput,
)
from .template_overlay import (
    build_dcr_extraction_prompt,
    build_vision_messages,
    overlay_on_template,
    parse_dcr_extraction_response,
    render_strokes_to_image,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default confidence threshold for low-confidence flagging (DCR-01)
# ---------------------------------------------------------------------------
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.40

# Default page dimensions (A4 in mm) used when the page doesn't specify them.
_DEFAULT_PAGE_WIDTH_MM = 210.0
_DEFAULT_PAGE_HEIGHT_MM = 297.0
# Landscape A4 — used when the template image is wider than tall.
_LANDSCAPE_PAGE_WIDTH_MM = 297.0
_LANDSCAPE_PAGE_HEIGHT_MM = 210.0


def _infer_page_size_from_template(template_image: bytes) -> Optional[Tuple[float, float]]:
    """
    Pick portrait or landscape A4 dimensions from a template image's aspect.

    Returns ``(width_mm, height_mm)`` or ``None`` if the image cannot be read.
    Pillow is already a dependency for stroke rasterisation; this function
    deliberately keeps the same import scope.
    """
    try:
        from PIL import Image  # local import to keep the module's import surface unchanged
        with Image.open(io.BytesIO(template_image)) as img:
            w, h = img.size
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    if w > h:
        return (_LANDSCAPE_PAGE_WIDTH_MM, _LANDSCAPE_PAGE_HEIGHT_MM)
    return (_DEFAULT_PAGE_WIDTH_MM, _DEFAULT_PAGE_HEIGHT_MM)

# Default DPI for stroke rasterisation.
_DEFAULT_DPI = 150

# Default high-confidence value assigned to successful LLM extractions.
# The LLM does not return a numeric confidence, so we assign a reasonable
# constant when the response is well-formed.
_LLM_DEFAULT_CONFIDENCE = 0.85

# Confidence assigned when the LLM returns empty text for a question.
_LLM_EMPTY_CONFIDENCE = 0.0

# Default model ID for the gate call.
_DEFAULT_GATE_MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# Abstract recognizer protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class RecognizerProtocol(Protocol):
    """
    Protocol that any HWR recognizer backend must satisfy.

    This exists so that the Vision implementation can be swapped for a mock
    in tests (U-DCR-01).
    """

    async def recognize_page(
        self,
        page: DCRSubmissionPage,
        question_regions: List[Dict[str, Any]],
        *,
        template_image: Optional[bytes] = None,
    ) -> List[RecognitionOutput]:
        """
        Recognise all question regions on a single page.

        Parameters
        ----------
        page
            Canonical page with stroke vectors.
        question_regions
            List of dicts describing each question's spatial region:
            ``{"question_id": str, "bbox": [x, y, w, h]}`` where bbox is in
            page-space millimetres matching the canonical stroke coordinate
            system.
        template_image
            Optional PNG bytes of the answer-sheet template.  When provided
            the strokes are overlaid onto this template before OCR.

        Returns
        -------
        list[RecognitionOutput]
            One entry per question region.
        """
        ...


# ---------------------------------------------------------------------------
# Gate protocol (matches LLMGate.call signature)
# ---------------------------------------------------------------------------

class _GateProtocol(Protocol):
    async def call(
        self,
        model_id: str,
        prompt: str,
        caller_id: str,
        *,
        messages: Optional[List[Dict[str, Any]]] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Vision Recognizer (template overlay + LLM OCR)
# ---------------------------------------------------------------------------

class HWRRecognizer:
    """
    LLM-Vision handwriting recogniser for the DCR engine.

    Pipeline per page:
        canonical strokes --> rasterise to PNG
            --> overlay on template grid (if template provided)
            --> LLM Vision OCR via gate (dcr_ai caller)
            --> parse JSON response
            --> RecognitionOutput per question

    The gate handles provider selection (OpenAI, Anthropic, Gemini) based on
    environment configuration.
    """

    def __init__(
        self,
        gate: _GateProtocol,
        *,
        gate_model_id: str = _DEFAULT_GATE_MODEL,
        low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        dpi: int = _DEFAULT_DPI,
    ) -> None:
        """
        Parameters
        ----------
        gate
            LLM gate instance (``LLMGate`` or any object satisfying the
            ``_GateProtocol``).  All LLM calls are routed through this.
        gate_model_id
            Model identifier passed to the gate for OCR calls.
        low_confidence_threshold
            Confidence values below this trigger the low-confidence flag
            (DCR-01).  Callers can route flagged results to a secondary
            fallback.
        dpi
            Resolution for stroke rasterisation (pixels per inch).
        """
        self._gate = gate
        self._gate_model_id = gate_model_id
        self.low_confidence_threshold = low_confidence_threshold
        self._dpi = dpi

    # ------------------------------------------------------------------
    # Public recognition interface
    # ------------------------------------------------------------------

    async def recognize_page(
        self,
        page: DCRSubmissionPage,
        question_regions: List[Dict[str, Any]],
        *,
        template_image: Optional[bytes] = None,
    ) -> List[RecognitionOutput]:
        """
        Recognise all question regions on a page using LLM Vision OCR.

        Parameters
        ----------
        page
            Canonical page with ``raw_strokes``.
        question_regions
            ``[{"question_id": str, "bbox": [x, y, w, h]}, ...]``
        template_image
            Optional template PNG.  When provided the rendered strokes are
            overlaid on this image before being sent to the LLM.

        Returns
        -------
        list[RecognitionOutput]
            One per question region.  Confidence below
            ``low_confidence_threshold`` signals DCR-01.
        """
        raw_strokes = page.raw_strokes or []
        question_ids = [r["question_id"] for r in question_regions]

        if not raw_strokes:
            # No strokes on the page — return empty results.
            logger.debug(
                "No strokes on page %d; returning empty recognition outputs.",
                page.page_number,
            )
            return [
                RecognitionOutput(
                    question_id=qid,
                    recognized_text="",
                    confidence=_LLM_EMPTY_CONFIDENCE,
                    page_number=page.page_number,
                    raw_logits=None,
                )
                for qid in question_ids
            ]

        # ── 1. Render strokes to image ──────────────────────────────────
        page_w = getattr(page, "page_width_mm", None)
        page_h = getattr(page, "page_height_mm", None)
        if (page_w is None or page_h is None) and template_image is not None:
            inferred = _infer_page_size_from_template(template_image)
            if inferred is not None:
                page_w, page_h = inferred
        page_w = page_w or _DEFAULT_PAGE_WIDTH_MM
        page_h = page_h or _DEFAULT_PAGE_HEIGHT_MM

        stroke_image = render_strokes_to_image(
            raw_strokes, page_w, page_h, dpi=self._dpi
        )

        # ── 2. Template overlay (if available) ──────────────────────────
        if template_image is not None:
            merged_image = overlay_on_template(stroke_image, template_image)
        else:
            merged_image = stroke_image

        # ── 3. Build extraction prompt ──────────────────────────────────
        prompt_text = build_dcr_extraction_prompt(
            question_count=len(question_ids),
            question_ids=question_ids,
        )

        # ── 4. Build multimodal messages and call gate ──────────────────
        messages = build_vision_messages(prompt_text, merged_image)

        try:
            gate_response = await self._gate.call(
                model_id=self._gate_model_id,
                prompt=prompt_text,  # used for logging / token estimation
                caller_id="dcr_ai",
                messages=messages,
                max_output_tokens=1024,
                temperature=0.0,
                metadata={
                    "exam_id": page.exam_id,
                    "student_id": page.student_id,
                    "page_number": page.page_number,
                },
            )
            llm_content = gate_response.content
        except Exception as exc:
            logger.error(
                "DCR Vision OCR gate call failed for page %d of exam %s: %s",
                page.page_number,
                page.exam_id,
                exc,
            )
            # Return zero-confidence empty results on gate failure.
            return [
                RecognitionOutput(
                    question_id=qid,
                    recognized_text="",
                    confidence=0.0,
                    page_number=page.page_number,
                    raw_logits=None,
                )
                for qid in question_ids
            ]

        # ── 5. Parse LLM response ──────────────────────────────────────
        extracted = parse_dcr_extraction_response(llm_content, question_ids)

        # ── 6. Build RecognitionOutput per question ─────────────────────
        outputs: List[RecognitionOutput] = []
        for qid in question_ids:
            text = extracted.get(qid, "")
            confidence = _LLM_DEFAULT_CONFIDENCE if text else _LLM_EMPTY_CONFIDENCE

            output = RecognitionOutput(
                question_id=qid,
                recognized_text=text,
                confidence=confidence,
                page_number=page.page_number,
                raw_logits=None,
            )
            outputs.append(output)

            if confidence < self.low_confidence_threshold:
                logger.info(
                    "DCR-01: Low confidence (%.3f) for question %s on page %d; "
                    "consider gate fallback.",
                    confidence,
                    qid,
                    page.page_number,
                )

        return outputs

    async def recognize_submission(
        self,
        pages: List[DCRSubmissionPage],
        question_regions_by_page: Dict[int, List[Dict[str, Any]]],
        *,
        template_image: Optional[bytes] = None,
    ) -> List[RecognitionOutput]:
        """
        Recognise all question regions across multiple pages.

        Parameters
        ----------
        pages
            List of canonical pages.
        question_regions_by_page
            Mapping from ``page_number`` to list of question regions for
            that page.
        template_image
            Optional template PNG applied to every page.

        Returns
        -------
        list[RecognitionOutput]
            Aggregated results across all pages.
        """
        all_outputs: List[RecognitionOutput] = []
        for page in pages:
            regions = question_regions_by_page.get(page.page_number, [])
            if not regions:
                # No explicit regions for this page.  Generate a whole-page
                # region so the recogniser processes all strokes on the page
                # rather than skipping it entirely.
                regions = [{"question_id": f"page_{page.page_number}", "bbox": None}]
                logger.debug(
                    "No regions for page %d -- using whole-page fallback",
                    page.page_number,
                )
            page_outputs = await self.recognize_page(
                page, regions, template_image=template_image
            )
            all_outputs.extend(page_outputs)
        return all_outputs

    # ------------------------------------------------------------------
    # Low-confidence check
    # ------------------------------------------------------------------

    def is_low_confidence(self, output: RecognitionOutput) -> bool:
        """
        Return ``True`` if the recognition output is below the configured
        low-confidence threshold (DCR-01).
        """
        return output.confidence < self.low_confidence_threshold
