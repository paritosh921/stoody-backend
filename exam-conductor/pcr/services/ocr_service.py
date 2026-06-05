"""
PCR OCR Service — Adapter layer for OCR / HWR engines backed by LLM Vision.

Defines a ``Protocol``-based adapter interface so the evaluation pipeline
is decoupled from any specific OCR backend.  Two concrete adapters are
provided:

- ``LLMVisionCameraAdapter`` — for camera/scan originated pages (JPEG/PNG)
- ``LLMVisionPenAdapter``    — for pen-originated pages (stroke vectors → PIL
  rasterisation → LLM Vision)

Both adapters route through the shared LLM gate (C4) and return ``PageOCR``
objects (list of ``TextBlock`` with bounding boxes) that feed directly into
the segmentation pipeline.

Spec authority: new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md §3
Test IDs:       I-PCR-01 (conducted artifact -> PageOCR -> detected responses)
Failure modes:  PCR-01 (detection failure -> flags + review)
"""

from __future__ import annotations

import base64
import importlib
import io
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ..domain.response_models import (
    BoundingBox,
    PageOCR,
    TextBlock,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# A4 defaults (mm)
# ---------------------------------------------------------------------------

_A4_WIDTH_MM = 210.0
_A4_HEIGHT_MM = 297.0

# Default stroke rendering dimensions (pixels)
_RENDER_WIDTH_PX = 1240   # ~148 DPI at A4 width
_RENDER_HEIGHT_PX = 1754  # ~148 DPI at A4 height

# Last-resort OCR model when the shared gate provider default cannot be resolved.
_DEFAULT_OCR_VISION_MODEL = "gpt-4o"

# LLM OCR extraction prompt
_OCR_PROMPT_VERSION = "exampen-qno-v1"
_OCR_EXTRACTION_PROMPT = (
    "This image is a high-contrast raster rendering of BLE digital pen strokes "
    "from an ExamPen answer sheet. Extract the visible handwritten or "
    "stroke-rendered text exactly as written. Pay special attention to exam "
    "answer markers in the form 'Q.No X.Ans' or close variants; these markers "
    "identify answer lines and must be preserved in the extracted text. "
    "Read line by line from top to bottom. For each detected text line or "
    "region, return one JSON object with \"text\" (string: recognised text) "
    "and \"confidence\" (float 0-1: recognition confidence). If text is faint "
    "or partially clipped, return the best visible transcription with lower "
    "confidence instead of dropping the line. If absolutely no text is "
    "visible, return an empty array: []. Return ONLY the JSON array, no "
    "markdown fences or extra text."
)

# Gate caller identity (already registered in ALLOWED_CALLER_IDS)
_CALLER_ID = "pcr_eval_core"


# ---------------------------------------------------------------------------
# OCR result envelope
# ---------------------------------------------------------------------------


@dataclass
class OCRResult:
    """Return envelope from any OCR adapter.

    Attributes
    ----------
    pages : list[PageOCR]
        One ``PageOCR`` per physical page, in page-number order.
    source : str
        ``"pen"`` or ``"camera"`` — forwarded to the segmenter.
    metadata : dict
        Adapter-specific metadata (model version, inference time, etc.).
    """

    pages: List[PageOCR] = field(default_factory=list)
    source: str = "camera"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol — the contract that any OCR backend must satisfy
# ---------------------------------------------------------------------------


@runtime_checkable
class OCRAdapter(Protocol):
    """Protocol for OCR / HWR adapters.

    Any backend that produces ``PageOCR`` from raw page artifacts
    can implement this protocol.  The eval core and submission service
    depend only on this interface, never on a concrete implementation.
    """

    async def recognize_pages(
        self,
        pages_data: List[Dict[str, Any]],
        *,
        source: str,
    ) -> OCRResult:
        """Run OCR / HWR on a list of raw page artifacts.

        Parameters
        ----------
        pages_data : list[dict]
            Raw page payloads from ``evalpen_answer_pages``.  Each dict
            contains at minimum:

            - ``page_number`` (int, 1-based)
            - ``raw_strokes`` (list[dict], pen path) **or**
            - ``raw_image_ref`` (str, camera path — S3 key or local path)
        source : str
            ``"pen"`` or ``"camera"`` — determines which recognition
            engine to use.

        Returns
        -------
        OCRResult
            Pages with recognized text blocks in page-space mm coordinates.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Gate protocol (vision-capable subset of LLMGate.call)
# ---------------------------------------------------------------------------


@runtime_checkable
class VisionGateProtocol(Protocol):
    """Minimal gate contract needed by vision OCR adapters.

    Satisfied by ``llm_gate.gate.LLMGate``.
    """

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
    ) -> Any:
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Shared: parse LLM OCR response into TextBlock list
# ---------------------------------------------------------------------------


def _parse_ocr_response_to_text_blocks(
    llm_content: str,
    page_width_mm: float,
    page_height_mm: float,
    source: str,
) -> List[TextBlock]:
    """Parse LLM vision OCR output into a list of ``TextBlock`` objects.

    The expected format from the LLM is a JSON array of objects::

        [{"text": "...", "confidence": 0.95}, ...]

    If the response is not valid JSON, the entire response is treated as
    a single text block spanning the full page.

    Parameters
    ----------
    llm_content : str
        Raw text content from the LLM gate response.
    page_width_mm : float
        Page width in mm (for default bbox calculation).
    page_height_mm : float
        Page height in mm (for default bbox calculation).
    source : str
        ``"pen"`` or ``"camera"`` — forwarded to each TextBlock.

    Returns
    -------
    list[TextBlock]
        Parsed text blocks with bounding boxes.
    """
    if not llm_content or not llm_content.strip():
        return []

    cleaned = llm_content.strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        # Remove opening fence (possibly with language tag)
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1:]
        # Remove closing fence
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3].rstrip()

    # Try JSON parse
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        # Fallback: entire response as one text block
        logger.debug(
            "LLM OCR response is not valid JSON — treating as single block"
        )
        if not cleaned:
            return []
        return [
            TextBlock(
                text=cleaned,
                bbox=BoundingBox(
                    x_min=0.0,
                    y_min=0.0,
                    x_max=page_width_mm,
                    y_max=page_height_mm,
                ),
                confidence=0.5,
                source=source,
            )
        ]

    # Handle JSON response
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict) and "text" in parsed:
        # Single block returned as an object instead of array
        items = [parsed]
    else:
        logger.warning(
            "LLM OCR response has unexpected JSON shape: %s",
            type(parsed).__name__,
        )
        return []

    text_blocks: List[TextBlock] = []
    num_items = max(len(items), 1)

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        text = item.get("text", "")
        if not text or not str(text).strip():
            continue

        confidence = item.get("confidence", 0.7)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.7

        # Distribute blocks vertically across the page when no bbox is
        # provided by the LLM.  Each block gets an equal vertical slice.
        y_min = (idx / num_items) * page_height_mm
        y_max = ((idx + 1) / num_items) * page_height_mm

        # If the LLM provided bbox coordinates, use them
        bbox_data = item.get("bbox")
        if isinstance(bbox_data, dict):
            try:
                bbox = BoundingBox(
                    x_min=float(bbox_data.get("x_min", 0.0)),
                    y_min=float(bbox_data.get("y_min", y_min)),
                    x_max=float(bbox_data.get("x_max", page_width_mm)),
                    y_max=float(bbox_data.get("y_max", y_max)),
                )
            except (TypeError, ValueError):
                bbox = BoundingBox(
                    x_min=0.0, y_min=y_min,
                    x_max=page_width_mm, y_max=y_max,
                )
        else:
            bbox = BoundingBox(
                x_min=0.0, y_min=y_min,
                x_max=page_width_mm, y_max=y_max,
            )

        text_blocks.append(
            TextBlock(
                text=str(text).strip(),
                bbox=bbox,
                confidence=confidence,
                source=source,
            )
        )

    return text_blocks


# ---------------------------------------------------------------------------
# Shared: build multimodal messages for vision OCR
# ---------------------------------------------------------------------------


def _build_vision_messages(image_base64: str, media_type: str = "image/png") -> List[Dict[str, Any]]:
    """Build OpenAI-compatible multimodal messages array for a single image.

    The messages array is forwarded through the gate to any supported
    provider (OpenAI, Anthropic, Gemini) — the gate and provider layer
    handle format translation.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _OCR_EXTRACTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_base64}",
                        "detail": "high",
                    },
                },
            ],
        }
    ]


def _get_ocr_vision_model() -> str:
    """Resolve the model to use for vision OCR calls.

    Reads from ``OCR_VISION_MODEL`` env var when an OCR-specific override is
    configured. Otherwise, defer to the shared gate provider default so the
    OCR adapter follows the active ``AI_PROVIDER`` configuration.
    """
    override = os.getenv("OCR_VISION_MODEL", "").strip()
    if override:
        return override
    try:
        # The package name contains a hyphen, so use importlib just like the
        # route-layer load_exampen helper.
        provider = importlib.import_module("exam-conductor.llm_gate.provider")
        get_default_model = provider.get_default_model
    except (ImportError, AttributeError):
        logger.exception(
            "Failed to import gate provider default resolver; "
            "falling back to %s",
            _DEFAULT_OCR_VISION_MODEL,
        )
        return _DEFAULT_OCR_VISION_MODEL
    try:
        return get_default_model()
    except Exception:
        logger.exception(
            "Failed to resolve OCR model from gate provider defaults"
        )
        raise


# ---------------------------------------------------------------------------
# LLM Vision Camera Adapter — camera / scan path
# ---------------------------------------------------------------------------


class LLMVisionCameraAdapter:
    """OCR adapter for camera/scan originated pages using LLM Vision.

    LLM Vision adapter for camera/scan pages.  Routes through the shared LLM gate
    (C4) with multimodal vision messages.

    Spec path (PCR_EVAL_ENGINE_SPEC §3):
        JPEG/PNG images → LLM Vision → TextBlocks + bbox
    """

    def __init__(self, gate: VisionGateProtocol) -> None:
        self._gate = gate
        logger.info("LLMVisionCameraAdapter initialized")

    async def recognize_pages(
        self,
        pages_data: List[Dict[str, Any]],
        *,
        source: str = "camera",
    ) -> OCRResult:
        """Recognize text from camera/scan page images via LLM Vision."""
        result_pages: List[PageOCR] = []
        model_id = _get_ocr_vision_model()

        for page_data in sorted(pages_data, key=lambda p: p.get("page_number", 0)):
            page_number = page_data.get("page_number", 1)
            raw_image_ref = page_data.get("raw_image_ref")

            if not raw_image_ref:
                logger.warning(
                    "Page %d has no raw_image_ref — skipping OCR",
                    page_number,
                )
                continue

            # A4 portrait defaults
            page_width_mm = _A4_WIDTH_MM
            page_height_mm = _A4_HEIGHT_MM
            image_width_px = page_data.get("image_width_px")
            image_height_px = page_data.get("image_height_px")

            # Get image as base64
            image_b64 = _resolve_image_base64(raw_image_ref)
            if image_b64 is None:
                logger.warning(
                    "Page %d: could not resolve image from %s — skipping",
                    page_number,
                    raw_image_ref[:60] if isinstance(raw_image_ref, str) else "?",
                )
                result_pages.append(PageOCR(
                    page_number=page_number,
                    page_width_mm=page_width_mm,
                    page_height_mm=page_height_mm,
                    text_blocks=[],
                    image_width_px=image_width_px,
                    image_height_px=image_height_px,
                    source="camera",
                    mean_ocr_confidence=0.0,
                ))
                continue

            # Detect media type from base64 header or default to JPEG
            media_type = _detect_media_type(raw_image_ref)

            # Build multimodal messages and call gate
            messages = _build_vision_messages(image_b64, media_type)
            try:
                gate_response = await self._gate.call(
                    model_id=model_id,
                    prompt="",
                    caller_id=_CALLER_ID,
                    messages=messages,
                    max_output_tokens=2048,
                    temperature=0.0,
                    metadata={
                        "pcr_stage": "ocr_camera",
                        "page_number": page_number,
                        "ocr_prompt_version": _OCR_PROMPT_VERSION,
                    },
                )
                llm_content = gate_response.content
            except Exception as exc:
                logger.exception(
                    "LLM Vision OCR failed for page %d (camera)",
                    page_number,
                )
                raise RuntimeError(
                    f"LLM Vision OCR failed for page {page_number} (camera)"
                ) from exc

            # Parse LLM response into TextBlocks
            text_blocks = _parse_ocr_response_to_text_blocks(
                llm_content, page_width_mm, page_height_mm, source="camera",
            )

            mean_conf = (
                sum(tb.confidence for tb in text_blocks) / len(text_blocks)
                if text_blocks
                else 0.0
            )

            page_ocr = PageOCR(
                page_number=page_number,
                page_width_mm=page_width_mm,
                page_height_mm=page_height_mm,
                text_blocks=text_blocks,
                image_width_px=image_width_px,
                image_height_px=image_height_px,
                source="camera",
                mean_ocr_confidence=round(mean_conf, 4),
            )
            result_pages.append(page_ocr)

            logger.debug(
                "LLM Vision camera OCR: page %d — %d text blocks",
                page_number,
                len(text_blocks),
            )

        return OCRResult(
            pages=result_pages,
            source="camera",
            metadata={"adapter": "LLMVisionCamera", "model": model_id},
        )


# ---------------------------------------------------------------------------
# LLM Vision Pen Adapter — BLE pen path
# ---------------------------------------------------------------------------


class LLMVisionPenAdapter:
    """OCR adapter for pen-originated pages using LLM Vision.

    LLM Vision adapter for pen-originated pages.  Rasterises stroke vectors to a
    PNG image using PIL/Pillow and then routes through the shared LLM gate
    (C4) with multimodal vision messages.

    Spec path (PCR_EVAL_ENGINE_SPEC §3):
        Stroke vectors → PIL rasterisation → LLM Vision → TextBlocks + bbox
    """

    def __init__(self, gate: VisionGateProtocol) -> None:
        self._gate = gate
        logger.info("LLMVisionPenAdapter initialized")

    async def recognize_pages(
        self,
        pages_data: List[Dict[str, Any]],
        *,
        source: str = "pen",
    ) -> OCRResult:
        """Recognize text from BLE pen stroke vectors via LLM Vision."""
        result_pages: List[PageOCR] = []
        model_id = _get_ocr_vision_model()

        for page_data in sorted(pages_data, key=lambda p: p.get("page_number", 0)):
            page_number = page_data.get("page_number", 1)
            raw_strokes = page_data.get("raw_strokes")

            if not raw_strokes:
                logger.warning(
                    "Page %d has no raw_strokes — skipping HWR",
                    page_number,
                )
                continue

            # Standard A4 portrait
            page_width_mm = _A4_WIDTH_MM
            page_height_mm = _A4_HEIGHT_MM

            # Render strokes to PNG image
            image_b64 = _render_strokes_to_base64(
                raw_strokes,
                page_width_mm,
                page_height_mm,
            )
            if image_b64 is None:
                logger.warning(
                    "Page %d: stroke rasterisation failed — skipping",
                    page_number,
                )
                result_pages.append(PageOCR(
                    page_number=page_number,
                    page_width_mm=page_width_mm,
                    page_height_mm=page_height_mm,
                    text_blocks=[],
                    source="pen",
                    mean_ocr_confidence=0.0,
                ))
                continue

            # Build multimodal messages and call gate
            messages = _build_vision_messages(image_b64, "image/png")
            try:
                gate_response = await self._gate.call(
                    model_id=model_id,
                    prompt="",
                    caller_id=_CALLER_ID,
                    messages=messages,
                    max_output_tokens=2048,
                    temperature=0.0,
                    metadata={
                        "pcr_stage": "ocr_pen",
                        "page_number": page_number,
                        "stroke_count": len(raw_strokes),
                        "ocr_prompt_version": _OCR_PROMPT_VERSION,
                    },
                )
                llm_content = gate_response.content
            except Exception as exc:
                logger.exception(
                    "LLM Vision OCR failed for page %d (pen)",
                    page_number,
                )
                raise RuntimeError(
                    f"LLM Vision OCR failed for page {page_number} (pen)"
                ) from exc

            # Parse LLM response into TextBlocks
            text_blocks = _parse_ocr_response_to_text_blocks(
                llm_content, page_width_mm, page_height_mm, source="pen",
            )

            mean_conf = (
                sum(tb.confidence for tb in text_blocks) / len(text_blocks)
                if text_blocks
                else 0.0
            )

            page_ocr = PageOCR(
                page_number=page_number,
                page_width_mm=page_width_mm,
                page_height_mm=page_height_mm,
                text_blocks=text_blocks,
                source="pen",
                mean_ocr_confidence=round(mean_conf, 4),
            )
            result_pages.append(page_ocr)

            logger.debug(
                "LLM Vision pen OCR: page %d — %d strokes, %d text blocks",
                page_number,
                len(raw_strokes),
                len(text_blocks),
            )

        return OCRResult(
            pages=result_pages,
            source="pen",
            metadata={"adapter": "LLMVisionPen", "model": model_id},
        )


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def create_ocr_adapter(source: str, gate: VisionGateProtocol) -> OCRAdapter:
    """Create the appropriate OCR adapter based on artifact source.

    Parameters
    ----------
    source : str
        ``"pen"`` / ``"ble_pen"`` — returns ``LLMVisionPenAdapter``
        ``"camera"``             — returns ``LLMVisionCameraAdapter``
    gate : VisionGateProtocol
        An initialised LLM gate instance (satisfies ``LLMGate``).

    Raises
    ------
    ValueError
        If the source is unrecognized.
    """
    if source in ("pen", "ble_pen"):
        return LLMVisionPenAdapter(gate=gate)
    elif source == "camera":
        return LLMVisionCameraAdapter(gate=gate)
    else:
        raise ValueError(
            f"Unsupported OCR source: {source!r}. "
            f"Expected 'pen', 'ble_pen', or 'camera'."
        )


# ---------------------------------------------------------------------------
# Image resolution helpers
# ---------------------------------------------------------------------------


def _resolve_image_base64(raw_image_ref: str) -> Optional[str]:
    """Resolve a raw_image_ref to a base64-encoded string.

    Handles:
    - Already base64 data (starts with ``/9j/`` for JPEG, ``iVBOR`` for PNG,
      or is a data URI ``data:image/...;base64,...``)
    - S3 keys (TODO: implement S3 fetch)
    - Local file paths (TODO: implement local file read)

    Returns ``None`` if the reference cannot be resolved.
    """
    if not raw_image_ref or not isinstance(raw_image_ref, str):
        return None

    ref = raw_image_ref.strip()

    # Data URI — extract the base64 payload
    if ref.startswith("data:"):
        parts = ref.split(",", 1)
        if len(parts) == 2:
            return parts[1]
        return None

    # Heuristic: if it looks like raw base64 (starts with known image
    # magic bytes in base64), use it directly
    if ref.startswith(("/9j/", "iVBOR")):
        return ref

    # Attempt to detect if the entire string is valid base64
    # (conservative: at least 100 chars, no slashes that look like paths)
    if len(ref) > 100 and "/" not in ref[:20]:
        try:
            base64.b64decode(ref[:64], validate=True)
            return ref
        except Exception:
            pass

    # TODO: S3 key — fetch from S3 and return base64
    # if ref.startswith("s3://") or "/" in ref:
    #     return await _fetch_from_s3(ref)

    logger.debug(
        "raw_image_ref does not appear to be base64 or data URI: %s...",
        ref[:60],
    )
    return None


def _detect_media_type(raw_image_ref: str) -> str:
    """Detect the media type of an image from its reference or content.

    Returns ``"image/jpeg"`` or ``"image/png"`` (default: ``"image/jpeg"``).
    """
    if not raw_image_ref:
        return "image/jpeg"

    ref = raw_image_ref.strip()

    # Data URI has explicit media type
    if ref.startswith("data:"):
        # data:image/png;base64,...
        try:
            media_part = ref.split(";")[0].split(":")[1]
            return media_part
        except (IndexError, ValueError):
            pass

    # PNG magic in base64
    if ref.startswith("iVBOR"):
        return "image/png"

    # JPEG magic in base64
    if ref.startswith("/9j/"):
        return "image/jpeg"

    return "image/jpeg"


# ---------------------------------------------------------------------------
# Stroke rasterisation (PIL / Pillow)
# ---------------------------------------------------------------------------


def _render_strokes_to_base64(
    raw_strokes: List[Dict[str, Any]],
    page_width_mm: float,
    page_height_mm: float,
    width_px: int = _RENDER_WIDTH_PX,
    height_px: int = _RENDER_HEIGHT_PX,
) -> Optional[str]:
    """Render stroke vectors to a PNG image and return as base64.

    Each stroke dict is expected to have a ``points`` list.  Each point
    is a list/tuple of at least ``[x, y, ...]`` where x and y are in
    page-space millimetres (canonical stroke model, TRACK_3_1 spec).

    Falls back gracefully if PIL is not available or strokes are empty.

    Parameters
    ----------
    raw_strokes : list[dict]
        Stroke dicts with ``points`` arrays.
    page_width_mm, page_height_mm : float
        Page dimensions in mm for coordinate mapping.
    width_px, height_px : int
        Output image size in pixels.

    Returns
    -------
    str or None
        Base64-encoded PNG image, or ``None`` on failure.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        logger.error(
            "PIL/Pillow is not installed — cannot rasterise strokes. "
            "Install with: pip install Pillow"
        )
        return None

    if not raw_strokes:
        return None

    # Scale factors: mm -> px
    scale_x = width_px / page_width_mm if page_width_mm > 0 else 1.0
    scale_y = height_px / page_height_mm if page_height_mm > 0 else 1.0

    # Create white background image
    img = Image.new("RGB", (width_px, height_px), "white")
    draw = ImageDraw.Draw(img)

    strokes_rendered = 0
    min_x = width_px
    min_y = height_px
    max_x = 0.0
    max_y = 0.0

    for stroke in raw_strokes:
        points = stroke.get("points", [])
        if not points or len(points) < 2:
            continue

        # Extract (x, y) pixel coordinates from canonical points
        px_points: List[tuple[float, float]] = []
        for pt in points:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                x_mm, y_mm = float(pt[0]), float(pt[1])
            elif isinstance(pt, dict):
                x_mm = float(pt.get("x", pt.get("0", 0)))
                y_mm = float(pt.get("y", pt.get("1", 0)))
            else:
                continue

            px_x = x_mm * scale_x
            px_y = y_mm * scale_y
            px_points.append((px_x, px_y))

        if len(px_points) >= 2:
            for px_x, px_y in px_points:
                min_x = min(min_x, px_x)
                min_y = min(min_y, px_y)
                max_x = max(max_x, px_x)
                max_y = max(max_y, px_y)

            # Get stroke width (default 2px)
            stroke_width = stroke.get("strokeWidth", 2)
            try:
                stroke_width = max(2, int(float(stroke_width)))
            except (TypeError, ValueError):
                stroke_width = 2

            # Get stroke colour (default black)
            colour = stroke.get("color", stroke.get("colour", "#000000"))
            if not isinstance(colour, str) or not colour.startswith("#"):
                colour = "#000000"

            draw.line(px_points, fill=colour, width=stroke_width)
            strokes_rendered += 1

    if strokes_rendered == 0:
        logger.debug("No renderable strokes found — skipping rasterisation")
        return None

    if max_x > min_x and max_y > min_y:
        margin_px = 48
        left = max(0, int(min_x) - margin_px)
        top = max(0, int(min_y) - margin_px)
        right = min(width_px, int(max_x) + margin_px)
        bottom = min(height_px, int(max_y) + margin_px)
        crop_width = right - left
        crop_height = bottom - top
        if crop_width > 0 and crop_height > 0:
            img = img.crop((left, top, right, bottom))
            scale = min(
                4.0,
                max(
                    1.0,
                    min(width_px / crop_width, height_px / crop_height),
                ),
            )
            if scale > 1.0:
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                img = img.resize(
                    (
                        max(1, int(crop_width * scale)),
                        max(1, int(crop_height * scale)),
                    ),
                    resampling,
                )

    # Encode as PNG and return base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")
