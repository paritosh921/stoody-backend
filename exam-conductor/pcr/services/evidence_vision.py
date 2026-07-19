"""Load student answer-page images for vision-based PCR evaluation.

Text-only OCR is enough for short numeric answers.  Diagrams (Venn diagrams,
factor sketches, boxed values, geometric constructions) are invisible to a
text-only marker — that is why a "good model" still looked dumb on diagram
questions.  This module attaches the original page image (cropped to the
answer region when possible) so evaluation can see what the student drew.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_DIAGRAM_HINT_RE = re.compile(
    r"\b(diagram|venn|draw|sketch|figure|graph|plot|label|construct|"
    r"circle|triangle|map|chart|table|shade|illustrat)\b",
    re.IGNORECASE,
)


def _ocr_verification_threshold() -> float:
    """Minimum OCR confidence before text-only marking is considered safe."""

    raw = os.getenv("PCR_OCR_VISION_VERIFY_THRESHOLD", "0.93")
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.93


def _optional_confidence(value: Any) -> Optional[float]:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence < 0.0 or confidence > 1.0:
        return None
    return confidence


def _assignment_method(question_assignment: Any) -> str:
    if not isinstance(question_assignment, dict):
        return ""
    return str(question_assignment.get("method") or "").strip().lower()


def requires_transcription_verification(
    *,
    ocr_confidence: Any = None,
    segmentation_confidence: Any = None,
    question_assignment: Any = None,
) -> bool:
    """Return whether original pixels must verify a scoreable transcription.

    A vision-generated transcription is still only a lossy intermediate.  Low
    confidence text must never be the sole evidence used to deny marks.
    """

    threshold = _ocr_verification_threshold()
    ocr_value = _optional_confidence(ocr_confidence)
    segmentation_value = _optional_confidence(segmentation_confidence)
    method = _assignment_method(question_assignment)
    vision_transcribed = method in {
        "verified_paper_page_order",
        "document_vision_mapping",
        "verified_paper_layout_unresolved",
    }
    if ocr_value is not None and ocr_value < threshold:
        return True
    if segmentation_value is not None and segmentation_value < threshold:
        return True
    # When no independent OCR confidence exists, a model-generated transcript
    # must be checked against the source image before it can affect a score.
    if vision_transcribed and ocr_value is None:
        return True
    return False


def needs_vision_evaluation(
    *,
    content_type: str,
    detected_text: str,
    question_text: str = "",
    has_page_images: bool = True,
    ocr_confidence: Any = None,
    segmentation_confidence: Any = None,
    question_assignment: Any = None,
) -> bool:
    """Decide whether marking must look at the answer image, not only OCR text."""
    if not has_page_images:
        return False

    if requires_transcription_verification(
        ocr_confidence=ocr_confidence,
        segmentation_confidence=segmentation_confidence,
        question_assignment=question_assignment,
    ):
        return True

    ctype = (content_type or "").upper()
    if ctype in {"MIXED", "DIAGRAM_HEAVY", "TABLE_PRESENT"}:
        return True

    text = (detected_text or "").strip()
    words = text.split()
    if _DIAGRAM_HINT_RE.search(question_text or ""):
        return True
    if _DIAGRAM_HINT_RE.search(text):
        return True
    # Extremely thin OCR often means a drawing-only answer (Venn/number boxes).
    # Keep pure short textual answers on the cheap text path.
    if not text or (len(words) <= 3 and len(text) < 24):
        return True
    return False


async def load_answer_page_docs(
    tenant_db: Any,
    submission_id: str,
) -> List[Dict[str, Any]]:
    """Load canonical answer-page artefacts for a submission."""
    if tenant_db is None or not submission_id:
        return []
    try:
        cursor = tenant_db["evalpen_answer_pages"].find(
            {"submission_id": submission_id}
        ).sort("page_number", 1)
        return await cursor.to_list(length=100)
    except Exception:
        logger.exception(
            "Failed to load answer pages for submission %s", submission_id
        )
        return []


async def build_vision_eval_messages(
    *,
    prompt: str,
    response_doc: Dict[str, Any],
    answer_pages: Sequence[Dict[str, Any]],
    question_text: str = "",
) -> Optional[List[Dict[str, Any]]]:
    """Build multimodal chat messages: marking instructions + page evidence."""
    from .ocr_service import _detect_media_type, _resolve_image_base64

    if not answer_pages:
        return None

    source_pages = response_doc.get("source_pages") or []
    page_numbers: List[int] = []
    for ref in source_pages:
        if not isinstance(ref, dict):
            continue
        try:
            page_numbers.append(int(ref.get("page_number") or 0))
        except (TypeError, ValueError):
            continue
    page_numbers = [n for n in page_numbers if n > 0]

    by_number = {
        int(p.get("page_number") or 0): p
        for p in answer_pages
        if int(p.get("page_number") or 0) > 0
    }
    # If region metadata is missing, still attach every page (small answer books).
    target_pages = page_numbers or sorted(by_number.keys())
    if not target_pages:
        return None

    verify_transcription = requires_transcription_verification(
        ocr_confidence=response_doc.get("ocr_confidence"),
        segmentation_confidence=response_doc.get("segmentation_confidence"),
        question_assignment=response_doc.get("question_assignment"),
    )
    question_number = response_doc.get("question_number")
    verification_instruction = ""
    if verify_transcription:
        verification_instruction = (
            " The OCR transcription is untrusted because its evidence confidence "
            "is below the automatic-marking threshold. Locate the exact printed "
            f"question {question_number or ''} using the supplied question text, "
            "read the handwriting from the original full page, and grade the image "
            "when it conflicts with OCR. Never borrow work from a neighbouring "
            "question. If the handwriting cannot be read reliably, set needs_review=true."
        )

    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                prompt
                + "\n\nVISION EVIDENCE:\n"
                "The following image(s) are the student's original handwritten "
                "answer page region(s). Use them as the PRIMARY evidence. OCR "
                "text may be incomplete — diagrams, Venn diagrams, circled "
                "options, tables, and constructions are valid answers even when "
                "OCR missed them. Award marks for what is visibly drawn or written."
                + verification_instruction
            ),
        }
    ]

    attached = 0
    for page_number in target_pages[:4]:
        page = by_number.get(page_number)
        if not page:
            continue
        raw_ref = page.get("raw_image_ref")
        if not isinstance(raw_ref, str) or not raw_ref.strip():
            continue
        expected_sha256 = page.get("asset_sha256")
        image_b64 = (
            await _resolve_image_base64(
                raw_ref,
                expected_sha256=expected_sha256,
            )
            if expected_sha256
            else await _resolve_image_base64(raw_ref)
        )
        if not image_b64:
            continue

        # Prefer a vertical crop to the mapped answer band when we have y range.
        region = next(
            (
                ref
                for ref in source_pages
                if isinstance(ref, dict)
                and int(ref.get("page_number") or 0) == page_number
            ),
            None,
        )
        cropped = _maybe_crop_page_image_b64(
            image_b64,
            page_doc=page,
            region=region if isinstance(region, dict) else None,
        )
        media = _detect_media_type(raw_ref)
        if verify_transcription:
            # The mapped band is itself model-produced and can be displaced.
            # Attach the canonical full page first so the evaluator can locate
            # the printed question independently, then add the crop as a zoom.
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Canonical full student page {page_number}; locate "
                        f"question {question_number or 'from the prompt'}:"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media};base64,{image_b64}",
                        "detail": "high",
                    },
                }
            )
            if cropped:
                content.append(
                    {
                        "type": "text",
                        "text": (
                            "Supplementary mapped-band crop; use only after "
                            "confirming it belongs to the same printed question:"
                        ),
                    }
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            # _maybe_crop_page_image_b64 always emits JPEG.
                            "url": f"data:image/jpeg;base64,{cropped}",
                            "detail": "high",
                        },
                    }
                )
        else:
            payload = cropped or image_b64
            payload_media = "image/jpeg" if cropped else media
            content.append(
                {
                    "type": "text",
                    "text": f"Student answer evidence - page {page_number}:",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{payload_media};base64,{payload}",
                        "detail": "high",
                    },
                }
            )
        attached += 1

    if attached == 0:
        return None

    return [{"role": "user", "content": content}]


def _maybe_crop_page_image_b64(
    image_b64: str,
    *,
    page_doc: Dict[str, Any],
    region: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Crop to the answer band when y_start/y_end are available (mm on A4)."""
    if not region:
        return None
    try:
        y_start = float(region.get("y_start"))
        y_end = float(region.get("y_end"))
    except (TypeError, ValueError):
        return None
    if y_end <= y_start:
        return None

    page_height_mm = float(page_doc.get("page_height_mm") or 297.0)
    if page_height_mm <= 0:
        page_height_mm = 297.0

    try:
        from PIL import Image

        raw = base64.b64decode(image_b64)
        with Image.open(io.BytesIO(raw)) as img:
            img = img.convert("RGB")
            width, height = img.size
            # Expand crop slightly so labels above/below are kept.
            pad_mm = 8.0
            top_mm = max(0.0, y_start - pad_mm)
            bottom_mm = min(page_height_mm, y_end + pad_mm)
            top_px = int((top_mm / page_height_mm) * height)
            bottom_px = int((bottom_mm / page_height_mm) * height)
            # If the band is almost the full page, skip crop.
            if (bottom_px - top_px) >= int(height * 0.85):
                return None
            if bottom_px - top_px < 40:
                return None
            cropped = img.crop((0, max(0, top_px), width, min(height, bottom_px)))
            buffer = io.BytesIO()
            cropped.save(buffer, format="JPEG", quality=88)
            return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        logger.debug("Answer region crop failed; using full page image", exc_info=True)
        return None
