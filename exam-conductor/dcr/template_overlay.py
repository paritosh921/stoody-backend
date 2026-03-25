"""
DCR Template Overlay — stroke rendering, template compositing, and LLM extraction.

Provides utilities for the template-overlay OCR approach:
  1. Render canonical stroke vectors to a rasterised PNG image.
  2. Composite the stroke image onto a template grid.
  3. Build / parse the LLM Vision extraction prompt.

Architecture: DUAL_MODE_ARCHITECTURE.md S4.3 (template overlay variant)
Test IDs: U-DCR-04 (stroke rasterisation), U-DCR-05 (overlay compositing)
Hard constraints:
  - C4: LLM calls happen in the recognizer via the gate; this module only
    builds the prompt / parses the response.
  - No external service imports (PIL/Pillow only).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_DPI = 150
_STROKE_COLOUR = (0, 0, 0, 255)        # opaque black
_STROKE_WIDTH_PX = 2                    # default line width in pixels
_BACKGROUND_COLOUR = (255, 255, 255, 0) # transparent white (for compositing)


# ---------------------------------------------------------------------------
# 1. Render canonical strokes to a PNG image
# ---------------------------------------------------------------------------

def render_strokes_to_image(
    raw_strokes: List[Dict[str, Any]],
    page_width_mm: float,
    page_height_mm: float,
    dpi: int = _DEFAULT_DPI,
) -> bytes:
    """
    Rasterise canonical stroke vectors onto a transparent-background PNG.

    Parameters
    ----------
    raw_strokes
        List of stroke dicts.  Each stroke has a ``points`` list where every
        point is ``[x, y, pressure, tiltX, tiltY, timestamp]`` with *x* and
        *y* in page-space **millimetres**.
    page_width_mm, page_height_mm
        Page dimensions in millimetres (used to determine output image size).
    dpi
        Rendering resolution.  150 DPI is a good balance between clarity and
        transfer size for LLM vision input.

    Returns
    -------
    bytes
        PNG-encoded image bytes.
    """
    # mm -> px conversion factor
    mm_to_px = dpi / 25.4

    width_px = max(1, int(page_width_mm * mm_to_px))
    height_px = max(1, int(page_height_mm * mm_to_px))

    img = Image.new("RGBA", (width_px, height_px), _BACKGROUND_COLOUR)
    draw = ImageDraw.Draw(img)

    for stroke in raw_strokes:
        points = stroke.get("points")
        if not points or len(points) < 2:
            # Single-point strokes are drawn as a small dot.
            if points and len(points) == 1:
                pt = points[0]
                px_x = pt[0] * mm_to_px
                px_y = pt[1] * mm_to_px
                r = max(1, _STROKE_WIDTH_PX // 2)
                draw.ellipse(
                    [px_x - r, px_y - r, px_x + r, px_y + r],
                    fill=_STROKE_COLOUR,
                )
            continue

        # Determine stroke width from average pressure (if available).
        # Pressure is the 3rd element (index 2) in each point, normalised
        # in [0, 1].  We scale linearly: 1px at 0 pressure, 4px at 1.0.
        pressures = [
            pt[2] for pt in points
            if len(pt) > 2 and isinstance(pt[2], (int, float))
        ]
        avg_pressure = sum(pressures) / len(pressures) if pressures else 0.5
        stroke_w = max(1, int(1 + 3 * avg_pressure))

        # Convert mm coordinates to pixels and draw polyline.
        px_coords = [
            (pt[0] * mm_to_px, pt[1] * mm_to_px)
            for pt in points
            if len(pt) >= 2
        ]
        if len(px_coords) >= 2:
            draw.line(px_coords, fill=_STROKE_COLOUR, width=stroke_w, joint="curve")

    # Encode to PNG bytes.
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 2. Overlay strokes on a template image
# ---------------------------------------------------------------------------

def overlay_on_template(
    stroke_image: bytes,
    template_image: bytes,
) -> bytes:
    """
    Composite the rendered stroke image on top of the template grid.

    The template is used as the background (opaque).  The stroke image is
    alpha-composited on top so that the answer-sheet grid lines remain
    visible beneath the student's handwriting.

    If the two images differ in size the stroke image is resized (with
    bilinear interpolation) to match the template dimensions.

    Parameters
    ----------
    stroke_image
        PNG bytes of rendered strokes (transparent background).
    template_image
        PNG bytes of the answer-sheet template (opaque background).

    Returns
    -------
    bytes
        Merged PNG bytes.
    """
    template = Image.open(io.BytesIO(template_image)).convert("RGBA")
    strokes = Image.open(io.BytesIO(stroke_image)).convert("RGBA")

    # Resize strokes to match template if dimensions differ.
    if strokes.size != template.size:
        strokes = strokes.resize(template.size, Image.BILINEAR)

    # Alpha composite: template as base, strokes on top.
    merged = Image.alpha_composite(template, strokes)

    buf = io.BytesIO()
    merged.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 3. Build the LLM extraction prompt
# ---------------------------------------------------------------------------

def build_dcr_extraction_prompt(
    question_count: int,
    question_ids: List[str],
) -> str:
    """
    Build the system/user prompt that instructs the LLM to extract student
    responses from the overlaid answer-sheet image.

    Parameters
    ----------
    question_count
        Total number of questions on the page.
    question_ids
        Ordered list of question identifiers (e.g. ``["Q1", "Q2", ...]``).

    Returns
    -------
    str
        The prompt text.
    """
    id_list = ", ".join(question_ids)
    return (
        "You are an expert OCR system for handwritten answer sheets. "
        "The attached image shows a student's handwritten responses overlaid "
        "on the answer template grid.\n\n"
        f"There are {question_count} questions on this page with IDs: {id_list}.\n\n"
        "For each question, identify the student's written response. "
        "Responses are typically single letters (A, B, C, D), short words, "
        "numbers, or brief phrases.\n\n"
        "Return your answer as a JSON object mapping each question ID to the "
        "recognized student response. If a question has no visible response, "
        'use an empty string "".\n\n'
        "Example output format:\n"
        "```json\n"
        "{\n"
        + "".join(f'  "{qid}": "",\n' for qid in question_ids[:3])
        + "  ...\n"
        "}\n"
        "```\n\n"
        "Return ONLY the JSON object, no other text."
    )


# ---------------------------------------------------------------------------
# 4. Parse LLM extraction response
# ---------------------------------------------------------------------------

def parse_dcr_extraction_response(
    llm_content: str,
    question_ids: List[str],
) -> Dict[str, str]:
    """
    Parse the LLM's JSON response into ``{question_id: recognized_text}``.

    Handles common LLM output quirks:
      - Markdown code fences (`` ```json ... ``` ``)
      - Trailing commas
      - Partial / truncated JSON
      - Whitespace padding

    Parameters
    ----------
    llm_content
        Raw text content from the LLM gate response.
    question_ids
        Expected question IDs (used as fallback keys).

    Returns
    -------
    dict[str, str]
        Mapping of question_id to recognized student response text.
        Missing questions default to empty string.
    """
    text = llm_content.strip()

    # Strip markdown code fences.
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Remove trailing commas before closing braces (common LLM quirk).
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    parsed: Dict[str, str] = {}
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            parsed = {str(k): str(v) for k, v in data.items()}
    except json.JSONDecodeError:
        # Attempt line-by-line key:value extraction as last resort.
        logger.warning(
            "DCR extraction response was not valid JSON; attempting "
            "line-by-line fallback parsing."
        )
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            # Match patterns like "Q1": "A" or Q1: A
            m = re.match(
                r'"?([^":\s]+)"?\s*:\s*"?([^",}]*)"?',
                line,
            )
            if m:
                parsed[m.group(1).strip()] = m.group(2).strip()

    # Ensure every expected question_id is present (default empty).
    result: Dict[str, str] = {}
    for qid in question_ids:
        result[qid] = parsed.get(qid, "")
    return result


# ---------------------------------------------------------------------------
# 5. Helper — encode PNG bytes as base64 data URI for multimodal messages
# ---------------------------------------------------------------------------

def png_bytes_to_data_uri(png_bytes: bytes) -> str:
    """
    Encode PNG bytes into a ``data:image/png;base64,...`` URI suitable for
    OpenAI-style multimodal message content.
    """
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_vision_messages(
    prompt_text: str,
    image_bytes: bytes,
) -> List[Dict[str, Any]]:
    """
    Build the OpenAI-compatible multimodal messages array for a vision call.

    The provider layer (and Gemini translation helper) expects this format.
    The gate forwards the ``messages`` list as-is to the selected provider.

    Returns
    -------
    list[dict]
        A single-user-message array containing the text prompt and the
        base64-encoded image.
    """
    data_uri = png_bytes_to_data_uri(image_bytes)
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                },
            ],
        }
    ]
