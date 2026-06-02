"""
Async PDF Processing API for SkillBot
PDF upload and OCR processing endpoints with Mistral AI OCR (primary) and GPT Vision (fallback)
"""

import logging
import base64
import asyncio
import uuid
import os
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

# Suppress verbose aiohttp logging
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
logging.getLogger("aiohttp.client").setLevel(logging.WARNING)
logging.getLogger("aiohttp.server").setLevel(logging.WARNING)

from fastapi import APIRouter, Request, HTTPException, Depends, status, UploadFile, File, Form, Query, Body
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
import aiofiles
from bson import ObjectId as BsonObjectId

from core.database import DatabaseManager
from core.cache import CacheManager
from core.observability import observe_ocr_job
from api.v1.auth_async import get_current_user, get_database, get_cache
from api.v1.student_async import require_student, require_student_or_admin
from config_async import OCR_TIMEOUT_SECONDS
from utils.path_utils import get_relative_path, get_absolute_path
from utils.s3_storage import upload_file as s3_upload_file, is_s3_enabled, get_public_url, download_file
from services.ai_gateway_service import (
    AIGatewayService,
    AIUsageLimitExceeded,
    estimate_ocr_tokens,
    estimate_text_tokens,
)
from services.answer_question_mapping_service import AnswerQuestionMappingService
from services.extraction_validator import ExtractionValidator
from services.layout_preflight_service import LayoutPreflightService
from services.option_layout_normalizer import OptionLayoutNormalizer
from services.region_crop_service import RegionCropService

logger = logging.getLogger(__name__)

router = APIRouter()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Mistral AI OCR configuration (primary)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_OCR_MODEL = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")

# Groq API configuration (primary for question extraction — fast + cheap)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Sarvam AI Document Intelligence configuration (disabled — kept for reference)
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

# OpenAI GPT — used as fallback for question extraction if Groq unavailable,
# and as primary for GPT Vision OCR fallback
OCR_FALLBACK_MODEL = os.getenv("OCR_FALLBACK_MODEL", "gpt-5-mini")

# IMPORTANT: Grade/Standard matching uses EXACT string matching
# Both student.grade and document.standard should come from the same admin settings
# (configured in School Settings), so they will match exactly
# For example: student.grade="12th Pass" matches document.standard="12th Pass"
# No normalization or fuzzy matching is needed or desired

# Pydantic models
class OCRImage(BaseModel):
    id: str
    top_left_x: int
    top_left_y: int
    bottom_right_x: int
    bottom_right_y: int
    image_base64: Optional[str] = None

class OCRPage(BaseModel):
    index: int
    markdown: str
    images: List[OCRImage]
    dimensions: Dict[str, Any]

class ExtractedQuestion(BaseModel):
    id: str
    text: str
    options: List[str] = []
    correct_answer: Optional[str] = None
    images: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    points: Optional[float] = 4.0  # Default 4 points for Test Series (JEE style)
    penalty: Optional[float] = 1.0  # Default 1 penalty (JEE style)

class PDFProcessingResult(BaseModel):
    job_id: str
    status: str  # 'processing', 'completed', 'error'
    progress: int
    extracted_questions: int = 0
    extracted_images: int = 0
    output_folder: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime
    pages: Optional[List[OCRPage]] = None

class AnswerSheetOCRRequest(BaseModel):
    documentAnchorText: Optional[str] = None

class QuestionImage(BaseModel):
    id: str
    filename: str
    path: str
    description: str
    type: str
    base64_data: Optional[str] = None
    bbox: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}

class Question(BaseModel):
    id: str
    text: str
    subject: str
    difficulty: str
    extracted_at: datetime
    pdf_source: str
    images: List[QuestionImage] = []
    options: List[str] = []
    correct_answer: Optional[str] = None
    metadata: Dict[str, Any] = {}
    points: Optional[float] = 4.0
    penalty: Optional[float] = 1.0

def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require admin access (regular or B2C)"""
    if current_user.get("user_type") not in ["admin", "b2c_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

def require_admin_or_tutor(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Allow admin, B2C admin, and tutor roles"""
    if current_user.get("user_type") not in ["admin", "b2c_admin", "tutor"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or Tutor access required"
        )
    return current_user

def is_b2c_admin(current_user: Dict[str, Any]) -> bool:
    """Check if the current user is a B2C admin"""
    return current_user.get("user_type") == "b2c_admin"


def _serialize_answer_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Return the admin-safe fields needed to display a worked-answer mapping."""
    return {
        "mapping_id": str(mapping.get("mapping_id") or ""),
        "question_id": str(mapping.get("question_id") or mapping.get("question_region_id") or ""),
        "question_region_id": str(mapping.get("question_region_id") or mapping.get("question_id") or ""),
        "answer_region_id": str(mapping.get("answer_region_id") or ""),
        "answer_text": mapping.get("answer_text") or "",
        "mapping_strategy": mapping.get("mapping_strategy") or "",
        "confidence": mapping.get("confidence"),
        "manual_review_required": bool(mapping.get("manual_review_required")),
        "source": mapping.get("source") or "answer_sheet",
        "correct_option_verified": mapping.get("correct_option_verified"),
        "generation_notes": mapping.get("generation_notes") or "",
    }


def _build_ai_gateway_context(
    *,
    current_user: Dict[str, Any],
    db: DatabaseManager,
    document_id: Optional[str] = None,
    region_id: Optional[str] = None,
    region_scope: Optional[str] = None,
    is_b2c: Optional[bool] = None,
) -> Dict[str, Any]:
    return {
        "db": db,
        "is_b2c": is_b2c_admin(current_user) if is_b2c is None else is_b2c,
        "user_id": current_user.get("user_id") or current_user.get("_id"),
        "tenant_id": current_user.get("tenant_id") or current_user.get("db_name") or current_user.get("institution_id"),
        "document_id": document_id,
        "region_id": region_id,
        "region_scope": region_scope,
    }

async def delete_existing_ocr_outputs(
    *,
    document: Dict[str, Any],
    current_user: Dict[str, Any],
    db: DatabaseManager,
    question_ids: Optional[List[str]] = None,
    delete_images: bool = True,
) -> Dict[str, Any]:
    """Delete previously extracted OCR questions/images for a document."""
    document_id = document.get("document_id")
    filename = document.get("filename", "")
    is_b2c = is_b2c_admin(current_user)
    question_filter: Dict[str, Any] = {"document_id": document_id}

    if question_ids:
        question_filter["id"] = {"$in": question_ids}

    if is_b2c:
        questions_deleted = await db.b2c_delete_many("questions", question_filter)
        images_deleted = (
            await db.b2c_delete_many("images", {"source_pdf": filename})
            if delete_images and filename
            else False
        )
    else:
        questions_deleted = await db.mongo_delete_many("questions", question_filter)
        images_deleted = (
            await db.mongo_delete_many("images", {"source_pdf": filename})
            if delete_images and filename
            else 0
        )

    logger.info(
        "Deleted existing OCR outputs for %s: questions=%s, images=%s",
        document_id,
        questions_deleted,
        images_deleted,
    )
    return {
        "questions_deleted": questions_deleted,
        "images_deleted": images_deleted,
    }

async def call_gpt_vision_ocr(file_content: bytes) -> Dict[str, Any]:
    """
    OCR using GPT Vision. Renders PDF pages and extracts text via GPT.
    All pages are processed IN PARALLEL for speed.
    Figures are detected deterministically with OpenCV after OCR.
    """
    import fitz  # PyMuPDF
    from openai import AsyncOpenAI

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        raise Exception("OPENAI_API_KEY is not configured — cannot use GPT Vision fallback")

    client = AsyncOpenAI(api_key=openai_key)
    print(f"[GPT-OCR] Starting OCR (PDF size: {len(file_content)} bytes)", flush=True)

    doc = fitz.open(stream=file_content, filetype="pdf")
    total_pages = len(doc)
    print(f"[GPT-OCR] PDF has {total_pages} page(s) — processing all in parallel", flush=True)

    # Step 1: Render all pages to images (CPU-bound, fast)
    page_renders = []
    for page_idx in range(total_pages):
        page = doc[page_idx]
        mat = fitz.Matrix(200 / 72, 200 / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        page_b64 = base64.b64encode(img_bytes).decode("utf-8")
        page_renders.append({
            "index": page_idx,
            "b64": page_b64,
            "width": pix.width,
            "height": pix.height,
        })
    doc.close()
    print(f"[GPT-OCR] Rendered {total_pages} pages, sending to GPT in parallel...", flush=True)

    ocr_prompt = (
        "Extract ALL text from this exam paper page as clean markdown.\n"
        "Rules:\n"
        "- Keep question numbers with a dot: 1. 2. 3. and on the SAME line as question text\n"
        "- Keep option labels exactly: (a), (b), a), b), A., B. etc.\n"
        "- Preserve LaTeX math: $...$ inline, $$...$$ display\n"
        "- Preserve Hindi/regional text as-is\n"
        "- For diagrams/figures, write: [FIGURE]\n"
        "- Output ONLY the extracted text, no explanations"
    )

    # Step 2: OCR all pages in parallel
    async def ocr_single_page(pr: Dict) -> Dict:
        page_idx = pr["index"]
        try:
            response = await client.chat.completions.create(
                model=OCR_FALLBACK_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ocr_prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{pr['b64']}",
                            "detail": "high"
                        }}
                    ]
                }],
                max_completion_tokens=4096,
            )
            md = response.choices[0].message.content or ""
            print(f"[GPT-OCR] Page {page_idx + 1}: {len(md)} chars", flush=True)
            return {"index": page_idx, "markdown": md, "b64": pr["b64"], "w": pr["width"], "h": pr["height"]}
        except Exception as e:
            print(f"[GPT-OCR] Page {page_idx + 1} failed: {e}", flush=True)
            return {"index": page_idx, "markdown": "", "b64": pr["b64"], "w": pr["width"], "h": pr["height"]}

    ocr_results = await asyncio.gather(*[ocr_single_page(pr) for pr in page_renders])
    ocr_results.sort(key=lambda r: r["index"])  # maintain page order

    # Build final result
    pages_result = []
    for r in ocr_results:
        pages_result.append({
            "index": r["index"],
            "markdown": r["markdown"],
            "images": [],
            "page_render": r["b64"],
            "dimensions": {"dpi": 200, "width": r["w"], "height": r["h"]}
        })

    print(f"[GPT-OCR] Done! {total_pages} pages OCR'd (figures detected with OpenCV later)", flush=True)
    return {"pages": pages_result}


async def call_gpt_vision_ocr_validation_fallback(
    file_content: bytes,
    *,
    gateway_context: Optional[Dict[str, Any]],
    fallback_reason: str,
) -> Dict[str, Any]:
    async def _raw_call():
        return await call_gpt_vision_ocr(file_content)

    if gateway_context:
        gateway = AIGatewayService(
            gateway_context.get("db"),
            is_b2c=bool(gateway_context.get("is_b2c")),
        )
        result = await gateway.call(
            user_id=str(gateway_context.get("user_id") or "unknown"),
            tenant_id=gateway_context.get("tenant_id"),
            document_id=gateway_context.get("document_id"),
            region_id=gateway_context.get("region_id"),
            region_scope=gateway_context.get("region_scope"),
            stage="ocr_fallback_validation",
            provider="openai",
            model=OCR_FALLBACK_MODEL,
            input_kind="pdf_region",
            estimated_input_tokens=estimate_ocr_tokens(pdf_bytes=len(file_content), page_count=1),
            estimated_output_tokens=2048,
            input_units={"pdf_bytes": len(file_content), "page_count": 1},
            call_fn=_raw_call,
        )
    else:
        result = await _raw_call()
    result["_ocr_provider"] = "openai"
    result["_ocr_model"] = OCR_FALLBACK_MODEL
    result["_fallback_reason"] = fallback_reason
    return result


async def call_mistral_ocr(file_content: bytes) -> Dict[str, Any]:
    """
    OCR using Mistral AI's dedicated OCR endpoint.
    Direct API call — no polling, returns immediately.

    Returns the same structure for downstream compatibility:
    {
        "pages": [
            {
                "index": 0,
                "markdown": "extracted text...",
                "images": [{"id": "img-0-0", "image_base64": "...", ...}],
                "dimensions": {"dpi": 200, "width": 1700, "height": 2200}
            }
        ]
    }
    """
    import time as _time
    from mistralai.client import Mistral

    if not MISTRAL_API_KEY:
        raise Exception("MISTRAL_API_KEY is not configured")

    client = Mistral(api_key=MISTRAL_API_KEY)
    pdf_b64 = base64.b64encode(file_content).decode("utf-8")

    print(f"[MISTRAL-OCR] Starting OCR (PDF size: {len(file_content)} bytes, model: {MISTRAL_OCR_MODEL})", flush=True)
    t0 = _time.monotonic()

    # Mistral OCR is synchronous SDK — run in thread pool to not block event loop
    def _run_mistral_sync() -> Any:
        return client.ocr.process(
            model=MISTRAL_OCR_MODEL,
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{pdf_b64}",
            },
            include_image_base64=True,
        )

    ocr_response = await asyncio.get_event_loop().run_in_executor(None, _run_mistral_sync)

    t1 = _time.monotonic()

    # Convert Mistral response to our standard format
    pages_result: List[Dict[str, Any]] = []
    total_images = 0

    for page in ocr_response.pages:
        page_idx = page.index
        markdown = page.markdown or ""

        # Extract images — Mistral returns them in page.images
        page_images: List[Dict[str, Any]] = []
        if hasattr(page, "images") and page.images:
            for i, img in enumerate(page.images):
                img_id = f"img-{page_idx}-{i}"
                img_b64 = None
                # Mistral image objects have image_base64 attribute
                if hasattr(img, "image_base64") and img.image_base64:
                    img_b64 = img.image_base64
                    # Replace the markdown image placeholder with our img_id
                    # Mistral uses ![img-N.jpeg](img-N.jpeg) style
                    if hasattr(img, "id") and img.id:
                        markdown = markdown.replace(f"![{img.id}]({img.id})", f"![{img_id}]({img_id})")

                page_images.append({
                    "id": img_id,
                    "image_base64": img_b64,
                    "top_left_x": 0,
                    "top_left_y": 0,
                    "bottom_right_x": 0,
                    "bottom_right_y": 0,
                })
                total_images += 1

        # Page dimensions
        dims = {"dpi": 200, "width": 0, "height": 0}
        if hasattr(page, "dimensions") and page.dimensions:
            dims = {
                "dpi": getattr(page.dimensions, "dpi", 200),
                "width": getattr(page.dimensions, "width", 0),
                "height": getattr(page.dimensions, "height", 0),
            }

        pages_result.append({
            "index": page_idx,
            "markdown": markdown,
            "images": page_images,
            "dimensions": dims,
        })

    print(
        f"[MISTRAL-OCR] Done! {len(pages_result)} pages, {total_images} images "
        f"({t1 - t0:.1f}s)",
        flush=True,
    )
    return {"pages": pages_result}


def _extract_pdf_images_with_positions(file_content: bytes) -> Dict[int, Dict[str, Any]]:
    """
    Deterministic image + question-position extraction using PyMuPDF.

    For every page in the PDF this returns:
      - Each embedded raster image with its bounding box (in PDF point coords)
        and a base64-encoded PNG payload.
      - The y-positions of every numbered question text block on the page,
        detected by matching `<n>.` / `<n>)` markers at the start of each
        text block via PyMuPDF's `get_text("blocks")`.

    The downstream pipeline uses these to:
      1. Populate `ocr_result["pages"][i]["images"]` when Mistral OCR returns
         an empty list (which it does for many Word/Pages-generated PDFs even
         when the PDF has real embedded images).
      2. Match each image to the question whose y-range sits immediately above
         it — fully positional, no LLM call.

    Returns a dict keyed by 0-based page index. Failures are non-fatal: any
    pages that fail individually are simply omitted from the result.
    """
    import fitz  # PyMuPDF
    import re as _re
    import io as _io
    from PIL import Image as _PILImage

    Q_NUM_RE = _re.compile(r"^\s*(\d{1,3})\s*[\.\)]\s+")
    # Per-image minimum bounding-box dimensions (in PDF points; 1pt ≈ 1/72").
    # 30pt ≈ 0.42 inch — anything smaller is almost certainly a decoration,
    # bullet-icon, or alpha mask, not a real diagram.
    MIN_IMG_DIM_PT = 30.0

    def _render_image_xref_to_png(doc_obj, xref: int, smask_xref: int) -> bytes:
        """
        Render an embedded image XObject to a PNG byte string with its soft
        mask correctly applied and any transparency composited over a white
        background.

        Why this exists: `fitz.Pixmap(doc, xref)` returns the raw image
        pixels WITHOUT applying the document's soft mask. Many PDFs (e.g.
        Word-generated chemistry diagrams with translucent backgrounds) store
        the visible content in the soft mask, so naively dropping alpha
        produces a fully-black image. We:
          1. Strip the misleading alpha=1 attribute from the colour pixmap.
          2. Combine it with the soft mask via Pixmap(color, mask) which
             produces a true RGBA pixmap with the smask in the alpha channel.
          3. Composite the RGBA result over white via PIL so the saved PNG
             renders identically against any UI background colour.
        """
        color_pix = fitz.Pixmap(doc_obj, xref)
        # Normalize to RGB (drop CMYK / DeviceN / Lab / etc.).
        if color_pix.colorspace is None or color_pix.colorspace.n not in (1, 3):
            color_pix = fitz.Pixmap(fitz.csRGB, color_pix)
        # `Pixmap(color, mask)` requires a colour pixmap WITHOUT alpha.
        if color_pix.alpha:
            color_pix = fitz.Pixmap(color_pix, 0)  # 0 = strip alpha
        # If the image has a soft mask, fold it in. The result has real alpha.
        if smask_xref and smask_xref > 0:
            try:
                mask_pix = fitz.Pixmap(doc_obj, smask_xref)
                color_pix = fitz.Pixmap(color_pix, mask_pix)
            except Exception as mask_err:
                logger.debug(f"[PYMUPDF] smask {smask_xref} merge failed: {mask_err}")
        png_bytes = color_pix.tobytes("png")
        # If the result has alpha, composite over white so transparent regions
        # render as white instead of leaking the UI background colour.
        if color_pix.alpha:
            try:
                pil = _PILImage.open(_io.BytesIO(png_bytes))
                if pil.mode == "RGBA":
                    bg = _PILImage.new("RGB", pil.size, (255, 255, 255))
                    bg.paste(pil, mask=pil.split()[3])
                    out_buf = _io.BytesIO()
                    bg.save(out_buf, format="PNG")
                    png_bytes = out_buf.getvalue()
            except Exception as pil_err:
                logger.debug(f"[PYMUPDF] PIL composite failed for xref {xref}: {pil_err}")
        return png_bytes

    result: Dict[int, Dict[str, Any]] = {}

    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
    except Exception as e:
        logger.warning(f"[PYMUPDF] Failed to open PDF: {e}")
        return result

    try:
        for page_idx, page in enumerate(doc):
            page_w = float(page.rect.width)
            page_h = float(page.rect.height)
            page_data: Dict[str, Any] = {
                "page_width": page_w,
                "page_height": page_h,
                "images": [],
                "question_blocks": [],
                # All text blocks on the page (used for substring search to
                # locate LLM-extracted questions whose markers PyMuPDF couldn't
                # detect — case studies, sub-numbered questions, etc.)
                "text_blocks": [],
            }

            # ---------- Embedded raster images ----------
            try:
                image_infos = page.get_image_info(xrefs=True)
            except Exception as e:
                logger.warning(f"[PYMUPDF] Page {page_idx}: get_image_info failed: {e}")
                image_infos = []

            # Build xref → smask_xref lookup from get_images() since
            # get_image_info() does not return smask information.
            try:
                smask_lookup: Dict[int, int] = {
                    rec[0]: rec[1] for rec in page.get_images(full=True) if len(rec) >= 2
                }
            except Exception:
                smask_lookup = {}

            raw_images: List[Dict[str, Any]] = []
            for info in image_infos:
                xref = info.get("xref", 0)
                if not xref:
                    continue
                bbox = info.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                w_pt = float(bbox[2]) - float(bbox[0])
                h_pt = float(bbox[3]) - float(bbox[1])
                if w_pt < MIN_IMG_DIM_PT or h_pt < MIN_IMG_DIM_PT:
                    continue
                try:
                    img_bytes = _render_image_xref_to_png(
                        doc, xref, smask_lookup.get(xref, 0)
                    )
                    img_b64 = base64.b64encode(img_bytes).decode("ascii")
                    raw_images.append({
                        "bbox": (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                        "b64": img_b64,
                    })
                except Exception as e:
                    logger.warning(f"[PYMUPDF] Page {page_idx} xref {xref}: extraction failed: {e}")

            # Sort images top-to-bottom by y of top-left corner so the cursor
            # fallback in extract_questions_with_gpt assigns them in reading order.
            raw_images.sort(key=lambda r: r["bbox"][1])

            for img_idx, raw in enumerate(raw_images):
                bx0, by0, bx1, by1 = raw["bbox"]
                # Pydantic OCRImage requires int coordinates — round and cast.
                # Sub-pixel precision is irrelevant for question matching since
                # we compare against question-block y-positions that are also
                # rounded to whole points by the PDF layout engine.
                page_data["images"].append({
                    "id": f"img-{page_idx}-{img_idx}",
                    "image_base64": raw["b64"],
                    "top_left_x": int(round(bx0)),
                    "top_left_y": int(round(by0)),
                    "bottom_right_x": int(round(bx1)),
                    "bottom_right_y": int(round(by1)),
                })

            # ---------- Question text-block positions ----------
            try:
                blocks = page.get_text("blocks")
            except Exception:
                blocks = []

            seen_numbers: set = set()
            for block in blocks:
                # block tuple: (x0, y0, x1, y1, text, block_no, block_type)
                if len(block) < 5:
                    continue
                _x0, y0, _x1, y1, text = block[0], block[1], block[2], block[3], block[4]
                if not isinstance(text, str):
                    continue
                # Stash every text block (normalized) for downstream substring
                # search. We collapse whitespace so search snippets aren't
                # tripped up by line wraps or stray spaces.
                normalized = " ".join(text.split())
                if normalized:
                    page_data["text_blocks"].append({
                        "y_start": float(y0),
                        "y_end": float(y1),
                        "text": normalized,
                    })
                m = Q_NUM_RE.match(text)
                if not m:
                    continue
                try:
                    qnum = int(m.group(1))
                except (ValueError, TypeError):
                    continue
                # Sanity bounds — exam papers don't have 4-digit question numbers
                # and number 0 is invalid. The regex already rejects "1 mark" /
                # bare digits because it requires a `.` or `)` after the number.
                if qnum < 1 or qnum > 999:
                    continue
                if qnum in seen_numbers:
                    continue
                seen_numbers.add(qnum)
                page_data["question_blocks"].append({
                    "number": qnum,
                    "y_start": float(y0),
                    "y_end": float(y1),
                })

            page_data["question_blocks"].sort(key=lambda b: b["y_start"])
            result[page_idx] = page_data
    finally:
        doc.close()

    total_imgs = sum(len(p["images"]) for p in result.values())
    total_qblocks = sum(len(p["question_blocks"]) for p in result.values())
    print(
        f"[PYMUPDF] Extracted {total_imgs} images and {total_qblocks} question blocks "
        f"from {len(result)} pages",
        flush=True,
    )
    return result


def _augment_ocr_with_pymupdf(
    ocr_result: Dict[str, Any],
    file_content: bytes,
) -> None:
    """
    Mutate `ocr_result` in place: for any page where the upstream OCR provider
    (Mistral) returned no images, populate `images` from PyMuPDF and stash the
    detected question-block y-positions on the page dict so positional matching
    can use them later.

    Pages where the OCR provider DID return images are left untouched — we
    trust the provider when it gives us something.
    """
    try:
        pymupdf_pages = _extract_pdf_images_with_positions(file_content)
    except Exception as e:
        logger.warning(f"[PYMUPDF] Augmentation failed: {e}")
        print(f"[PYMUPDF] Augmentation failed: {e}", flush=True)
        return

    if not pymupdf_pages:
        return

    pages = ocr_result.get("pages", [])
    injected = 0
    for page in pages:
        pidx = page.get("index", 0)
        pmu_page = pymupdf_pages.get(pidx)
        if not pmu_page:
            continue
        # Always stash the question-block positions and full text blocks for
        # the matching pass — they're useful even if Mistral DID return images,
        # because they let us verify the position-based assignment later.
        page["_pymupdf_question_blocks"] = pmu_page["question_blocks"]
        page["_pymupdf_text_blocks"] = pmu_page["text_blocks"]
        page["_pymupdf_page_width"] = pmu_page["page_width"]
        page["_pymupdf_page_height"] = pmu_page["page_height"]
        # Only inject images if the OCR provider returned nothing.
        if not page.get("images") and pmu_page["images"]:
            page["images"] = pmu_page["images"]
            injected += len(pmu_page["images"])
    if injected:
        print(
            f"[PYMUPDF] Injected {injected} embedded images into OCR result "
            f"(Mistral returned 0 for those pages)",
            flush=True,
        )


async def call_sarvam_ocr(
    file_content: bytes,
    *,
    gateway_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Primary OCR entry point for the pipeline.
    Uses Mistral OCR as primary, GPT Vision as fallback.

    Returns:
    {
        "pages": [
            {
                "index": 0,
                "markdown": "extracted text...",
                "images": [{"id": "img-0-0", "image_base64": "...", ...}],
                "dimensions": {"dpi": 200, "width": 1700, "height": 2200}
            }
        ]
    }
    """
    async def _gateway_call(stage: str, provider: str, model: str, call_fn):
        if not gateway_context:
            return await call_fn()
        gateway = AIGatewayService(
            gateway_context.get("db"),
            is_b2c=bool(gateway_context.get("is_b2c")),
        )
        return await gateway.call(
            user_id=str(gateway_context.get("user_id") or "unknown"),
            tenant_id=gateway_context.get("tenant_id"),
            document_id=gateway_context.get("document_id"),
            region_id=gateway_context.get("region_id"),
            region_scope=gateway_context.get("region_scope"),
            stage=stage,
            provider=provider,
            model=model,
            input_kind="pdf_region",
            estimated_input_tokens=estimate_ocr_tokens(pdf_bytes=len(file_content), page_count=1),
            estimated_output_tokens=2048,
            input_units={"pdf_bytes": len(file_content), "page_count": 1},
            call_fn=call_fn,
        )

    # --- Mistral OCR primary, GPT Vision fallback ---
    fallback_reason = None
    if MISTRAL_API_KEY:
        try:
            result = await _gateway_call(
                "ocr_primary",
                "mistral",
                MISTRAL_OCR_MODEL,
                lambda: call_mistral_ocr(file_content),
            )
            result["_ocr_provider"] = "mistral"
            result["_ocr_model"] = MISTRAL_OCR_MODEL
            print("[OCR] Provider: Mistral AI", flush=True)
            return result
        except AIUsageLimitExceeded as limit_err:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=limit_err.payload,
            )
        except Exception as mistral_err:
            print(f"[OCR] Mistral OCR failed ({type(mistral_err).__name__}: {mistral_err}), falling back to GPT Vision...", flush=True)
            logger.warning(f"Mistral OCR failed: {mistral_err}")
            fallback_reason = f"mistral_error:{type(mistral_err).__name__}"
    else:
        print("[OCR] MISTRAL_API_KEY not set, skipping Mistral...", flush=True)
        fallback_reason = "mistral_api_key_missing"

    # GPT Vision fallback
    try:
        result = await _gateway_call(
            "ocr_fallback",
            "openai",
            OCR_FALLBACK_MODEL,
            lambda: call_gpt_vision_ocr(file_content),
        )
        result["_ocr_provider"] = "openai"
        result["_ocr_model"] = OCR_FALLBACK_MODEL
        result["_fallback_reason"] = fallback_reason or "primary_unavailable"
        print("[OCR] Provider: GPT Vision (fallback)", flush=True)
        return result
    except AIUsageLimitExceeded as limit_err:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=limit_err.payload,
        )
    except Exception as gpt_err:
        logger.error(f"GPT Vision OCR also failed: {gpt_err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"All OCR providers failed. GPT Vision: {gpt_err}"
        )

    # --- Sarvam AI (disabled — kept for reference) ---
    # if not SARVAM_API_KEY:
    #     logger.warning("SARVAM_API_KEY is not configured, skipping Sarvam — trying GPT Vision directly")
    #     print("[OCR] SARVAM_API_KEY not set, using GPT Vision directly...", flush=True)
    #     try:
    #         result = await call_gpt_vision_ocr(file_content)
    #         print("[OCR] Provider: GPT Vision (no Sarvam key)", flush=True)
    #         return result
    #     except Exception as gpt_err:
    #         logger.error(f"GPT Vision OCR failed (no Sarvam key): {gpt_err}")
    #         raise HTTPException(
    #             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    #             detail=f"Sarvam API key not configured and GPT Vision fallback failed: {gpt_err}"
    #         )
    #
    # # Sarvam can take 3-5 minutes for large PDFs — use generous timeout
    # sarvam_timeout = max(OCR_TIMEOUT_SECONDS, 600)  # At least 10 minutes
    #
    # def _run_sarvam_sync() -> Dict[str, Any]:
    #     """Run synchronous Sarvam SDK calls (executed in thread pool)."""
    #     import tempfile
    #     import zipfile
    #     import time as _time
    #     from sarvamai import SarvamAI
    #
    #     print(f"[SARVAM] Starting OCR (PDF size: {len(file_content)} bytes)", flush=True)
    #     logger.info(f"Calling Sarvam AI Document Intelligence (PDF size: {len(file_content)} bytes)")
    #
    #     client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    #     start_time = _time.time()
    #
    #     # Save PDF to temp file (Sarvam SDK needs a file path)
    #     tmp_pdf_fd, tmp_pdf_path = tempfile.mkstemp(suffix=".pdf")
    #     try:
    #         with os.fdopen(tmp_pdf_fd, "wb") as tmp_pdf:
    #             tmp_pdf.write(file_content)
    #
    #         # Create document intelligence job
    #         print("[SARVAM] Creating job...", flush=True)
    #         job = client.document_intelligence.create_job(
    #             language="en-IN",
    #             output_format="html"
    #         )
    #         print(f"[SARVAM] Job created: {job.job_id}", flush=True)
    #         logger.info(f"Sarvam job created: {job.job_id}")
    #
    #         # Upload and start processing
    #         job.upload_file(tmp_pdf_path)
    #         print("[SARVAM] File uploaded, starting processing...", flush=True)
    #         logger.info("Sarvam: File uploaded, starting processing...")
    #         job.start()
    #         print("[SARVAM] Job started, polling for completion...", flush=True)
    #
    #         # Poll with get_status() instead of wait_until_complete() for better control
    #         completed_states = {"completed", "Completed", "COMPLETED"}
    #         failed_states = {"failed", "Failed", "FAILED", "error", "Error", "ERROR"}
    #         max_polls = 180  # 180 x 5s = 15 minutes max
    #         poll_interval = 5  # seconds
    #
    #         for poll_num in range(1, max_polls + 1):
    #             _time.sleep(poll_interval)
    #             elapsed = _time.time() - start_time
    #
    #             try:
    #                 job_status = job.get_status()
    #                 state = job_status.job_state
    #                 print(f"[SARVAM] Poll {poll_num}: state={state} ({elapsed:.0f}s)", flush=True)
    #
    #                 if state in completed_states:
    #                     logger.info(f"Sarvam job completed in {elapsed:.0f}s")
    #                     break
    #                 elif state in failed_states:
    #                     raise Exception(f"Sarvam job failed with state: {state}")
    #             except AttributeError:
    #                 # If get_status() not available, fall back to wait_until_complete
    #                 print(f"[SARVAM] get_status() not available, using wait_until_complete()", flush=True)
    #                 job.wait_until_complete()
    #                 logger.info(f"Sarvam job completed via wait_until_complete in {elapsed:.0f}s")
    #                 break
    #         else:
    #             raise Exception(f"Sarvam job did not complete after {max_polls * poll_interval}s")
    #
    #         # Download output ZIP
    #         tmp_out_fd, tmp_zip_path = tempfile.mkstemp(suffix=".zip")
    #         os.close(tmp_out_fd)
    #         try:
    #             print("[SARVAM] Downloading output...", flush=True)
    #             job.download_output(tmp_zip_path)
    #             zip_size = os.path.getsize(tmp_zip_path)
    #             print(f"[SARVAM] Output downloaded ({zip_size} bytes), parsing...", flush=True)
    #             logger.info(f"Sarvam output downloaded ({zip_size} bytes)")
    #
    #             result = _parse_sarvam_zip(tmp_zip_path)
    #             total_time = _time.time() - start_time
    #             print(f"[SARVAM] Done! {len(result.get('pages', []))} pages, {sum(len(p.get('images', [])) for p in result.get('pages', []))} images ({total_time:.0f}s total)", flush=True)
    #             return result
    #         finally:
    #             if os.path.exists(tmp_zip_path):
    #                 os.unlink(tmp_zip_path)
    #     except Exception as e:
    #         print(f"[SARVAM] ERROR: {type(e).__name__}: {e}", flush=True)
    #         raise
    #     finally:
    #         if os.path.exists(tmp_pdf_path):
    #             os.unlink(tmp_pdf_path)
    #
    # try:
    #     result = await asyncio.wait_for(
    #         asyncio.get_event_loop().run_in_executor(None, _run_sarvam_sync),
    #         timeout=sarvam_timeout
    #     )
    #     logger.info(f"Sarvam OCR completed: {len(result.get('pages', []))} pages extracted")
    #     print("[OCR] Provider: Sarvam AI", flush=True)
    #     return result
    # except Exception as sarvam_err:
    #     print(f"[OCR] Sarvam failed ({type(sarvam_err).__name__}), falling back to GPT Vision...", flush=True)
    #     logger.warning(f"Sarvam OCR failed ({type(sarvam_err).__name__}: {sarvam_err}), attempting GPT Vision fallback")
    #     try:
    #         result = await call_gpt_vision_ocr(file_content)
    #         print("[OCR] Provider: GPT Vision (fallback)", flush=True)
    #         logger.info(f"GPT Vision fallback succeeded: {len(result.get('pages', []))} pages extracted")
    #         return result
    #     except Exception as gpt_err:
    #         print(f"[OCR] GPT Vision also failed: {gpt_err}", flush=True)
    #         logger.error(f"Both OCR providers failed. Sarvam: {sarvam_err}, GPT: {gpt_err}", exc_info=True)
    #         raise HTTPException(
    #             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    #             detail=f"Both OCR providers failed. Sarvam: {sarvam_err}, GPT: {gpt_err}"
    #         )


def _parse_sarvam_zip(zip_path: str) -> Dict[str, Any]:
    """Parse Sarvam output ZIP using metadata JSON for text (preserves LaTeX)
    and HTML for base64 images.

    Metadata JSON gives us clean OCR text with layout info and reading order.
    HTML is only used to extract embedded base64 images.
    """
    import zipfile
    import json
    import re as _re

    pages: List[Dict[str, Any]] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        file_list = zf.namelist()

        # ── Collect base64 images from HTML + standalone image files ──
        html_images: List[str] = []  # ordered base64 strings
        html_files = [f for f in file_list if f.lower().endswith(".html")]
        if html_files:
            html_content = zf.read(html_files[0]).decode("utf-8")
            for m in _re.finditer(
                r'<img[^>]+src="data:image/[^;]+;base64,([^"]+)"', html_content
            ):
                html_images.append(m.group(1))

        image_exts = (".png", ".jpg", ".jpeg", ".gif", ".webp")
        for entry in sorted(file_list):
            if entry.lower().endswith(image_exts):
                img_data = zf.read(entry)
                html_images.append(base64.b64encode(img_data).decode("utf-8"))

        # ── Read metadata JSON files (preferred path) ──
        meta_files = sorted(
            f for f in file_list if f.startswith("metadata/") and f.endswith(".json")
        )

        if not meta_files:
            # Fallback: old HTML-based conversion
            logger.warning("No metadata JSON in Sarvam ZIP, falling back to HTML")
            try:
                import html2text
                h2t = html2text.HTML2Text()
                h2t.body_width = 0
                h2t.unicode_snob = True
                h2t.images_to_alt = False
                h2t.single_line_break = False
                for idx, hf in enumerate(html_files):
                    pages.append({
                        "index": idx,
                        "markdown": h2t.handle(zf.read(hf).decode("utf-8")),
                        "images": [],
                        "dimensions": {"dpi": 150, "width": 0, "height": 0},
                    })
            except ImportError:
                logger.error("html2text not installed and no metadata JSON available")
            return {"pages": pages}

        global_img_idx = 0  # tracks which HTML image to assign next
        print(f"[SARVAM-PARSE] ZIP files: {file_list}", flush=True)
        print(f"[SARVAM-PARSE] {len(meta_files)} metadata files, {len(html_images)} HTML images", flush=True)

        for meta_file in meta_files:
            meta = json.loads(zf.read(meta_file).decode("utf-8"))
            page_num = meta.get("page_num", 1)
            page_idx = page_num - 1
            img_w = meta.get("image_width", 0)
            img_h = meta.get("image_height", 0)

            # Sort blocks by reading_order (they arrive sorted by confidence)
            blocks = sorted(
                meta.get("blocks", []),
                key=lambda b: b.get("reading_order", 0),
            )
            print(f"[SARVAM-PARSE] Page {page_idx}: {len(blocks)} blocks", flush=True)

            page_imgs: List[Dict[str, Any]] = []
            md_parts: List[str] = []

            for block in blocks:
                tag = block.get("layout_tag", "paragraph")
                text = (block.get("text") or "").strip()
                coords = block.get("coordinates", {})
                ro = block.get("reading_order", 0)
                preview = text[:80].replace('\n', '\\n') if text else "(empty)"
                print(f"[SARVAM-PARSE]   [{ro:2d}] {tag:15s} | {preview}", flush=True)

                if tag == "image":
                    img_id = f"img-{page_idx}-{len(page_imgs)}"
                    if global_img_idx < len(html_images):
                        page_imgs.append({
                            "id": img_id,
                            "image_base64": html_images[global_img_idx],
                            "top_left_x": int(coords.get("x1", 0)),
                            "top_left_y": int(coords.get("y1", 0)),
                            "bottom_right_x": int(coords.get("x2", 0)),
                            "bottom_right_y": int(coords.get("y2", 0)),
                        })
                        global_img_idx += 1
                    md_parts.append(f"![{img_id}]({img_id})")
                elif tag == "formula":
                    md_parts.append(text if text.startswith("$$") else f"$$ {text} $$")
                elif tag in ("section-title", "headline"):
                    md_parts.append(f"## {text}")
                elif tag == "table":
                    # Convert HTML table to plain text with question numbers.
                    # Many Indian exam papers are formatted as tables:
                    #   <td>Q.No</td><td>Question text</td><td>Marks</td>
                    tbl = (text or "").replace("<br/>", "\n").replace("<br>", "\n")
                    rows = _re.findall(r"<tr[^>]*>(.*?)</tr>", tbl, _re.DOTALL | _re.IGNORECASE)
                    tbl_lines: List[str] = []
                    for row in rows:
                        cells = _re.findall(
                            r"<t[dh][^>]*>(.*?)</t[dh]>", row, _re.DOTALL | _re.IGNORECASE
                        )
                        clean = [_re.sub(r"<[^>]+>", "", c).strip() for c in cells]
                        if not any(clean):
                            continue
                        # First cell is a question number → "N. question_text"
                        if (
                            clean[0]
                            and _re.match(r"^\d{1,3}$", clean[0])
                            and len(clean) > 1
                            and clean[1]
                        ):
                            tbl_lines.append(f"{clean[0]}. {clean[1]}")
                        else:
                            combined = " ".join(c for c in clean if c)
                            if combined:
                                tbl_lines.append(combined)
                    print(f"[SARVAM-PARSE]   Table converted: {len(rows)} rows -> {len(tbl_lines)} text lines", flush=True)
                    for tl in tbl_lines[:3]:
                        preview = tl[:100].replace('\n', '\\n')
                        print(f"[SARVAM-PARSE]     => {preview}", flush=True)
                    if tbl_lines:
                        md_parts.append("\n\n".join(tbl_lines))
                else:  # paragraph, footnote, etc.
                    if text:
                        md_parts.append(text)

            pages.append({
                "index": page_idx,
                "markdown": "\n\n".join(md_parts),
                "images": page_imgs,
                "dimensions": {"dpi": 150, "width": int(img_w), "height": int(img_h)},
            })

    logger.info(
        f"Parsed Sarvam ZIP (metadata): {len(pages)} pages, "
        f"{sum(len(p['images']) for p in pages)} images"
    )
    return {"pages": pages}

def split_composite_image(image_data: bytes, image_id: str) -> List[bytes]:
    """
    Detect if image contains multiple option figures (A, B, C, D) arranged horizontally or in grid
    and split them into individual images. Returns list of image bytes.
    """
    try:
        from PIL import Image
        import io
        import numpy as np

        # Load image
        img = Image.open(io.BytesIO(image_data))
        width, height = img.size

        # Convert to grayscale for analysis
        gray_img = img.convert('L')
        img_array = np.array(gray_img)

        # Check aspect ratio - if very wide (width > 2.5 * height), likely horizontal arrangement
        # If roughly square/tall, likely vertical or grid arrangement
        aspect_ratio = width / height if height > 0 else 1

        logger.info(f"Analyzing image {image_id}: size={width}x{height}, aspect_ratio={aspect_ratio:.2f}")

        # Heuristic: If aspect ratio > 2.5, likely 4 figures arranged horizontally
        # If aspect ratio between 1.5 and 2.5, likely 2 figures side by side
        if aspect_ratio > 2.0:
            # Try splitting horizontally into 4 parts for (A) (B) (C) (D)
            num_splits = 4 if aspect_ratio > 2.5 else 2
            split_width = width // num_splits

            logger.info(f"Image {image_id} has wide aspect ratio {aspect_ratio:.2f}, splitting into {num_splits} horizontal parts")

            split_images = []
            for i in range(num_splits):
                left = i * split_width
                right = (i + 1) * split_width if i < num_splits - 1 else width
                cropped = img.crop((left, 0, right, height))

                # Convert back to bytes
                output = io.BytesIO()
                cropped.save(output, format='JPEG', quality=95)
                split_images.append(output.getvalue())

            logger.info(f"Successfully split {image_id} into {len(split_images)} images")
            return split_images

        # If aspect ratio suggests grid (roughly square), try 2x2 split
        elif 0.8 <= aspect_ratio <= 1.5 and width > 300 and height > 300:
            logger.info(f"Image {image_id} has grid-like aspect ratio {aspect_ratio:.2f}, splitting into 2x2 grid")

            split_images = []
            mid_width = width // 2
            mid_height = height // 2

            # Top-left, top-right, bottom-left, bottom-right
            for row in range(2):
                for col in range(2):
                    left = col * mid_width
                    right = (col + 1) * mid_width if col == 0 else width
                    top = row * mid_height
                    bottom = (row + 1) * mid_height if row == 0 else height

                    cropped = img.crop((left, top, right, bottom))
                    output = io.BytesIO()
                    cropped.save(output, format='JPEG', quality=95)
                    split_images.append(output.getvalue())

            logger.info(f"Successfully split {image_id} into {len(split_images)} grid images")
            return split_images

        # Not a composite image
        logger.info(f"Image {image_id} does not appear to be a composite (aspect_ratio={aspect_ratio:.2f})")
        return [image_data]

    except Exception as e:
        logger.warning(f"Failed to analyze/split image {image_id}: {str(e)}")
        # Return original if splitting fails
        return [image_data]

async def save_image_to_disk(
    image_base64: str,
    image_id: str,
    pdf_filename: str,
    db: DatabaseManager,
    user_id: str,
    split_composite: bool = True,
    is_b2c: bool = False
) -> List[Dict[str, Any]]:
    """
    Save extracted image to disk and return metadata.
    If split_composite=True, detects and splits composite images with multiple option figures.
    Returns list of saved image metadata (1 item if not split, multiple if split).
    """
    try:
        # Log first 100 chars of base64 to debug
        logger.info(f"Saving image {image_id}, base64 preview: {image_base64[:100]}...")

        # Strip data URI prefix if present (e.g., "data:image/png;base64,")
        if ',' in image_base64 and image_base64.startswith('data:'):
            logger.info(f"Stripping data URI prefix from {image_id}")
            image_base64 = image_base64.split(',', 1)[1]

        # Decode base64 image
        image_data = base64.b64decode(image_base64)

        # Log decoded data info
        logger.info(f"Decoded {len(image_data)} bytes, first 16 bytes: {image_data[:16].hex()}")

        # Detect actual image format from magic bytes
        def detect_image_format(data: bytes) -> tuple[str, str]:
            """Detect image format from binary data, return (extension, content_type)"""
            if data.startswith(b'\xFF\xD8\xFF'):
                return 'jpeg', 'image/jpeg'
            elif data.startswith(b'\x89PNG\r\n\x1a\n'):
                return 'png', 'image/png'
            elif data.startswith(b'GIF87a') or data.startswith(b'GIF89a'):
                return 'gif', 'image/gif'
            elif data.startswith(b'RIFF') and b'WEBP' in data[:12]:
                return 'webp', 'image/webp'
            else:
                # Default to PNG if unknown
                logger.warning(f"Unknown image format for {image_id}, defaulting to PNG. First 16 bytes: {data[:16].hex()}")
                return 'png', 'image/png'

        # Split composite image if enabled
        image_parts = split_composite_image(image_data, image_id) if split_composite else [image_data]
        was_split = len(image_parts) > 1

        # Define uploads directory structure (only create if NOT using S3)
        upload_dir = os.path.join(os.getcwd(), "uploads", "pdf_images", pdf_filename.replace('.pdf', ''))

        # Generate filename with correct extension based on actual format
        # Strip any existing extension from image_id
        base_image_id = image_id.split('.')[0] if '.' in image_id else image_id

        saved_images = []
        
        # ALWAYS save the original unsplit image first with base_image_id
        # This ensures question figures have access to the complete image
        original_detected_ext, original_content_type = detect_image_format(image_data)
        original_image_filename = f"{base_image_id}.{original_detected_ext}"
        original_file_path = os.path.join(upload_dir, original_image_filename)
        
        # Use S3 storage if enabled, otherwise save locally
        if is_s3_enabled():
            # Upload to S3
            success, storage_path = await s3_upload_file(
                file_data=image_data,
                local_path=original_file_path,
                content_type=original_content_type
            )
            if success:
                logger.info(f"✅ Saved image to S3: {storage_path}")
                original_relative_path = storage_path  # Store S3 path
            else:
                logger.warning(f"S3 upload failed, image not saved: {original_image_filename}")
                original_relative_path = ""
        else:
            # Save locally (fallback)
            os.makedirs(upload_dir, exist_ok=True)
            async with aiofiles.open(original_file_path, "wb") as f:
                await f.write(image_data)
            logger.info(f"Saved original image locally: {original_image_filename}")
            original_relative_path = get_relative_path(original_file_path)
        
        # Save original to database
        original_metadata = {
            "_id": base_image_id,
            "filename": original_image_filename,
            "original_filename": original_image_filename,
            "size": len(image_data),
            "content_type": original_content_type,
            "uploaded_by": user_id,
            "uploaded_at": datetime.utcnow(),
            "is_processed": True,
            "file_path": original_relative_path,
            "source_pdf": pdf_filename,
            "tags": ["pdf_extracted", "ocr", "original"],
            "was_split": was_split,
            "is_s3": is_s3_enabled()  # Track storage type
        }
        
        # Save to database (use update_one with upsert to handle re-processing)
        if is_b2c:
            await db.b2c_update_one("images", {"_id": base_image_id}, {"$set": original_metadata}, upsert=True)
        else:
            await db.mongo_update_one("images", {"_id": base_image_id}, {"$set": original_metadata}, upsert=True)
        
        saved_images.append({
            "id": base_image_id,
            "filename": original_image_filename,
            "path": original_relative_path,
            "url": f"/api/v1/images/{base_image_id}",
            "size": len(image_data),
            "is_original": True
        })
        
        # If the image was split, also save split parts with -A, -B, etc. suffixes
        if was_split:
            for idx, img_data in enumerate(image_parts):
                # Detect format for this part
                detected_ext, content_type = detect_image_format(img_data)
                
                # Create unique ID for each split part
                db_image_id = f"{base_image_id}-{chr(65+idx)}"  # img-9-A, img-9-B, img-9-C, img-9-D
                image_filename = f"{db_image_id}.{detected_ext}"
                file_path = os.path.join(upload_dir, image_filename)

                # Use S3 storage if enabled, otherwise save locally
                if is_s3_enabled():
                    success, storage_path = await s3_upload_file(
                        file_data=img_data,
                        local_path=file_path,
                        content_type=content_type
                    )
                    if success:
                        logger.info(f"✅ Saved split part {idx+1} to S3: {storage_path}")
                        relative_path = storage_path
                    else:
                        logger.warning(f"S3 upload failed for split part: {image_filename}")
                        relative_path = ""
                else:
                    os.makedirs(upload_dir, exist_ok=True)
                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(img_data)
                    logger.info(f"Saved split part {idx+1}/{len(image_parts)} locally: {image_filename}")
                    relative_path = get_relative_path(file_path)

                # Create image metadata for database
                image_metadata = {
                    "_id": db_image_id,
                    "filename": image_filename,
                    "original_filename": original_image_filename,
                    "size": len(img_data),
                    "content_type": content_type,
                    "uploaded_by": user_id,
                    "uploaded_at": datetime.utcnow(),
                    "is_processed": True,
                    "file_path": relative_path,
                    "source_pdf": pdf_filename,
                    "tags": ["pdf_extracted", "ocr", "split_composite"],
                    "parent_image_id": base_image_id,
                    "split_index": idx,
                    "is_s3": is_s3_enabled()
                }

                # Save to database
                if is_b2c:
                    await db.b2c_update_one("images", {"_id": db_image_id}, {"$set": image_metadata}, upsert=True)
                else:
                    await db.mongo_update_one("images", {"_id": db_image_id}, {"$set": image_metadata}, upsert=True)

                saved_images.append({
                    "id": db_image_id,
                    "filename": image_filename,
                    "path": relative_path,
                    "url": f"/api/v1/images/{db_image_id}",
                    "size": len(img_data),
                    "is_original": False
                })

        return saved_images

    except Exception as e:
        logger.error(f"Failed to save image {image_id}: {str(e)}")
        return []

async def extract_questions_with_gpt(
    ocr_result: Dict[str, Any],
    subject: str,
    difficulty: str,
    skip_option_extraction: bool = False,
    document_anchor_text: Optional[str] = None,
    gateway_context: Optional[Dict[str, Any]] = None,
    layout_report: Optional[Dict[str, Any]] = None,
    retry_reason: Optional[str] = None,
) -> List[ExtractedQuestion]:
    """
    Use LLM to extract structured questions from OCR text.
    Uses Groq (GPT-OSS 120B) as primary, falls back to OpenAI GPT-5-mini.
    Works with ANY question paper format — no hardcoded regex patterns.
    """
    import json as _json
    from openai import AsyncOpenAI

    pages = ocr_result.get("pages", [])
    if not pages:
        return []

    # Build full text with page markers so LLM can report which page each question is on
    full_text = ""
    for page in pages:
        pidx = page.get("index", 0)
        md = page.get("markdown", "")
        full_text += f"\n=== PAGE {pidx} ===\n{md}"

    # Select LLM provider: Groq (primary) or OpenAI (fallback)
    # Groq API is OpenAI-compatible — same AsyncOpenAI client, different base_url
    if GROQ_API_KEY:
        client = AsyncOpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        extract_model = GROQ_MODEL
        provider_name = "Groq"
    else:
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key:
            raise Exception("Neither GROQ_API_KEY nor OPENAI_API_KEY configured — cannot extract questions")
        client = AsyncOpenAI(api_key=openai_key)
        extract_model = OCR_FALLBACK_MODEL
        provider_name = "OpenAI"

    # --- To switch back to OpenAI GPT-5-mini, comment out the Groq block above
    # --- and uncomment these two lines:
    # client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    # extract_model = OCR_FALLBACK_MODEL

    print(f"[Q-EXTRACT] Sending {len(full_text)} chars from {len(pages)} pages (provider: {provider_name}, model: {extract_model})", flush=True)
    anchor_instruction = ""
    if document_anchor_text and document_anchor_text.strip():
        anchor_instruction = (
            "\nDOCUMENT ANCHOR TEXT PROVIDED BY TEACHER:\n"
            f"{document_anchor_text.strip()}\n"
            "Use this anchor text as an additional hint when locating and organizing relevant question content. "
            "Do not invent content from the anchor text; only extract what is present in the document.\n"
        )
    layout_instruction = ""
    if layout_report:
        layout_instruction = (
            "\nLAYOUT PREFLIGHT REPORT:\n"
            f"{json.dumps(layout_report, default=str)[:6000]}\n"
            "Use this deterministic crop-layout report to avoid trusting OCR reading order blindly.\n"
        )
        if "staggered_options" in (layout_report.get("layout_risks") or []):
            layout_instruction += (
                "This cropped question region may have staggered MCQ options. "
                "Some option labels may appear below their option text. "
                "Use visual/layout association and the supplied layout report. "
                "Do not bind options only by reading order. "
                "Return exactly four options when four visual choices exist.\n"
            )
    if retry_reason:
        layout_instruction += (
            f"\nRETRY REASON: {retry_reason}. Re-check option association before returning JSON.\n"
        )

    extraction_prompt = (
        "You are a question paper parser. Extract ONLY the questions from the text below.\n\n"
        f"{anchor_instruction}"
        f"{layout_instruction}"
        "RULES:\n"
        "- Extract every question (MCQ, subjective, fill-in-the-blank, true/false, assertion-reason, case study, etc.)\n"
        "- Ignore headers, instructions, school name, exam title, general instructions, section headers, marks info\n"
        "- For MCQs: separate the question text from the options\n"
        "- For subjective questions: include the full question text, leave options as empty array\n"
        "- IMPORTANT: If a question has sub-parts (a, b, c or i, ii, iii or (1), (2) etc.), keep them as ONE question. Include ALL sub-parts in the question text. Do NOT split sub-parts into separate questions.\n"
        "- For case study / passage-based questions: include the passage/context + ALL sub-questions as ONE question\n"
        "- If a question has an OR alternative, include the OR part in the same question text\n"
        "- Preserve ALL math notation, LaTeX, symbols, superscripts, subscripts exactly as they appear\n"
        "- Preserve Hindi or regional language text exactly as-is\n"
        "- If a question references a figure/diagram/graph/image/table, set has_figure to true\n"
        "- Report the page number (from the === PAGE N === markers) where each question starts\n\n"
        "Return ONLY valid JSON in this exact format (no markdown fences, no explanation):\n"
        '{"questions": [\n'
        '  {"number": "1", "text": "full question text here", "options": ["option a", "option b", "option c", "option d"], "page": 0, "has_figure": false},\n'
        '  {"number": "27", "text": "(a) first sub-part here\\n(b) second sub-part here", "options": [], "page": 3, "has_figure": true}\n'
        "]}\n\n"
        "--- DOCUMENT TEXT ---\n"
        f"{full_text}"
    )

    # Try extraction with retry — model can return empty responses
    raw_response = ""
    max_retries = 2
    async def _chat_completion(prompt: str):
        async def _raw_call():
            return await client.chat.completions.create(
                model=extract_model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=16384,
            )

        if not gateway_context:
            return await _raw_call()
        gateway = AIGatewayService(
            gateway_context.get("db"),
            is_b2c=bool(gateway_context.get("is_b2c")),
        )
        return await gateway.call(
            user_id=str(gateway_context.get("user_id") or "unknown"),
            tenant_id=gateway_context.get("tenant_id"),
            document_id=gateway_context.get("document_id"),
            region_id=gateway_context.get("region_id"),
            region_scope=gateway_context.get("region_scope"),
            stage="question_structuring_retry" if retry_reason else "question_structuring",
            provider=provider_name.lower(),
            model=extract_model,
            input_kind="text",
            estimated_input_tokens=estimate_text_tokens(prompt),
            estimated_output_tokens=4096,
            max_output_tokens=16384,
            call_fn=_raw_call,
        )

    for attempt in range(1, max_retries + 1):
        try:
            response = await _chat_completion(extraction_prompt)
            raw_response = response.choices[0].message.content or ""
            print(f"[Q-EXTRACT] Attempt {attempt}: got {len(raw_response)} chars", flush=True)
            if raw_response.strip():
                break
            if attempt < max_retries:
                print(f"[Q-EXTRACT] Empty response, retrying...", flush=True)
        except Exception as e:
            print(f"[Q-EXTRACT] Attempt {attempt} failed: {e}", flush=True)
            if attempt >= max_retries:
                logger.error(f"Question extraction failed after {max_retries} attempts: {e}")
                return []

    if not raw_response.strip():
        print(f"[Q-EXTRACT] All {max_retries} attempts returned empty response", flush=True)
        return []

    # Parse JSON — handle markdown fences, then fix LaTeX backslash issues
    import re as _re
    raw_response = raw_response.strip()
    if raw_response.startswith("```"):
        raw_response = raw_response.split("\n", 1)[-1]  # remove first ```json line
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3].strip()

    def _fix_json_backslashes(s: str) -> str:
        """Fix invalid backslash escapes in JSON strings caused by LaTeX.

        The tricky part: \\r is a valid JSON escape (carriage return) but in
        LaTeX context it means \\rho. Same for \\b (backspace vs \\beta),
        \\f (form feed vs \\frac), \\n (newline vs \\nu), \\t (tab vs \\times).

        Strategy: if \\ + [bfnrt] is followed by more alphabetic chars (e.g.
        \\rho, \\beta, \\frac, \\nu, \\times), it's LaTeX — escape it.
        If it stands alone (\\n at end, \\t followed by non-alpha), it's JSON.
        """
        result = []
        in_string = False
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == '"' and (i == 0 or s[i - 1] != '\\'):
                in_string = not in_string
                result.append(ch)
                i += 1
            elif in_string and ch == '\\':
                if i + 1 < len(s):
                    nxt = s[i + 1]
                    if nxt == '\\' or nxt == '"' or nxt == '/':
                        # Definitely valid JSON escape — keep
                        result.append(ch)
                        result.append(nxt)
                        i += 2
                    elif nxt == 'u' and i + 5 < len(s) and all(c in '0123456789abcdefABCDEF' for c in s[i+2:i+6]):
                        # Unicode escape \uXXXX — keep
                        result.append(ch)
                        result.append(nxt)
                        i += 2
                    elif nxt in ('b', 'f', 'n', 'r', 't'):
                        # Ambiguous: could be JSON escape OR LaTeX command
                        # Check if followed by more alpha chars → LaTeX
                        after = i + 2
                        if after < len(s) and s[after].isalpha():
                            # LaTeX: \rho, \beta, \frac, \nu, \times etc.
                            result.append('\\')
                            result.append('\\')
                            result.append(nxt)
                            i += 2
                        else:
                            # Standalone JSON escape: \n, \t, \r etc.
                            result.append(ch)
                            result.append(nxt)
                            i += 2
                    else:
                        # Not a valid JSON escape — must be LaTeX (\mu, \lambda, etc.)
                        result.append('\\')
                        result.append('\\')
                        result.append(nxt)
                        i += 2
                else:
                    result.append(ch)
                    i += 1
            else:
                result.append(ch)
                i += 1
        return ''.join(result)

    # Try parsing raw first, fix backslashes only if needed
    data = None
    try:
        data = _json.loads(raw_response)
    except _json.JSONDecodeError:
        # Fix LaTeX backslashes and retry
        try:
            fixed = _fix_json_backslashes(raw_response)
            data = _json.loads(fixed)
            print(f"[Q-EXTRACT] Fixed LaTeX backslashes in JSON response", flush=True)
        except _json.JSONDecodeError as e2:
            logger.error(f"Failed to parse GPT extraction response as JSON: {e2}")
            print(f"[Q-EXTRACT] JSON parse failed even after backslash fix: {e2}", flush=True)
            print(f"[Q-EXTRACT] Raw response (first 500): {raw_response[:500]}", flush=True)
            return []

    raw_questions = data.get("questions", [])
    print(f"[Q-EXTRACT] Parsed {len(raw_questions)} questions from GPT response", flush=True)

    # Build page → image IDs map from OCR result so we can associate real image IDs
    page_image_ids: Dict[int, List[str]] = {}
    for page in pages:
        pidx = page.get("index", 0)
        img_ids = [img.get("id", "") for img in page.get("images", []) if img.get("image_base64")]
        if img_ids:
            page_image_ids[pidx] = img_ids
    if page_image_ids:
        print(f"[Q-EXTRACT] Page image map: {page_image_ids}", flush=True)

    # NOTE: the positional image→question map is built AFTER the LLM extraction
    # below (we need the LLM's page numbers as a fallback anchor source for
    # questions that PyMuPDF couldn't detect by text-block matching, like
    # case-study headings that use "Case Study Based- 3" instead of "19.").
    positional_image_map: Dict[int, List[str]] = {}

    # Build page text lookup for targeted retries
    page_texts: Dict[int, str] = {}
    for page in pages:
        page_texts[page.get("index", 0)] = page.get("markdown", "")

    # Detect questions with empty text — collect for targeted retry
    failed_questions: List[Dict[str, Any]] = []  # {"number", "page"} for retry
    good_questions: List[Dict[str, Any]] = []
    for q in raw_questions:
        q_text = q.get("text", "").strip()
        if not q_text:
            failed_questions.append(q)
        else:
            good_questions.append(q)

    if failed_questions:
        print(
            f"[Q-EXTRACT] {len(failed_questions)} questions have empty text — "
            f"retrying: {[q.get('number', '?') for q in failed_questions]}",
            flush=True,
        )

        # Collect ONLY the page text for the failed questions' pages
        failed_pages: set = set()
        for q in failed_questions:
            p = q.get("page", 0)
            failed_pages.add(p)

        retry_text = ""
        for pidx in sorted(failed_pages):
            md = page_texts.get(pidx, "")
            if md:
                retry_text += f"\n=== PAGE {pidx} ===\n{md}"

        if retry_text.strip():
            failed_nums = [q.get("number", "?") for q in failed_questions]
            retry_prompt = (
                "You previously failed to extract the text for these question numbers: "
                f"{failed_nums}\n\n"
                "Below is the page text containing those questions. "
                "Extract ONLY the questions listed above.\n\n"
                f"{anchor_instruction}"
                "RULES:\n"
                "- Include full question text with all sub-parts\n"
                "- For MCQs: separate question text from options\n"
                "- Preserve ALL math notation, LaTeX, symbols exactly\n"
                "- If a question references a figure/diagram, set has_figure to true\n"
                "- Report the page number from === PAGE N === markers\n\n"
                "Return ONLY valid JSON:\n"
                '{"questions": [{"number": "1", "text": "...", "options": [...], "page": 0, "has_figure": false}]}\n\n'
                "--- PAGE TEXT ---\n"
                f"{retry_text}"
            )

            try:
                retry_response = await _chat_completion(retry_prompt)
                retry_raw = (retry_response.choices[0].message.content or "").strip()
                if retry_raw:
                    if retry_raw.startswith("```"):
                        retry_raw = retry_raw.split("\n", 1)[-1]
                        if retry_raw.endswith("```"):
                            retry_raw = retry_raw[:-3].strip()
                    retry_fixed = _fix_json_backslashes(retry_raw)
                    try:
                        retry_data = _json.loads(retry_raw)
                    except _json.JSONDecodeError:
                        retry_data = _json.loads(retry_fixed)
                    retry_qs = retry_data.get("questions", [])

                    # Merge retried questions back — match by question number
                    retry_by_num = {str(rq.get("number", "")): rq for rq in retry_qs if rq.get("text", "").strip()}
                    recovered = 0
                    for fq in failed_questions:
                        fnum = str(fq.get("number", ""))
                        if fnum in retry_by_num:
                            good_questions.append(retry_by_num[fnum])
                            recovered += 1
                            print(f"[Q-EXTRACT] Recovered Q#{fnum} via retry", flush=True)
                        else:
                            # Still failed — add as empty placeholder so numbering is preserved
                            good_questions.append(fq)
                            print(f"[Q-EXTRACT] Q#{fnum} still empty after retry", flush=True)

                    print(f"[Q-EXTRACT] Retry recovered {recovered}/{len(failed_questions)} questions", flush=True)
                else:
                    print(f"[Q-EXTRACT] Retry returned empty — keeping original results", flush=True)
                    good_questions.extend(failed_questions)
            except Exception as retry_err:
                print(f"[Q-EXTRACT] Retry failed: {retry_err} — keeping original results", flush=True)
                good_questions.extend(failed_questions)

    # Sort by question number to restore original order
    def _sort_key(q: Dict) -> int:
        try:
            return int(q.get("number", 0))
        except (ValueError, TypeError):
            return 9999
    good_questions.sort(key=_sort_key)

    # ---------- Build positional image→question map ----------
    # Anchors are points along the PDF where a numbered question starts. For
    # each image we walk a globally ordered list of anchors and attribute the
    # image to the most recent anchor whose (page, y) precedes the image's
    # (page, y). This handles single-page questions AND multi-page case
    # studies (where follow-on figure pages have no new numbered question and
    # so naturally inherit the case-study question from a previous page).
    #
    # Anchor sources (in priority order):
    #   1. PyMuPDF text-block detection — gives us exact y-position when the
    #      question marker is "11." / "11)" plain text.
    #   2. LLM-reported page index — fallback when PyMuPDF couldn't detect the
    #      question (e.g., case-study questions with non-numeric headings).
    #      Synthetic anchor lives at (LLM page, y=0) so it dominates anything
    #      on later pages but defers to PyMuPDF anchors on the same page.
    def _q_num_int(q: Any) -> Optional[int]:
        try:
            s = str(q.get("number", "")).strip()
            digits = ""
            for ch in s:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            return int(digits) if digits else None
        except (ValueError, TypeError):
            return None

    pymupdf_detected_qnums: set = set()
    anchors: List[tuple] = []  # (page_idx, y_start, q_number)
    for page in pages:
        pidx = page.get("index", 0)
        qblocks = page.get("_pymupdf_question_blocks") or []
        for qb in sorted(qblocks, key=lambda b: b["y_start"]):
            qnum = int(qb["number"])
            anchors.append((pidx, float(qb["y_start"]), qnum))
            pymupdf_detected_qnums.add(qnum)

    # For LLM-extracted questions PyMuPDF missed (case studies, sub-numbered
    # questions, etc.), search the PDF text blocks for a distinctive snippet
    # of the question text. The first match position becomes the anchor.
    # This handles case-study headings like "Case Study Based- 3" that don't
    # use a standard "<num>." prefix and would otherwise be invisible to
    # PyMuPDF's question-marker regex.
    def _find_text_anchor(q_text: str) -> Optional[tuple]:
        """Return (page_idx, y_start) of the earliest PyMuPDF text block that
        contains a distinctive snippet of `q_text`, or None if not found."""
        if not q_text:
            return None
        # Pull a meaningful snippet — skip leading punctuation/whitespace,
        # take 3-6 distinctive words. Case studies start with phrases like
        # "Case Study Based- 3 Applications of Parabolas" — the first 30
        # chars are usually unique enough across the document.
        snippet = " ".join(q_text.split())[:50].strip()
        if len(snippet) < 8:
            return None
        # Reduce snippet length until it's likely to be a substring of a single
        # text block (long snippets risk straddling block boundaries).
        for needle_len in (50, 35, 20):
            needle = snippet[:needle_len].strip()
            if len(needle) < 8:
                continue
            for page in pages:
                pidx = page.get("index", 0)
                for tb in page.get("_pymupdf_text_blocks") or []:
                    if needle in tb["text"]:
                        return (pidx, float(tb["y_start"]))
        return None

    for q in good_questions:
        qni = _q_num_int(q)
        if qni is None or qni in pymupdf_detected_qnums:
            continue
        text_anchor = _find_text_anchor(q.get("text", ""))
        if text_anchor is not None:
            anchors.append((text_anchor[0], text_anchor[1], qni))
            continue
        # Last resort: use the LLM's reported page at y=0.
        try:
            qpage = int(q.get("page", 0))
        except (ValueError, TypeError):
            continue
        anchors.append((qpage, 0.0, qni))

    # Sort by (page, y) so the linear walk in the matching loop is correct.
    anchors.sort(key=lambda a: (a[0], a[1]))

    for page in pages:
        pidx = page.get("index", 0)
        # Only process images that have a real bbox (top_left_y > 0). Mistral
        # OCR currently returns images with bbox=0 (no positional info), and
        # those should fall through to the cursor matching path which doesn't
        # rely on position. PyMuPDF-injected images always have real bboxes.
        page_imgs = sorted(
            [
                img for img in page.get("images", [])
                if img.get("image_base64") and float(img.get("top_left_y") or 0) > 0
            ],
            key=lambda img: float(img.get("top_left_y") or 0),
        )
        for img in page_imgs:
            img_y = float(img.get("top_left_y") or 0)
            owning_qnum: Optional[int] = None
            for ap, ay, an in anchors:
                if ap < pidx or (ap == pidx and ay <= img_y):
                    owning_qnum = an
                elif ap > pidx:
                    break
            if owning_qnum is None and anchors:
                owning_qnum = anchors[0][2]
            if owning_qnum is not None:
                positional_image_map.setdefault(owning_qnum, []).append(img.get("id", ""))
    if positional_image_map:
        print(f"[Q-EXTRACT] Positional image map: {positional_image_map}", flush=True)

    # Track every image ID consumed by positional matching so the cursor
    # fallback below cannot re-assign the same image to a different question.
    used_image_ids: set = set()
    for refs in positional_image_map.values():
        used_image_ids.update(refs)

    # Build final ExtractedQuestion list
    # Track which image has been assigned per page so each figure-question
    # gets the NEXT unassigned image instead of ALL images on the page.
    page_image_cursor: Dict[int, int] = {}
    questions: List[ExtractedQuestion] = []
    for q in good_questions:
        q_text = q.get("text", "").strip()
        if not q_text:
            continue

        q_num = q.get("number", "")
        q_page = q.get("page", 0)
        options = q.get("options", [])
        has_image = q.get("has_figure", False) or "![" in q_text or "figure" in q_text.lower() or "diagram" in q_text.lower() or "graph" in q_text.lower()

        # Image assignment: prefer the deterministic positional map (built from
        # PyMuPDF text-block y-positions) over the cursor heuristic. The
        # positional map is authoritative when present because it doesn't rely
        # on the LLM correctly setting `has_figure` — it just looks at where
        # the image actually sits on the PDF page.
        img_refs: List[str] = []
        q_num_int: Optional[int] = None
        try:
            # Question number may come back as "11", "11.", "11)", "11a", etc.
            _num_str = str(q_num).strip()
            _digits = ""
            for ch in _num_str:
                if ch.isdigit():
                    _digits += ch
                else:
                    break
            if _digits:
                q_num_int = int(_digits)
        except (ValueError, TypeError):
            q_num_int = None

        positional_hit = (
            q_num_int is not None
            and q_num_int in positional_image_map
        )
        if positional_hit:
            img_refs = list(positional_image_map[q_num_int])
            # If we found images via position, the question definitely has a
            # figure regardless of what the LLM said.
            has_image = True
        elif has_image and q_page in page_image_ids:
            # Fallback: cursor-based 1:1 assignment in reading order, skipping
            # any image already claimed by positional matching. Used for PDFs
            # where question-number detection failed (e.g., scanned PDFs with
            # no recoverable text blocks) but the LLM did flag the question as
            # having a figure.
            all_page_imgs = page_image_ids[q_page]
            cursor = page_image_cursor.get(q_page, 0)
            while cursor < len(all_page_imgs) and all_page_imgs[cursor] in used_image_ids:
                cursor += 1
            if cursor < len(all_page_imgs):
                img_refs = [all_page_imgs[cursor]]
                used_image_ids.add(all_page_imgs[cursor])
                page_image_cursor[q_page] = cursor + 1
            else:
                page_image_cursor[q_page] = cursor

        # For Practice Sets mode, inline options into the question text
        if skip_option_extraction and options:
            inline = q_text + "\n\n"
            for i, opt in enumerate(options):
                inline += f"({chr(65 + i)}) {opt}\n"
            questions.append(ExtractedQuestion(
                id=str(uuid.uuid4()),
                text=inline.strip(),
                options=[],
                metadata={
                    "subject": subject,
                    "difficulty": difficulty,
                    "page": q_page,
                    "question_number": q_num,
                    "has_figure": has_image,
                    "image_refs": img_refs,
                    "question_image_refs": img_refs,
                    "is_image_based_mcq": False,
                    "options_inline": True,
                },
            ))
        else:
            questions.append(ExtractedQuestion(
                id=str(uuid.uuid4()),
                text=q_text,
                options=[o for o in options if o.strip()],
                metadata={
                    "subject": subject,
                    "difficulty": difficulty,
                    "page": q_page,
                    "question_number": q_num,
                    "has_figure": has_image,
                    "image_refs": img_refs,
                    "question_image_refs": img_refs,
                    "is_image_based_mcq": False,
                },
            ))

        q_preview = q_text[:80].replace('\n', ' | ')
        fig_flag = " [HAS_FIGURE]" if has_image else ""
        print(f"[Q-EXTRACT]   Q#{q_num} p{q_page}: \"{q_preview}\" opts={len(options)}{fig_flag}", flush=True)

    # Validation: warn if extracted count seems low for the number of pages
    expected_min = max(1, len(pages) * 2)  # heuristic: at least 2 questions per page
    if len(questions) < expected_min:
        print(
            f"[Q-EXTRACT] WARNING: Only {len(questions)} questions extracted from "
            f"{len(pages)} pages (expected at least ~{expected_min}). "
            f"Response may be truncated.",
            flush=True
        )
        logger.warning(
            f"Low question count: {len(questions)} from {len(pages)} pages "
            f"(expected >= {expected_min})"
        )

    logger.info(f"Extracted {len(questions)} questions from {len(pages)} pages (provider: {provider_name})")
    return questions


def _region_sort_key(region: Dict[str, Any]) -> tuple:
    return (
        int(region.get("pageNumber", 0) or 0),
        float(region.get("y", 0) or 0),
        float(region.get("x", 0) or 0),
        str(region.get("id") or ""),
    )


def _number_cues_from_text(text: str) -> List[str]:
    import re

    cues: List[str] = []
    for pattern in (
        r"^\s*(?:q(?:uestion)?\.?\s*)?(\d{1,3})[\.\)]\s+",
        r"\bq(?:uestion)?\.?\s*(\d{1,3})\b",
    ):
        match = re.search(pattern, text or "", re.IGNORECASE)
        if match:
            cues.append(str(int(match.group(1))))
    return cues


def _resolve_question_context_for_answer_region(
    *,
    answer_region: Dict[str, Any],
    answer_text: str,
    answer_region_order: int,
    question_regions: List[Dict[str, Any]],
    questions_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    questions_by_number: Dict[str, Dict[str, Any]] = {}
    for index, region in enumerate(question_regions, start=1):
        for number in {
            str(index),
            *_number_cues_from_text(str(region.get("label") or "")),
            *_number_cues_from_text(str(region.get("extractedText") or "")),
            *_number_cues_from_text(str(region.get("id") or "")),
        }:
            if number and number not in questions_by_number:
                questions_by_number[number] = region

    matched_region: Optional[Dict[str, Any]] = None
    for number in _number_cues_from_text(answer_text):
        candidate = questions_by_number.get(number)
        if candidate:
            matched_region = candidate
            break

    match_strategy = "question_number" if matched_region else "region_order"
    if matched_region is None and 0 <= answer_region_order < len(question_regions):
        matched_region = question_regions[answer_region_order]

    question_id = str((matched_region or {}).get("id") or "")
    question_doc = questions_by_id.get(question_id, {})
    return {
        "question_id": question_id,
        "question_label": (matched_region or {}).get("label"),
        "match_strategy": match_strategy,
        "question_text": question_doc.get("text") or question_doc.get("question_text") or (matched_region or {}).get("extractedText") or "",
        "options": question_doc.get("options") or [],
        "correct_answer": question_doc.get("correct_answer"),
        "answer_region_id": answer_region.get("id"),
    }


async def extract_worked_answer_with_gpt(
    *,
    ocr_result: Dict[str, Any],
    raw_answer_text: str,
    question_context: Dict[str, Any],
    document_anchor_text: Optional[str] = None,
    gateway_context: Optional[Dict[str, Any]] = None,
    layout_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract only the worked solution from an OCR'd answer-sheet region."""
    import json as _json
    from openai import AsyncOpenAI

    if GROQ_API_KEY:
        client = AsyncOpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        extract_model = GROQ_MODEL
        provider_name = "Groq"
    else:
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key:
            raise Exception("Neither GROQ_API_KEY nor OPENAI_API_KEY configured - cannot structure worked answer")
        client = AsyncOpenAI(api_key=openai_key)
        extract_model = OCR_FALLBACK_MODEL
        provider_name = "OpenAI"

    pages_text = _ocr_pages_to_plain_text(ocr_result)
    anchor_instruction = ""
    if document_anchor_text and document_anchor_text.strip():
        anchor_instruction = (
            "\nTEACHER DOCUMENT ANCHOR TEXT / INSTRUCTION:\n"
            f"{document_anchor_text.strip()}\n"
            "Use this as the teacher's hint for where the actual worked solution starts "
            "or how final solution syntax should be organized. If the OCR region contains "
            "content before this anchor, treat that earlier content as question/restatement "
            "unless it is clearly required for the worked solution.\n"
        )

    layout_instruction = ""
    if layout_report:
        layout_instruction = (
            "\nLAYOUT PREFLIGHT REPORT:\n"
            f"{json.dumps(layout_report, default=str)[:4000]}\n"
            "Use this crop-layout report as a hint when OCR reading order is noisy.\n"
        )

    question_text = str(question_context.get("question_text") or "").strip()
    options_text = "\n".join(
        f"{chr(65 + idx)}. {option}"
        for idx, option in enumerate(question_context.get("options") or [])
    )
    prompt = (
        "You are cleaning OCR output from a worked-answer sheet for an objective test.\n"
        "Extract ONLY the worked solution/explanation for the mapped question.\n\n"
        f"{anchor_instruction}"
        f"{layout_instruction}"
        "MAPPED QUESTION CONTEXT - do not copy this into the answer unless a formula/value is needed:\n"
        f"Question ID: {question_context.get('question_id') or ''}\n"
        f"Question label: {question_context.get('question_label') or ''}\n"
        f"Correct option: {question_context.get('correct_answer') or ''}\n"
        f"Question:\n{question_text[:6000]}\n"
        f"Options:\n{options_text[:3000]}\n\n"
        "RULES:\n"
        "- Remove any restated question text, option list, exam headers, page headers, or repeated prompt text.\n"
        "- Keep the mathematical derivation, explanation, final answer, diagrams described in text, and conclusion.\n"
        "- If the teacher anchor text is present in OCR, start the solution from that anchor or immediately after it.\n"
        "- If no worked solution is present, return an empty answer_text and set manual_review_required true.\n"
        "- Preserve math notation, LaTeX, symbols, and line breaks.\n"
        "- Return ONLY valid JSON, with no markdown fences.\n\n"
        "JSON format:\n"
        '{"answer_text": "worked solution only", "confidence": 0.0, "manual_review_required": false, "notes": ""}\n\n'
        "--- RAW ANSWER REGION OCR ---\n"
        f"{pages_text or raw_answer_text}"
    )

    async def _raw_call():
        return await client.chat.completions.create(
            model=extract_model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=8192,
        )

    if gateway_context:
        gateway = AIGatewayService(
            gateway_context.get("db"),
            is_b2c=bool(gateway_context.get("is_b2c")),
        )
        response = await gateway.call(
            user_id=str(gateway_context.get("user_id") or "unknown"),
            tenant_id=gateway_context.get("tenant_id"),
            document_id=gateway_context.get("document_id"),
            region_id=gateway_context.get("region_id"),
            region_scope=gateway_context.get("region_scope"),
            stage="answer_structuring",
            provider=provider_name.lower(),
            model=extract_model,
            input_kind="text",
            estimated_input_tokens=estimate_text_tokens(prompt),
            estimated_output_tokens=2048,
            max_output_tokens=8192,
            call_fn=_raw_call,
        )
    else:
        response = await _raw_call()

    raw_response = (response.choices[0].message.content or "").strip()
    if raw_response.startswith("```"):
        raw_response = raw_response.split("\n", 1)[-1]
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3].strip()

    data = _json.loads(raw_response)
    answer_text = str(data.get("answer_text") or "").strip()
    return {
        "answer_text": answer_text,
        "confidence": data.get("confidence"),
        "manual_review_required": bool(data.get("manual_review_required") or not answer_text),
        "notes": data.get("notes") or "",
        "provider": provider_name.lower(),
        "model": extract_model,
    }


def _normalise_correct_answer_label(value: Any) -> str:
    label = str(value or "").strip().upper()
    if label in {"A", "B", "C", "D"}:
        return label
    if label in {"1", "2", "3", "4"}:
        return chr(64 + int(label))
    return label


async def generate_worked_solution_batch(
    *,
    questions: List[Dict[str, Any]],
    document: Dict[str, Any],
    gateway_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate worked MCQ explanations for a batch of extracted questions."""
    import json as _json
    from openai import AsyncOpenAI

    if GROQ_API_KEY:
        client = AsyncOpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        extract_model = GROQ_MODEL
        provider_name = "Groq"
    else:
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key:
            raise Exception("Neither GROQ_API_KEY nor OPENAI_API_KEY configured - cannot generate solutions")
        client = AsyncOpenAI(api_key=openai_key)
        extract_model = OCR_FALLBACK_MODEL
        provider_name = "OpenAI"

    question_payload = []
    for question in questions:
        options = question.get("options") or []
        enhanced_options = question.get("enhanced_options") or []
        if not options and enhanced_options:
            options = [
                opt.get("content") if isinstance(opt, dict) else str(opt)
                for opt in enhanced_options
            ]
        question_payload.append({
            "question_id": str(question.get("id") or ""),
            "question_text": question.get("text") or question.get("question_text") or "",
            "options": options,
            "correct_answer": _normalise_correct_answer_label(question.get("correct_answer")),
        })

    prompt = (
        "You are generating worked solutions for objective test-series questions.\n"
        "Each item is already extracted by OCR and reviewed by the tutor before this call.\n\n"
        "For every item:\n"
        "- Use ONLY the provided question text, options, and tutor-selected correct answer label.\n"
        "- First judge whether the selected correct option is actually consistent with the question and options.\n"
        "- If the selected option appears correct, provide a detailed worked explanation leading to that option.\n"
        "- If the selected option appears wrong, ambiguous, or the OCR/options are insufficient, still explain the issue, "
        "set manual_review_required true, and do not pretend the answer is verified.\n"
        "- Do not use the original PDF, screenshots, or any external document context.\n"
        "- Preserve math notation and use clear rich text / markdown-compatible formatting.\n\n"
        "Return ONLY valid JSON, with no markdown fences:\n"
        '{"solutions": ['
        '{"question_id": "id", "answer_text": "worked explanation", "confidence": 0.0, '
        '"correct_option_verified": true, "manual_review_required": false, "notes": ""}'
        "]}\n\n"
        f"Document: {document.get('title') or document.get('document_id')}\n"
        f"Subject: {document.get('subject') or 'General'}\n"
        "--- QUESTION OPTION SETS ---\n"
        f"{json.dumps(question_payload, ensure_ascii=False, default=str)}"
    )

    async def _raw_call():
        return await client.chat.completions.create(
            model=extract_model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=16384,
        )

    if gateway_context:
        gateway = AIGatewayService(
            gateway_context.get("db"),
            is_b2c=bool(gateway_context.get("is_b2c")),
        )
        response = await gateway.call(
            user_id=str(gateway_context.get("user_id") or "unknown"),
            tenant_id=gateway_context.get("tenant_id"),
            document_id=gateway_context.get("document_id"),
            region_id=None,
            region_scope="generated_solution_batch",
            stage="solution_generation_batch",
            provider=provider_name.lower(),
            model=extract_model,
            input_kind="text",
            estimated_input_tokens=estimate_text_tokens(prompt),
            estimated_output_tokens=4096,
            max_output_tokens=16384,
            input_units={"questions": len(question_payload)},
            call_fn=_raw_call,
        )
    else:
        response = await _raw_call()

    raw_response = (response.choices[0].message.content or "").strip()
    if raw_response.startswith("```"):
        raw_response = raw_response.split("\n", 1)[-1]
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3].strip()
    data = _json.loads(raw_response)
    return {
        "solutions": data.get("solutions", []),
        "provider": provider_name.lower(),
        "model": extract_model,
    }


# ══════════════════════════════════════════════════════════════════════════════
# OpenCV + LLM Figure Detection Pipeline — DISABLED
# ══════════════════════════════════════════════════════════════════════════════
#
# WHAT: Two functions that detect figures in exam papers and assign them to
#       the correct questions:
#
#   1. _llm_assign_candidates() — Sends cropped figure images to GPT and asks
#      "which image goes with which question?" Used only when the heuristic
#      (1:1, N:N top-to-bottom) can't decide.
#
#   2. _extract_figures_for_questions() — Full pipeline:
#      - Uses OpenCV (utils/figure_extractor.py) to detect figure regions
#      - Assigns figures to questions via heuristic or LLM
#      - Mutates ocr_result to add images + updates question metadata
#
# WHY DISABLED: With Mistral OCR as the primary OCR provider, this pipeline
#   is no longer needed because:
#   - Mistral extracts images natively with include_image_base64=True
#   - Mistral places image references inline in markdown (![img-N](img-N))
#     right next to the question they belong to
#   - The image-to-question mapping is handled in extract_questions_with_gpt()
#     using the page_image_ids lookup (lines ~1025-1050 above)
#
# WHEN TO RE-ENABLE: If switching back to Sarvam AI or GPT Vision as primary
#   OCR (which don't reliably extract/position images), uncomment the two
#   functions below AND the call site in run_document_ocr_pipeline().
#
# DEPENDS ON: utils/figure_extractor.py (OpenCV module), opencv-python-headless
# ══════════════════════════════════════════════════════════════════════════════

# async def _llm_assign_candidates(
#     questions: List[ExtractedQuestion],
#     candidates: List,  # List[FigureCandidate] — imported at call site
#     page_index: int,
# ) -> Dict[str, Any]:
#     """
#     Use GPT to assign figure candidates to questions on ambiguous pages.
#
#     Sends candidate crop thumbnails (detail="low") + question list and asks the
#     model to return {assignments: [{qid, candidate_id}]} — just mapping, no
#     bounding-box coordinates.
#
#     Returns dict mapping question id → FigureCandidate.
#     On any failure returns {} (non-fatal — questions just lack figures).
#     """
#     import json as _json
#     from openai import AsyncOpenAI
#
#     openai_key = os.getenv("OPENAI_API_KEY", "")
#     if not openai_key:
#         return {}
#
#     client = AsyncOpenAI(api_key=openai_key)
#
#     content_blocks: List[Dict[str, Any]] = []
#
#     for cand in candidates:
#         if not cand.crop_b64:
#             continue
#         content_blocks.append({"type": "text", "text": f"Candidate {cand.candidate_id}:"})
#         content_blocks.append({"type": "image_url", "image_url": {
#             "url": f"data:image/png;base64,{cand.crop_b64}", "detail": "low"
#         }})
#
#     if not content_blocks:
#         return {}
#
#     image_count = sum(1 for b in content_blocks if b.get("type") == "image_url")
#     if image_count > 8:
#         limited = []
#         kept = 0
#         for b in content_blocks:
#             limited.append(b)
#             if b.get("type") == "image_url":
#                 kept += 1
#                 if kept >= 8:
#                     break
#         content_blocks = limited
#
#     q_list_text = "Questions that need figures:\n"
#     for q in questions:
#         q_short = q.text[:200].replace('\n', ' ')
#         q_list_text += f"- qid={q.id}: {q_short}\n"
#
#     cand_ids = [c.candidate_id for c in candidates if c.crop_b64]
#     prompt_text = (
#         f"{q_list_text}\nAvailable candidate IDs: {cand_ids}\n\n"
#         "For each question, decide which candidate image (if any) is its "
#         "diagram/figure/graph/circuit.\n"
#         "Return ONLY valid JSON:\n"
#         '{"assignments": [{"qid": "...", "candidate_id": "..."}]}\n'
#     )
#     content_blocks.append({"type": "text", "text": prompt_text})
#
#     try:
#         response = await client.chat.completions.create(
#             model=OCR_FALLBACK_MODEL,
#             messages=[{"role": "user", "content": content_blocks}],
#             max_completion_tokens=512,
#         )
#         raw = (response.choices[0].message.content or "").strip()
#     except Exception as e:
#         return {}
#
#     if not raw:
#         return {}
#
#     if raw.startswith("```"):
#         raw = raw.split("\n", 1)[-1]
#         if raw.endswith("```"):
#             raw = raw[:-3].strip()
#
#     try:
#         data = _json.loads(raw)
#     except _json.JSONDecodeError:
#         return {}
#
#     cand_lookup = {c.candidate_id: c for c in candidates}
#     valid_qids = {q.id for q in questions}
#     result = {}
#     for assignment in data.get("assignments", []):
#         qid = assignment.get("qid", "")
#         cid = assignment.get("candidate_id", "")
#         if qid and cid and cid in cand_lookup and qid in valid_qids:
#             result[qid] = cand_lookup[cid]
#     return result


# async def _extract_figures_for_questions(
#     ocr_result: Dict[str, Any],
#     extracted_questions: List[ExtractedQuestion],
#     document_id: str,
#     pdf_bytes: Optional[bytes] = None,
# ) -> None:
#     """
#     Deterministic figure extraction using OpenCV + heuristic/LLM assignment.
#
#     Pipeline:
#     1. Group questions needing figures by page
#     2. Render pages on demand if page_render missing (Sarvam path)
#     3. Detect figure candidates with OpenCV on relevant pages
#     4. Assign candidates to questions (heuristic first, LLM for ambiguous)
#
#     Mutates ocr_result in-place: adds cropped images to pages and updates
#     question metadata with image_refs.
#     """
#     pass  # Full implementation preserved in git history


async def run_document_ocr_pipeline(
    document: Dict[str, Any],
    file_content: bytes,
    job_id: str,
    processing_result: Dict[str, Any],
    current_user: Dict[str, Any],
    db: DatabaseManager,
    cache: CacheManager
) -> PDFProcessingResult:
    """Run the full OCR extraction pipeline for a stored document."""
    document_id = document["document_id"]
    try:
        logger.info(f"Calling OCR for job {job_id}")
        ocr_result = await call_sarvam_ocr(
            file_content,
            gateway_context=_build_ai_gateway_context(
                current_user=current_user,
                db=db,
                document_id=document_id,
                region_scope="document",
                is_b2c=is_b2c_admin(current_user),
            ),
        )

        # Mistral OCR's image extraction is unreliable for many Word/Pages-generated
        # PDFs (it returns 0 images even when the PDF has real embedded raster
        # diagrams). Augment with PyMuPDF, which reads embedded Image XObjects
        # directly. Runs in a thread pool because PyMuPDF is synchronous.
        await asyncio.get_event_loop().run_in_executor(
            None, _augment_ocr_with_pymupdf, ocr_result, file_content
        )

        processing_result["progress"] = 60
        await cache.set(f"pdf_job:{job_id}", processing_result, 3600, "admin")

        document_type = document.get("document_type", "Chapter Notes")
        logger.info(f"Extracting questions from OCR result for job {job_id}, document_type: {document_type}")

        # Delete old questions/images for this document before re-processing
        is_b2c_pre = is_b2c_admin(current_user)
        if is_b2c_pre:
            old_q = await db.b2c_delete_many("questions", {"document_id": document_id})
            old_i = await db.b2c_delete_many("images", {"source_pdf": document.get("filename", "")})
        else:
            old_q = await db.mongo_delete_many("questions", {"document_id": document_id})
            old_i = await db.mongo_delete_many("images", {"source_pdf": document.get("filename", "")})
        print(f"[OCR-PIPELINE] Cleaned up old data for {document_id}: questions={old_q}, images={old_i}", flush=True)

        # For Practice Sets, don't extract options separately - keep them in question text
        skip_option_extraction = document_type == "Practice Sets"

        extracted_questions = await extract_questions_with_gpt(
            ocr_result,
            document.get("subject", "General"),
            document.get("difficulty", "medium"),
            skip_option_extraction=skip_option_extraction,
            gateway_context=_build_ai_gateway_context(
                current_user=current_user,
                db=db,
                document_id=document_id,
                region_scope="document",
                is_b2c=is_b2c_pre,
            ),
        )
        
        if skip_option_extraction:
            logger.info(f"📝 Practice Sets mode: Options kept inline with question text")

        # Per-question figure extraction — DISABLED with Mistral OCR.
        # Mistral OCR already extracts images and places them inline in markdown
        # (e.g. ![img-2-0](img-2-0)) next to the questions they belong to.
        # The image-to-question assignment is handled above in extract_questions_with_gpt()
        # at lines 1047-1050 using page_image_ids mapping.
        #
        # The OpenCV + LLM pipeline below was needed when using Sarvam/GPT Vision OCR
        # which did NOT reliably extract images or associate them with questions.
        # To re-enable (e.g. if switching back to Sarvam), uncomment the two lines below.
        #
        # if document_type in ["Practice Sets", "Test Series"]:
        #     await _extract_figures_for_questions(ocr_result, extracted_questions, document_id, file_content)

        print(f"[OCR-PIPELINE] Processing embedded images...", flush=True)

        all_images: List[Dict[str, Any]] = []
        image_base64_map: Dict[str, Dict[str, Any]] = {}

        # Determine if B2C admin
        is_b2c = is_b2c_admin(current_user)

        for page in ocr_result.get("pages", []):
            for img in page.get("images", []):
                if img.get("image_base64"):
                    try:
                        saved_images = await save_image_to_disk(
                            img["image_base64"],
                            img["id"],
                            document["filename"],
                            db,
                            current_user.get("user_id"),
                            split_composite=True,
                            is_b2c=is_b2c
                        )
                        if saved_images:
                            all_images.extend(saved_images)
                            for saved_img in saved_images:
                                image_base64_map[img["id"]] = {
                                    "image_base64": img.get("image_base64", ""),
                                    "top_left_x": img.get("top_left_x", 0),
                                    "top_left_y": img.get("top_left_y", 0),
                                    "bottom_right_x": img.get("bottom_right_x", 0),
                                    "bottom_right_y": img.get("bottom_right_y", 0),
                                    "page": page.get("index", 0)
                                }
                                if saved_img["id"] != img["id"]:
                                    image_base64_map[saved_img["id"]] = image_base64_map[img["id"]]
                    except Exception as img_err:
                        print(f"[OCR-PIPELINE] Warning: Failed to save image {img.get('id')}: {img_err}", flush=True)

        print(f"[OCR-PIPELINE] Saved {len(all_images)} images", flush=True)

        processing_result["progress"] = 80
        processing_result["extracted_questions"] = len(extracted_questions)
        processing_result["extracted_images"] = len(all_images)
        await cache.set(f"pdf_job:{job_id}", processing_result, 3600, "admin")

        print(f"[OCR-PIPELINE] Storing {len(extracted_questions)} questions for {document_type}...", flush=True)

        for question in extracted_questions:
            if document_type in ["Practice Sets", "Test Series"]:
                page_index = question.metadata.get('page', 0)
                image_refs = question.metadata.get('image_refs', [])
                question_image_refs = question.metadata.get('question_image_refs', [])
                page_images: List[Dict[str, Any]] = []
                question_figures: List[Dict[str, Any]] = []

                logger.info(
                    f"Question {question.id} references {len(image_refs)} total images "
                    f"({len(question_image_refs)} question figures)"
                )

                if image_refs:
                    # Iterate ALL pages, not just the question's "home" page —
                    # multi-page case studies (e.g. a parabola case study with
                    # diagrams spread across two PDF pages) will have image_refs
                    # pointing at images that live on a different page than the
                    # question's nominal start page.
                    for page in ocr_result.get("pages", []):
                        actual_page_idx = page.get("index", page_index)
                        for ocr_img in page.get("images", []):
                            ocr_img_id = ocr_img.get('id')
                            base_img_id = ocr_img_id.split('.')[0] if '.' in ocr_img_id else ocr_img_id

                            is_referenced = any(
                                base_img_id in ref or ocr_img_id in ref
                                for ref in image_refs
                            )

                            if not is_referenced:
                                logger.debug(f"Skipping non-referenced image {ocr_img_id}")
                                continue

                            logger.info(f"Including {ocr_img_id} - referenced in question")

                            # Find saved images - check for both exact match and split variants
                            # If image was split, the IDs become img-X-A, img-X-B, etc.
                            matching_saved_images = [
                                img for img in all_images
                                if img['id'] == base_img_id or img['id'].startswith(f"{base_img_id}-")
                            ]

                            img_base64_data = image_base64_map.get(ocr_img_id) or image_base64_map.get(base_img_id, {})

                            if matching_saved_images and img_base64_data:
                                is_question_figure = any(
                                    base_img_id in ref or ocr_img_id in ref
                                    for ref in question_image_refs
                                )

                                is_image_based_mcq = question.metadata.get("is_image_based_mcq", False)
                                if is_image_based_mcq and not is_question_figure:
                                    is_question_figure = False
                                    logger.info(f"Treating {ocr_img_id} as option image for image-based MCQ")

                                # For question figures, use the first image (or unsplit original)
                                # For option images (image-based MCQ), include all split parts
                                if is_question_figure:
                                    # For question diagrams, prefer the original (unsplit) image
                                    saved_img = next(
                                        (img for img in matching_saved_images if img.get('is_original', False)),
                                        next(
                                            (img for img in matching_saved_images if img['id'] == base_img_id),
                                            matching_saved_images[0]  # Fall back to first part
                                        )
                                    )
                                    image_obj = {
                                        'id': saved_img['id'],
                                        'filename': saved_img['filename'],
                                        'path': saved_img['path'],
                                        'base64Data': img_base64_data.get('image_base64', ''),
                                        'description': '',
                                        'type': 'diagram',
                                        'bbox': {
                                            'top_left_x': img_base64_data.get('top_left_x', 0),
                                            'top_left_y': img_base64_data.get('top_left_y', 0),
                                            'bottom_right_x': img_base64_data.get('bottom_right_x', 0),
                                            'bottom_right_y': img_base64_data.get('bottom_right_y', 0)
                                        },
                                        'metadata': {
                                            'source': 'sarvam_ocr',
                                            'page': actual_page_idx,
                                            'extractedAt': datetime.utcnow().isoformat()
                                        }
                                    }
                                    question_figures.append(image_obj)
                                    logger.info(f"✅ Added question figure: {saved_img['id']}")
                                else:
                                    # For option images, prefer split parts over original
                                    # Filter to only use split parts if available
                                    split_images = [img for img in matching_saved_images if not img.get('is_original', True)]
                                    images_to_use = split_images if split_images else matching_saved_images

                                    for saved_img in images_to_use:
                                        # Get base64 data for this specific split part if available
                                        split_base64_data = image_base64_map.get(saved_img['id'], img_base64_data)
                                        image_obj = {
                                            'id': saved_img['id'],
                                            'filename': saved_img['filename'],
                                            'path': saved_img['path'],
                                            'base64Data': split_base64_data.get('image_base64', ''),
                                            'description': '',
                                            'type': 'diagram',
                                            'bbox': {
                                                'top_left_x': split_base64_data.get('top_left_x', 0),
                                                'top_left_y': split_base64_data.get('top_left_y', 0),
                                                'bottom_right_x': split_base64_data.get('bottom_right_x', 0),
                                                'bottom_right_y': split_base64_data.get('bottom_right_y', 0)
                                            },
                                            'metadata': {
                                                'source': 'sarvam_ocr',
                                                'page': actual_page_idx,
                                                'extractedAt': datetime.utcnow().isoformat()
                                            }
                                        }
                                        page_images.append(image_obj)
                                    logger.info(f"✅ Added {len(images_to_use)} option images from {base_img_id}")
                            else:
                                # Log why the image wasn't matched
                                if not matching_saved_images:
                                    logger.warning(f"⚠️ Image {base_img_id} not found in all_images (available: {[img['id'] for img in all_images[:10]]}...)")
                                if not img_base64_data:
                                    logger.warning(f"⚠️ Image {ocr_img_id} not found in image_base64_map (keys: {list(image_base64_map.keys())[:10]}...)")

                logger.info(
                    f"Associated {len(question_figures)} question figures and "
                    f"{len(page_images)} option images with question {question.id}"
                )

                enhanced_options = []
                is_image_based_mcq = question.metadata.get("is_image_based_mcq", False)

                if is_image_based_mcq and page_images:
                    logger.info(
                        f"Creating image-based MCQ options: {len(page_images)} images for question {question.id}"
                    )
                    for idx, img in enumerate(page_images):
                        option_label = chr(65 + idx)
                        enhanced_options.append({
                            'id': f"{question.id}_opt_{idx}",
                            'type': 'image',
                            'content': img.get('base64Data', ''),
                            'label': option_label,
                            'description': img.get('description', ''),
                            'image_id': img.get('id', ''),
                            'metadata': img.get('metadata', {})
                        })
                    logger.info(f"Created {len(enhanced_options)} image-based MCQ options")
                else:
                    logger.info(f"Non image-based MCQ: building text options for question {question.id}")
                    for idx, option_text in enumerate(question.options):
                        option_label = chr(65 + idx)  # A, B, C, D, etc.
                        enhanced_options.append({
                            'id': f"{question.id}_opt_{idx}",
                            'type': 'text',
                            'content': option_text,
                            'label': option_label,
                            'description': ''
                        })

                question_doc = {
                    "id": question.id,
                    "text": question.text,
                    "subject": document.get("subject", "General"),
                    "difficulty": document.get("difficulty", "medium"),
                    "question_type": document.get("question_type", "mcq"),
                    "document_type": document_type,
                    "extracted_at": datetime.utcnow(),
                    "pdf_source": document["filename"],
                    "document_id": document_id,
                    "images": page_images,
                    "question_figures": question_figures,
                    "options": question.options,
                    "enhanced_options": enhanced_options,
                    "correct_answer": question.correct_answer,
                    "is_image_based_mcq": question.metadata.get("is_image_based_mcq", False),
                    "metadata": question.metadata,
                    "points": question.points if hasattr(question, 'points') else 1.0,
                    "penalty": question.penalty if hasattr(question, 'penalty') else 0.0,
                    "created_by": current_user.get("user_id"),
                    "created_at": datetime.utcnow()
                }
            else:
                logger.info(f"Using simple extraction for {document_type} - no image association")
                enhanced_options = []
                for idx, option_text in enumerate(question.options):
                    option_label = chr(65 + idx)  # A, B, C, D, etc.
                    enhanced_options.append({
                        'id': f"{question.id}_opt_{idx}",
                        'type': 'text',
                        'content': option_text,
                        'label': option_label,
                        'description': ''
                    })

                question_doc = {
                    "id": question.id,
                    "text": question.text,
                    "subject": document.get("subject", "General"),
                    "difficulty": document.get("difficulty", "medium"),
                    "question_type": document.get("question_type", "mcq"),
                    "document_type": document_type,
                    "extracted_at": datetime.utcnow(),
                    "pdf_source": document["filename"],
                    "document_id": document_id,
                    "images": [],
                    "question_figures": [],
                    "options": question.options,
                    "enhanced_options": enhanced_options,
                    "correct_answer": question.correct_answer,
                    "metadata": question.metadata,
                    "points": question.points if hasattr(question, 'points') else 1.0,
                    "penalty": question.penalty if hasattr(question, 'penalty') else 0.0,
                    "created_by": current_user.get("user_id"),
                    "created_at": datetime.utcnow()
                }

            # Save question to appropriate database (B2C or regular)
            try:
                if is_b2c:
                    await db.b2c_insert_one("questions", question_doc)
                else:
                    await db.mongo_insert_one("questions", question_doc)
            except Exception as db_err:
                print(f"[OCR-PIPELINE] ERROR saving question {question.id}: {db_err}", flush=True)

        print(f"[OCR-PIPELINE] All {len(extracted_questions)} questions stored in DB", flush=True)

        # Check if B2C admin for database routing
        is_b2c = is_b2c_admin(current_user)
        
        if is_b2c:
            document_fresh = await db.b2c_find_one("documents", {"document_id": document_id})
        else:
            document_fresh = await db.mongo_find_one("documents", {"document_id": document_id})
        total_calculated_points = sum(
            q.points if hasattr(q, 'points') and q.points else 1.0
            for q in extracted_questions
        )

        update_data = {
            "ocr_status": "completed",
            "extracted_questions_count": len(extracted_questions),
            "extracted_images_count": len(all_images),
            "ocr_completed_at": datetime.utcnow()
        }

        if document_fresh and document_fresh.get("document_type") == "Test Series":
            existing_total = document_fresh.get("total_points")
            if existing_total is None or existing_total == 0:
                update_data["total_points"] = total_calculated_points
                logger.info(f"Auto-calculated total_points for {document_id}: {total_calculated_points}")

        if is_b2c:
            await db.b2c_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": update_data}
            )
        else:
            await db.mongo_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": update_data}
            )

        processing_result["status"] = "completed"
        processing_result["progress"] = 100
        processing_result["pages"] = ocr_result.get("pages", [])
        await cache.set(f"pdf_job:{job_id}", processing_result, 3600, "admin")

        print(f"[OCR-PIPELINE] DONE! {document_id}: {len(extracted_questions)} questions, {len(all_images)} images", flush=True)
        return PDFProcessingResult(**processing_result)
    except Exception as exc:
        print(f"[OCR-PIPELINE] FAILED for {document_id}: {type(exc).__name__}: {exc}", flush=True)
        logger.error(f"OCR pipeline failed for document {document_id}: {exc}", exc_info=True)
        # Check if B2C admin for error update
        is_b2c = is_b2c_admin(current_user)
        if is_b2c:
            await db.b2c_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": {"ocr_status": "error"}}
            )
        else:
            await db.mongo_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": {"ocr_status": "error"}}
            )

        error_result = {
            "job_id": job_id,
            "status": "error",
            "progress": 100,
            "error": str(exc),
            "timestamp": datetime.utcnow()
        }
        await cache.set(f"pdf_job:{job_id}", error_result, 3600, "admin")
        raise

class DocumentMetadata(BaseModel):
    document_id: str
    title: str
    document_type: str
    subject: str
    difficulty: str
    course_plan: Optional[str] = None
    standard: Optional[str] = None
    section: Optional[str] = None  # Section A-F for filtering
    teacher_ids: Optional[List[str]] = None  # Array of teacher IDs for filtering
    file_path: str
    filename: str
    uploaded_by: str
    uploaded_by_name: Optional[str] = None  # Display name of who uploaded (Admin or teacher username)
    uploaded_at: datetime
    ocr_status: str
    ocr_job_id: Optional[str] = None
    extracted_questions_count: int = 0
    extracted_images_count: int = 0
    pages_count: int = 0  # Number of pages in the PDF (for Notes display)
    total_points: Optional[float] = None  # Total points for Test Series documents
    total_minutes: Optional[int] = None  # Total minutes for Test Series documents
    file_exists: bool = True  # Whether the physical file exists on disk
    is_active: bool = True  # Whether the document is enabled for students
    instructions: Optional[str] = None
    exam_mode: Optional[str] = None
    exam_template_path: Optional[str] = None
    answer_sheet_path: Optional[str] = None
    answer_sheet_filename: Optional[str] = None
    answer_sheet_uploaded_at: Optional[datetime] = None
    answer_sheet_pages_count: Optional[int] = None
    answer_sheet_ocr_status: Optional[str] = None
    answer_sheet_ocr_job_id: Optional[str] = None
    answer_sheet_ocr_completed_at: Optional[datetime] = None
    answer_sheet_processed_regions_count: Optional[int] = None
    answer_sheet_mapped_answers_count: Optional[int] = None
    has_answer_sheet: bool = False
    exam_finalized: Optional[bool] = None
    exam_finalized_at: Optional[datetime] = None
    exam_sync_summary: Optional[Dict[str, Any]] = None
    orientation_applied: Optional[int] = None  # Rotation degrees baked into the uploaded PDF at upload time (0/90/180/270). Audit-only — file is already pre-rotated.
    exam_template_orientation_applied: Optional[int] = None  # Same as above for the DCR answer template.
    tally_num_questions: Optional[int] = None
    tally_max_marks_per_question: Optional[float] = None
    tally_marking_scheme: Optional[List[Dict[str, float]]] = None
    tally_validate_paper_set: Optional[bool] = None
    tally_expected_paper_set: Optional[str] = None

class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]
    total: int
    page: int
    limit: int


def _document_file_candidates(document: Dict[str, Any]) -> List[Path]:
    backend_dir = Path(os.getcwd())
    stored_path = str(document.get("file_path", "") or "").replace("\\", "/")
    document_id = str(document.get("document_id", "") or "")
    document_type = str(document.get("document_type", "") or "")
    candidates: List[Path] = []

    if stored_path:
        path = Path(stored_path)
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(backend_dir / stored_path)

        if "uploads/" in stored_path:
            try:
                uploads_index = stored_path.index("uploads/")
                candidates.append(backend_dir / stored_path[uploads_index:])
            except ValueError:
                pass

    if document_id and document_type:
        candidates.append(backend_dir / "uploads" / "documents" / document_type / f"{document_id}.pdf")

    return candidates


def _answer_sheet_file_candidates(document: Dict[str, Any]) -> List[Path]:
    backend_dir = Path(os.getcwd())
    stored_path = str(document.get("answer_sheet_path", "") or "").replace("\\", "/")
    document_id = str(document.get("document_id", "") or "")
    document_type = str(document.get("document_type", "") or "")
    candidates: List[Path] = []

    if stored_path:
        path = Path(stored_path)
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(backend_dir / stored_path)

        if "uploads/" in stored_path:
            try:
                uploads_index = stored_path.index("uploads/")
                candidates.append(backend_dir / stored_path[uploads_index:])
            except ValueError:
                pass

    if document_id and document_type:
        candidates.append(
            backend_dir
            / "uploads"
            / "documents"
            / document_type
            / "answer_sheets"
            / f"{document_id}_answer_sheet.pdf"
        )

    return candidates


def _resolve_document_file_path(document: Dict[str, Any]) -> Optional[Path]:
    for candidate in _document_file_candidates(document):
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def _resolve_answer_sheet_file_path(document: Dict[str, Any]) -> Optional[Path]:
    for candidate in _answer_sheet_file_candidates(document):
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def _document_file_exists(document: Dict[str, Any]) -> bool:
    stored_path = str(document.get("file_path", "") or "").replace("\\", "/")
    if stored_path.startswith("s3://"):
        return True
    return _resolve_document_file_path(document) is not None


@router.post("/upload")
@limiter.limit("10/minute")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    exam_template: Optional[UploadFile] = File(None),
    answer_sheet: Optional[UploadFile] = File(None),
    document_id: str = Form(...),
    title: str = Form(...),
    document_type: str = Form(...),
    subject: str = Form(...),
    difficulty: Optional[str] = Form("medium"),
    course_plan: Optional[str] = Form("CBSE"),
    standard: Optional[str] = Form("11"),
    section: Optional[str] = Form(None),  # Section A-F for filtering
    teacher_ids: Optional[str] = Form(None),  # Comma-separated teacher IDs for filtering
    total_points: Optional[float] = Form(None),
    total_minutes: Optional[int] = Form(None),
    question_type: Optional[str] = Form(None),  # "mcq" or "subjective" - default type for all questions
    instructions: Optional[str] = Form(None),  # Paper instructions for practice/test
    exam_mode: Optional[str] = Form(None),  # "dcr" or "pcr" — offline exam conduction mode
    answer_solution_mode: Optional[str] = Form(None),  # none, upload, or auto
    tally_num_questions: Optional[int] = Form(None),
    tally_max_marks_per_question: Optional[float] = Form(None),
    tally_marking_scheme: Optional[str] = Form(None),
    tally_validate_paper_set: Optional[bool] = Form(None),
    tally_expected_paper_set: Optional[str] = Form(None),
    orientation_applied: Optional[int] = Form(None),  # Rotation (deg) the client baked into the PDF — audit only
    exam_template_orientation_applied: Optional[int] = Form(None),  # Same for the DCR answer template
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """
    Upload PDF file and save metadata (without OCR processing)

    - Accepts PDF file upload
    - Validates document_id (alphanumeric only, no duplicates)
    - Saves file to appropriate folder based on document_type
    - Stores metadata in MongoDB
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )
        if answer_sheet is not None and not (answer_sheet.filename or "").endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported for answer sheet uploads"
            )

        # Validate document_id (alphanumeric only, no spaces or special chars)
        if not document_id.isalnum():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document ID must be alphanumeric only (no spaces or special characters)"
            )

        # Check if B2C admin
        is_b2c = is_b2c_admin(current_user)
        
        # Check for duplicate document_id in appropriate database
        if is_b2c:
            existing_doc = await db.b2c_find_one("documents", {"document_id": document_id})
        else:
            existing_doc = await db.mongo_find_one("documents", {"document_id": document_id})
        if existing_doc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Document ID '{document_id}' already exists"
            )

        # Validate document_type
        allowed_types = ["Practice Sets", "Test Series", "Chapter Notes"]
        if document_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid document type. Allowed: {', '.join(allowed_types)}"
            )

        # Validate exam_mode
        if exam_mode:
            if exam_mode not in ("dcr", "pcr"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid exam mode. Allowed: dcr, pcr"
                )
            if document_type == "Chapter Notes":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Exam mode is not allowed for Chapter Notes"
                )
            if exam_mode == "dcr" and question_type == "subjective":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="DCR documents must use Objective question type",
                )
            if exam_mode == "dcr" and exam_template is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="DCR documents require an answer template during upload",
                )
        elif exam_template is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Answer template upload is only allowed for DCR exam documents",
            )

        if answer_sheet is not None:
            if document_type != "Test Series" or exam_mode or question_type == "subjective":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Answer sheet upload is only allowed for online objective Test Series documents",
                )

        answer_solution_mode = (answer_solution_mode or ("upload" if answer_sheet is not None else "none")).strip().lower()
        if answer_solution_mode not in {"none", "upload", "auto"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="answer_solution_mode must be one of: none, upload, auto",
            )
        if answer_solution_mode == "upload" and answer_sheet is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Upload mode requires an answer sheet PDF",
            )
        if answer_solution_mode == "auto":
            if document_type != "Test Series" or exam_mode or question_type == "subjective":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Auto-generated solutions are only allowed for online objective Test Series documents",
                )

        # Validate title length
        if len(title) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document title must not exceed 100 characters"
            )

        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Count PDF pages using pypdf (already in requirements.txt)
        pages_count = 0
        try:
            import io
            from pypdf import PdfReader
            pdf_reader = PdfReader(io.BytesIO(file_content))
            pages_count = len(pdf_reader.pages)
            logger.info(f"PDF {document_id} has {pages_count} pages")
        except Exception as pdf_err:
            logger.warning(f"Failed to count PDF pages for {document_id}: {pdf_err}")

        logger.info(f"Uploading document: {document_id}, Title: {title}, Type: {document_type}, Size: {file_size} bytes")

        # Create folder structure based on document type
        # Use Path for consistent path handling across Windows/Linux
        from pathlib import Path
        backend_dir = Path(os.getcwd())
        upload_dir = backend_dir / "uploads" / "documents" / document_type
        file_path = upload_dir / f"{document_id}.pdf"
        
        # Store relative path with forward slashes (universal format)
        local_relative_path = f"uploads/documents/{document_type}/{document_id}.pdf"

        # Use S3 storage if enabled, otherwise save locally
        if is_s3_enabled():
            # Upload PDF to S3
            success, storage_path = await s3_upload_file(
                file_data=file_content,
                local_path=str(file_path),
                content_type="application/pdf"
            )
            if success:
                relative_path = storage_path  # s3://bucket/documents/...
                logger.info(f"✅ Uploaded PDF to S3: {storage_path}")
            else:
                # Fallback to local if S3 fails
                logger.warning("S3 upload failed, falling back to local storage")
                upload_dir.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(str(file_path), "wb") as f:
                    await f.write(file_content)
                relative_path = local_relative_path
        else:
            # Save file locally
            upload_dir.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(str(file_path), "wb") as f:
                await f.write(file_content)
            relative_path = local_relative_path
            logger.info(f"Saved PDF locally: {file_path}")

        # Validate total_points for Test Series
        if document_type == "Test Series" and total_points is not None:
            if total_points <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Total points must be greater than 0"
                )

        # Validate total_minutes for Test Series
        if document_type == "Test Series" and total_minutes is not None:
            if total_minutes <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Total minutes must be greater than 0"
                )

        # Parse teacher_ids from comma-separated string to list
        teacher_ids_list = []
        if teacher_ids:
            teacher_ids_list = [tid.strip() for tid in teacher_ids.split(",") if tid.strip()]

        tally_marking_scheme_list: Optional[List[Dict[str, float]]] = None
        if tally_marking_scheme:
            try:
                parsed_scheme = json.loads(tally_marking_scheme)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid tally marking scheme JSON",
                )
            if not isinstance(parsed_scheme, list):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Tally marking scheme must be a list",
                )
            tally_marking_scheme_list = []
            for item in parsed_scheme:
                if not isinstance(item, dict):
                    continue
                try:
                    from_q = int(item.get("from"))
                    to_q = int(item.get("to"))
                    marks = float(item.get("marks"))
                except (TypeError, ValueError):
                    continue
                if from_q > 0 and to_q >= from_q and marks > 0:
                    tally_marking_scheme_list.append(
                        {"from": from_q, "to": to_q, "marks": marks}
                    )

        # If a tutor is uploading, ensure their ID is in teacher_ids
        if current_user.get("user_type") == "tutor":
            tutor_id = current_user.get("tutor_id") or current_user.get("user_id")
            if tutor_id and str(tutor_id) not in teacher_ids_list:
                teacher_ids_list.append(str(tutor_id))

        # Create document metadata
        # Attach tenant context — tutors use their admin's ID for multi-tenancy
        try:
            if current_user.get("user_type") == "tutor" and current_user.get("admin_id"):
                admin_oid = BsonObjectId(current_user["admin_id"])
            else:
                admin_oid = BsonObjectId(current_user.get("user_id"))
        except Exception:
            admin_oid = None

        exam_template_path = None
        if exam_mode == "dcr" and exam_template is not None:
            exam_template_path = await _store_exam_template_file(
                document_id=document_id,
                upload=exam_template,
            )

        answer_sheet_path = None
        answer_sheet_filename = None
        answer_sheet_file_size = None
        answer_sheet_pages_count = None
        answer_sheet_uploaded_at = None
        if answer_sheet is not None:
            answer_sheet_content = await answer_sheet.read()
            answer_sheet_file_size = len(answer_sheet_content)
            answer_sheet_filename = answer_sheet.filename
            answer_sheet_uploaded_at = datetime.utcnow()

            try:
                import io
                from pypdf import PdfReader
                answer_sheet_reader = PdfReader(io.BytesIO(answer_sheet_content))
                answer_sheet_pages_count = len(answer_sheet_reader.pages)
                logger.info(f"Answer sheet for {document_id} has {answer_sheet_pages_count} pages")
            except Exception as pdf_err:
                logger.warning(f"Failed to count answer sheet pages for {document_id}: {pdf_err}")

            answer_sheet_dir = upload_dir / "answer_sheets"
            answer_sheet_file_path = answer_sheet_dir / f"{document_id}_answer_sheet.pdf"
            local_answer_sheet_relative_path = (
                f"uploads/documents/{document_type}/answer_sheets/{document_id}_answer_sheet.pdf"
            )

            if is_s3_enabled():
                success, storage_path = await s3_upload_file(
                    file_data=answer_sheet_content,
                    local_path=str(answer_sheet_file_path),
                    content_type="application/pdf",
                )
                if success:
                    answer_sheet_path = storage_path
                    logger.info(f"✅ Uploaded answer sheet to S3: {storage_path}")
                else:
                    logger.warning("S3 answer sheet upload failed, falling back to local storage")
                    answer_sheet_dir.mkdir(parents=True, exist_ok=True)
                    async with aiofiles.open(str(answer_sheet_file_path), "wb") as f:
                        await f.write(answer_sheet_content)
                    answer_sheet_path = local_answer_sheet_relative_path
            else:
                answer_sheet_dir.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(str(answer_sheet_file_path), "wb") as f:
                    await f.write(answer_sheet_content)
                answer_sheet_path = local_answer_sheet_relative_path
                logger.info(f"Saved answer sheet locally: {answer_sheet_file_path}")

        document_metadata = {
            "document_id": document_id,
            "title": title,
            "document_type": document_type,
            "subject": subject or "General",
            "difficulty": difficulty or "medium",
            "course_plan": course_plan or "CBSE",
            "standard": standard or "11",
            "section": section,  # Section A-F for filtering
            "teacher_ids": teacher_ids_list,  # Array of teacher IDs for filtering
            "file_path": relative_path,
            "filename": file.filename,
            "file_size": file_size,
            "uploaded_by": current_user.get("user_id"),
            "admin_id": admin_oid,
            "uploaded_at": datetime.utcnow(),
            "ocr_status": "completed" if document_type == "Chapter Notes" else "not_processed",
            "ocr_job_id": None,
            "extracted_questions_count": 0,
            "extracted_images_count": 0,
            "pages_count": pages_count,  # Store page count for Notes display
            "total_points": total_points if document_type == "Test Series" else None,
            "total_minutes": total_minutes if document_type == "Test Series" else None,
            "is_validated": False,
            "question_type": (
                "mcq"
                if exam_mode == "dcr"
                else question_type if question_type in ["mcq", "subjective"] else "mcq"
            ),  # DCR papers are objective-only by contract
            "instructions": instructions.strip() if instructions else None,  # Paper instructions
            "is_active": False,  # Default to inactive until admin enables
            "is_s3": is_s3_enabled(),  # Track storage location
            "exam_mode": exam_mode if exam_mode in ("dcr", "pcr") else None,
            "exam_template_path": exam_template_path,
            "answer_sheet_path": answer_sheet_path,
            "answer_sheet_filename": answer_sheet_filename,
            "answer_sheet_file_size": answer_sheet_file_size,
            "answer_sheet_uploaded_at": answer_sheet_uploaded_at,
            "answer_sheet_pages_count": answer_sheet_pages_count,
            "answer_sheet_ocr_status": "not_processed" if answer_sheet_path else None,
            "answer_sheet_ocr_job_id": None,
            "answer_sheet_mapped_answers_count": 0,
            "answer_solution_mode": answer_solution_mode,
            "generated_solutions_status": "not_generated" if answer_solution_mode == "auto" else None,
            "generated_solutions_count": 0,
            "exam_finalized": False,
            "exam_finalized_at": None,
            "exam_sync_summary": None,
            "tally_num_questions": tally_num_questions,
            "tally_max_marks_per_question": tally_max_marks_per_question,
            "tally_marking_scheme": tally_marking_scheme_list,
            "tally_validate_paper_set": tally_validate_paper_set,
            "tally_expected_paper_set": (
                tally_expected_paper_set.strip() if tally_expected_paper_set else None
            ),
            "orientation_applied": (
                orientation_applied if orientation_applied in (0, 90, 180, 270) else None
            ),
            "exam_template_orientation_applied": (
                exam_template_orientation_applied
                if exam_template_orientation_applied in (0, 90, 180, 270)
                else None
            ),
        }

        # Save to appropriate MongoDB database (B2C or regular)
        if is_b2c:
            await db.b2c_insert_one("documents", document_metadata)
        else:
            await db.mongo_insert_one("documents", document_metadata)

        logger.info(f"Document {document_id} uploaded successfully to {'B2C' if is_b2c else 'regular'} database")
        
        # NOTE: Auto-OCR has been disabled to allow manual question segmentation
        # The admin can now:
        # 1. Preview the PDF and draw bounding boxes around each question
        # 2. Use the "Segment" button to manually define question regions
        # 3. Then trigger OCR which will process each region individually
        # This gives better control over question extraction, especially for complex PDFs
        
        # Document starts with 'not_processed' status - admin must manually trigger OCR
        ocr_status = "not_processed"
        ocr_job_id = None

        return {
            "message": (
                "DCR document uploaded successfully. OCR was not started."
                if exam_mode == "dcr"
                else "Document uploaded successfully. Use 'Segment' to define question regions before processing OCR."
            ),
            "document_id": document_id,
            "file_path": "",
            "answer_sheet_path": None,
            "has_answer_sheet": bool(answer_sheet_path),
            "ocr_status": ocr_status,
            "ocr_job_id": ocr_job_id,
            "pages_count": pages_count,
            "requires_segmentation": False # Auto-OCR/Direct OCR enabled for all types
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {str(e)}"
        )

@router.post("/documents/{document_id}/finalize-exam")
@limiter.limit("5/minute")
async def finalize_exam(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """Finalize an offline exam document for ExamPen evaluation.

    Syncs reviewed questions into ExamPen metadata collections:
      - DCR → exampen_answer_keys (via sync_dcr_answer_keys)
      - PCR → evalpen_questions (via sync_questions_to_exampen)

    After finalization, the document and its questions become read-only
    for exam integrity.
    """
    try:
        is_b2c = is_b2c_admin(current_user)

        # Load document
        if is_b2c:
            doc = await db.b2c_find_one("documents", {"document_id": document_id})
        else:
            doc = await db.mongo_find_one("documents", {"document_id": document_id})

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        exam_mode = doc.get("exam_mode")
        if not exam_mode:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document has no exam_mode set. Only DCR/PCR documents can be finalized.",
            )

        if doc.get("exam_finalized"):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document is already finalized. Cannot re-finalize.",
            )

        if exam_mode != "dcr" and doc.get("ocr_status") != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OCR must be completed before finalizing.",
            )

        # Load questions for this document
        if is_b2c:
            questions_cursor = db.b2c_db["questions"].find({"document_id": document_id})
        else:
            questions_cursor = (await db.get_tenant_db(current_user.get("db_name")))["questions"].find(
                {"document_id": document_id}
            )
        questions = await questions_cursor.to_list(length=10000)

        if not questions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "No answer keys found for this DCR document. Add objective questions/answers first."
                    if exam_mode == "dcr"
                    else "No questions found for this document. Extract questions via OCR first."
                ),
            )

        # Validate and sync based on exam_mode
        sync_summary = {}

        if exam_mode == "dcr":
            # DCR: require answer template
            if not doc.get("exam_template_path"):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "message": "DCR finalization requires an answer template. Upload one before finalizing.",
                        "errors": ["Missing answer template (blank answer sheet for overlay)"],
                    },
                )

            # DCR: all questions must be objective with correct_answer
            errors = []
            for q in questions:
                q_id = q.get("id", "?")
                q_type = q.get("question_type", "mcq")
                if q_type == "subjective":
                    errors.append(f"Q {q_id}: subjective questions not allowed in DCR paper")
                if not q.get("correct_answer"):
                    errors.append(f"Q {q_id}: missing correct answer")

            if errors:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "message": "DCR finalization failed: all questions must be objective with correct answers",
                        "errors": errors[:20],
                    },
                )

            # Sync to exampen_answer_keys
            from api.v1.tutor_async import sync_dcr_answer_keys

            if is_b2c:
                tenant_db = db.b2c_db
            else:
                tenant_db = await db.get_tenant_db(current_user.get("db_name"))

            result = await sync_dcr_answer_keys(
                tenant_db=tenant_db,
                questions=questions,
                exam_id=document_id,
                exam_doc=doc,
            )
            sync_summary = {"engine": "dcr", "answer_keys_upserted": (result or {}).get("upserted", 0)}

        elif exam_mode == "pcr":
            # PCR: sync all questions to evalpen_questions
            from api.v1.tutor_async import sync_questions_to_exampen

            if is_b2c:
                tenant_db = db.b2c_db
            else:
                tenant_db = await db.get_tenant_db(current_user.get("db_name"))

            result = await sync_questions_to_exampen(
                tenant_db=tenant_db,
                questions=questions,
                exam_id=document_id,
                default_subject=doc.get("subject"),
            )
            sync_summary = {
                "engine": "pcr",
                "questions_inserted": (result or {}).get("inserted", 0),
                "questions_updated": (result or {}).get("updated", 0),
            }

        # Mark document as finalized
        finalized_update = {
            "$set": {
                "exam_finalized": True,
                "exam_finalized_at": datetime.utcnow(),
                "exam_sync_summary": sync_summary,
            }
        }

        if is_b2c:
            await db.b2c_db["documents"].update_one(
                {"document_id": document_id}, finalized_update
            )
        else:
            tenant_db = await db.get_tenant_db(current_user.get("db_name"))
            await tenant_db["documents"].update_one(
                {"document_id": document_id}, finalized_update
            )

        logger.info(f"Document {document_id} finalized as {exam_mode}: {sync_summary}")

        return {
            "message": f"Exam finalized successfully as {exam_mode.upper()}",
            "document_id": document_id,
            "exam_mode": exam_mode,
            "sync_summary": sync_summary,
            "question_count": len(questions),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Finalize exam error for {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to finalize exam: {str(e)}",
        )


async def _store_exam_template_file(document_id: str, upload: UploadFile) -> str:
    """Persist a DCR answer template as a raster overlay asset and return its relative path."""
    filename = upload.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("png", "jpg", "jpeg", "pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Template must be PNG, JPG, or PDF",
        )

    file_content = await upload.read()
    if not file_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file",
        )

    if ext == "pdf":
        try:
            import io
            try:
                from pdf2image import convert_from_bytes
                images = convert_from_bytes(file_content, first_page=1, last_page=1, dpi=200)
            except ImportError:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="PDF answer templates require server-side PDF-to-image conversion support",
                )
            if not images:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Could not extract the first page from the PDF template",
                )
            buf = io.BytesIO()
            images[0].save(buf, format="PNG")
            file_content = buf.getvalue()
            ext = "png"
        except HTTPException:
            raise
        except Exception as pdf_err:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to convert PDF template to image: {pdf_err}",
            )

    from pathlib import Path
    backend_dir = Path(os.getcwd())
    template_dir = backend_dir / "uploads" / "documents" / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)

    template_filename = f"{document_id}_template.{ext}"
    template_path = template_dir / template_filename
    relative_path = f"uploads/documents/templates/{template_filename}"

    import aiofiles
    async with aiofiles.open(str(template_path), "wb") as f:
        await f.write(file_content)

    return relative_path


@router.post("/documents/{document_id}/upload-template")
@limiter.limit("10/minute")
async def upload_exam_template(
    request: Request,
    document_id: str,
    file: UploadFile = File(...),
    orientation_applied: Optional[int] = Form(None),  # Rotation (deg) the client baked into the template
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """Upload a blank answer-sheet template for DCR overlay."""
    try:
        is_b2c = is_b2c_admin(current_user)

        if is_b2c:
            doc = await db.b2c_find_one("documents", {"document_id": document_id})
        else:
            doc = await db.mongo_find_one("documents", {"document_id": document_id})

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        if doc.get("exam_mode") != "dcr":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Answer template upload is only applicable to DCR exam documents",
            )

        if doc.get("exam_finalized"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot modify a finalized exam document",
            )

        relative_path = await _store_exam_template_file(document_id=document_id, upload=file)

        update_set: Dict[str, Any] = {"exam_template_path": relative_path}
        if orientation_applied in (0, 90, 180, 270):
            update_set["exam_template_orientation_applied"] = orientation_applied
        update_op = {"$set": update_set}
        if is_b2c:
            await db.b2c_db["documents"].update_one(
                {"document_id": document_id}, update_op
            )
        else:
            tenant_db = await db.get_tenant_db(current_user.get("db_name"))
            await tenant_db["documents"].update_one(
                {"document_id": document_id}, update_op
            )

        logger.info(f"Answer template uploaded for {document_id}: {relative_path}")

        return {
            "message": "Answer template uploaded successfully",
            "document_id": document_id,
            "template_path": relative_path,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template upload error for {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload template: {str(e)}",
        )


def _ocr_pages_to_plain_text(ocr_result: Dict[str, Any]) -> str:
    parts: List[str] = []
    for page in ocr_result.get("pages", []) or []:
        page_index = page.get("index", 0)
        markdown = str(page.get("markdown") or "").strip()
        if markdown:
            parts.append(f"=== PAGE {page_index + 1} ===\n{markdown}")
    return "\n\n".join(parts).strip()


def _ocr_pages_for_storage(ocr_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []
    for page in ocr_result.get("pages", []) or []:
        pages.append({
            "index": page.get("index", 0),
            "markdown": page.get("markdown", ""),
            "images": [],
            "dimensions": page.get("dimensions", {}),
        })
    return pages


async def run_answer_sheet_ocr_pipeline(
    document: Dict[str, Any],
    file_content: bytes,
    job_id: str,
    processing_result: Dict[str, Any],
    current_user: Dict[str, Any],
    db: DatabaseManager,
    cache: CacheManager,
    document_anchor_text: Optional[str] = None,
) -> PDFProcessingResult:
    """Run OCR for the answer sheet without creating student-visible questions."""
    document_id = document["document_id"]
    is_b2c = is_b2c_admin(current_user)

    try:
        logger.info(f"Calling answer-sheet OCR for job {job_id}")
        ocr_result = await call_sarvam_ocr(
            file_content,
            gateway_context=_build_ai_gateway_context(
                current_user=current_user,
                db=db,
                document_id=document_id,
                region_scope="answer_document",
                is_b2c=is_b2c,
            ),
        )
        extracted_text = _ocr_pages_to_plain_text(ocr_result)
        page_summaries = _ocr_pages_for_storage(ocr_result)

        processing_result["progress"] = 90
        await cache.set(f"pdf_answer_sheet_job:{job_id}", processing_result, 3600, "admin")

        update_data = {
            "answer_sheet_ocr_status": "completed",
            "answer_sheet_ocr_job_id": job_id,
            "answer_sheet_ocr_completed_at": datetime.utcnow(),
            "answer_sheet_extracted_text": extracted_text,
            "answer_sheet_ocr_pages": page_summaries,
            "answer_sheet_document_anchor_text": document_anchor_text.strip() if document_anchor_text else None,
            "answer_sheet_mapped_answers_count": 0,
        }

        if is_b2c:
            await db.b2c_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": update_data},
            )
        else:
            await db.mongo_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": update_data},
            )

        processing_result["status"] = "completed"
        processing_result["progress"] = 100
        processing_result["pages"] = page_summaries
        await cache.set(f"pdf_answer_sheet_job:{job_id}", processing_result, 3600, "admin")

        logger.info(
            f"Answer-sheet OCR completed for {document_id}: "
            f"{len(page_summaries)} pages, {len(extracted_text)} chars"
        )
        return PDFProcessingResult(**processing_result)

    except Exception as exc:
        logger.error(f"Answer-sheet OCR pipeline failed for {document_id}: {exc}", exc_info=True)
        update_data = {
            "answer_sheet_ocr_status": "error",
            "answer_sheet_ocr_error": str(exc),
            "answer_sheet_ocr_job_id": job_id,
        }
        if is_b2c:
            await db.b2c_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": update_data},
            )
        else:
            await db.mongo_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": update_data},
            )

        error_result = {
            "job_id": job_id,
            "status": "error",
            "progress": 100,
            "error": str(exc),
            "timestamp": datetime.utcnow(),
        }
        await cache.set(f"pdf_answer_sheet_job:{job_id}", error_result, 3600, "admin")
        raise


@router.post("/documents/{document_id}/answer-sheet/process-ocr", response_model=PDFProcessingResult)
@limiter.limit("5/minute")
async def process_answer_sheet_ocr(
    request: Request,
    document_id: str,
    ocr_request: Optional[AnswerSheetOCRRequest] = Body(default=None),
    async_mode: bool = Query(True, description="Queue OCR and return immediately when true"),
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache),
):
    """Trigger OCR processing on an uploaded answer sheet without exposing it to students."""
    ocr_started_at = datetime.utcnow()
    try:
        is_b2c = is_b2c_admin(current_user)
        if is_b2c:
            document = await db.b2c_find_one("documents", {"document_id": document_id})
        else:
            document = await db.mongo_find_one("documents", {"document_id": document_id})

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found",
            )

        if not document.get("answer_sheet_path"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No answer sheet uploaded for this document",
            )

        if document.get("answer_sheet_ocr_status") == "processing":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Answer sheet OCR processing already in progress",
            )

        stored_path_raw = str(document.get("answer_sheet_path") or "").replace("\\", "/")
        file_content = None

        if stored_path_raw.startswith("s3://"):
            file_content = await download_file(stored_path_raw)
            if not file_content:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Failed to download answer sheet from S3: {stored_path_raw}",
                )
        else:
            file_path = _resolve_answer_sheet_file_path(document)
            if not file_path:
                checked = ", ".join(str(candidate) for candidate in _answer_sheet_file_candidates(document))
                logger.error(f"Answer sheet file not found for {document_id}. Checked: {checked}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Answer sheet PDF file not found on server.",
                )

            async with aiofiles.open(str(file_path), "rb") as f:
                file_content = await f.read()

        job_id = str(uuid.uuid4())
        update_data = {
            "answer_sheet_ocr_status": "processing",
            "answer_sheet_ocr_job_id": job_id,
            "answer_sheet_ocr_started_at": datetime.utcnow(),
            "answer_sheet_ocr_error": None,
            "answer_sheet_document_anchor_text": (
                ocr_request.documentAnchorText.strip()
                if ocr_request and ocr_request.documentAnchorText
                else None
            ),
        }

        if is_b2c:
            await db.b2c_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": update_data},
            )
        else:
            await db.mongo_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": update_data},
            )

        processing_result = {
            "job_id": job_id,
            "status": "processing",
            "progress": 20,
            "extracted_questions": 0,
            "extracted_images": 0,
            "output_folder": f"answer_sheet_{document_id}_{int(datetime.utcnow().timestamp())}",
            "timestamp": datetime.utcnow(),
        }

        await cache.set(f"pdf_answer_sheet_job:{job_id}", processing_result, 3600, "admin")

    except HTTPException as exc:
        observe_ocr_job(
            job_type="answer_sheet",
            status=f"error_{exc.status_code}",
            duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
        )
        raise
    except Exception as exc:
        observe_ocr_job(
            job_type="answer_sheet",
            status="error_500",
            duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
        )
        logger.error(f"Failed to prepare answer-sheet OCR job for {document_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start answer sheet OCR processing: {exc}",
        )

    async def execute_pipeline() -> PDFProcessingResult:
        return await run_answer_sheet_ocr_pipeline(
            document=document,
            file_content=file_content,
            job_id=job_id,
            processing_result=processing_result,
            current_user=current_user,
            db=db,
            cache=cache,
            document_anchor_text=ocr_request.documentAnchorText if ocr_request else None,
        )

    async def execute_with_semaphore() -> PDFProcessingResult:
        semaphore = getattr(request.app.state, "ocr_semaphore", None)
        if semaphore:
            async with semaphore:
                return await execute_pipeline()
        return await execute_pipeline()

    if async_mode:
        tasks = getattr(request.app.state, "ocr_tasks", None)

        async def background_runner():
            try:
                await execute_with_semaphore()
                observe_ocr_job(
                    job_type="answer_sheet",
                    status="success_async",
                    duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
                )
            except HTTPException as exc:
                logger.error(
                    f"Background answer-sheet OCR job {job_id} failed with HTTP {exc.status_code}: {exc.detail}"
                )
                observe_ocr_job(
                    job_type="answer_sheet",
                    status=f"error_{exc.status_code}",
                    duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
                )
            except Exception as exc:
                observe_ocr_job(
                    job_type="answer_sheet",
                    status="error_500",
                    duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
                )
                logger.error(f"Background answer-sheet OCR job {job_id} failed: {exc}", exc_info=True)

        task = asyncio.create_task(background_runner())
        if isinstance(tasks, dict):
            tasks[job_id] = task

            def _cleanup(_):
                tasks.pop(job_id, None)

            task.add_done_callback(_cleanup)

        observe_ocr_job(
            job_type="answer_sheet",
            status="queued",
            duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=jsonable_encoder(PDFProcessingResult(**processing_result)),
        )

    try:
        result = await execute_with_semaphore()
        observe_ocr_job(
            job_type="answer_sheet",
            status="success",
            duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
        )
        return result
    except HTTPException as exc:
        observe_ocr_job(
            job_type="answer_sheet",
            status=f"error_{exc.status_code}",
            duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
        )
        raise
    except Exception as exc:
        observe_ocr_job(
            job_type="answer_sheet",
            status="error_500",
            duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
        )
        logger.error(f"Answer-sheet OCR processing failed for {document_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer sheet OCR processing failed: {exc}",
        )


@router.post("/documents/{document_id}/process-ocr", response_model=PDFProcessingResult)
@limiter.limit("5/minute")
async def process_document_ocr(
    request: Request,
    document_id: str,
    async_mode: bool = Query(True, description="Queue OCR and return immediately when true"),
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Trigger OCR processing on an existing uploaded document."""
    ocr_started_at = datetime.utcnow()
    try:
        # Check if B2C admin or B2C user
        user_type = current_user.get("user_type")
        is_b2c = user_type in ["b2c_admin", "b2c_user"]

        if is_b2c:
            document = await db.b2c_find_one("documents", {"document_id": document_id})
        else:
            document = await db.mongo_find_one("documents", {"document_id": document_id})

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        if document.get("ocr_status") == "processing":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="OCR processing already in progress"
            )

        if document.get("ocr_status") == "completed":
            logger.info(f"Reprocessing document {document_id} - cleaning up old data")

            if is_b2c:
                questions_deleted_result = await db.b2c_delete_many("questions", {"document_id": document_id})
                # delete_many returns bool in our implementation, not count directly unless we change it. 
                # Checking database.py implementation: returns bool (deleted_count > 0).
                # Actually, standard mongo driver returns DeleteResult.
                # Our wrapper b2c_delete_many returns bool.
                logger.info(f"Deleted questions for document {document_id} from B2C DB")
            else:
                questions_deleted = await db.mongo_delete_many("questions", {"document_id": document_id})
                logger.info(f"Deleted {questions_deleted} questions for document {document_id}")

            if is_b2c:
                images_result = await db.b2c_find("images", {"source_pdf": document["filename"]})
            else:
                images_result = await db.mongo_find("images", {"source_pdf": document["filename"]})

            for img in images_result:
                file_path = img.get("file_path")
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted image file: {file_path}")
                    except Exception as exc:
                        logger.error(f"Failed to delete image file {file_path}: {exc}")

            if is_b2c:
                await db.b2c_delete_many("images", {"source_pdf": document["filename"]})
            else:
                await db.mongo_delete_many("images", {"source_pdf": document["filename"]})

        from pathlib import Path as _Path
        backend_dir = _Path(os.getcwd())
        stored_path_raw = str(document.get("file_path", "")).replace("\\", "/")
        file_content = None

        if stored_path_raw.startswith("s3://"):
            logger.info(f"Downloading document from S3: {stored_path_raw}")
            file_content = await download_file(stored_path_raw)
            if not file_content:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Failed to download file from S3: {stored_path_raw}"
                )
        else:
            # Build a set of candidate locations to handle legacy absolute Windows paths
            candidates: list[_Path] = []

            if stored_path_raw:
                sp = _Path(stored_path_raw)
                # 1) Use as absolute if it is absolute
                if sp.is_absolute():
                    candidates.append(sp)
                # 2) Treat as repo-relative (current behavior)
                candidates.append(backend_dir / stored_path_raw)

                # 3) If path contains an embedded Windows drive with an 'uploads' segment, strip until '/uploads/...'
                if "uploads/" in stored_path_raw:
                    try:
                        uploads_index = stored_path_raw.index("uploads/")
                        rel_after_uploads = stored_path_raw[uploads_index:]
                        candidates.append(backend_dir / rel_after_uploads)
                    except ValueError:
                        pass

            # 4) Final fallback to canonical expected location
            canonical_fallback = backend_dir / f"uploads/documents/{document.get('document_type','')}/{document_id}.pdf"
            candidates.append(canonical_fallback)

            file_path: _Path | None = None
            for p in candidates:
                try:
                    if p.exists():
                        file_path = p
                        break
                except Exception:
                    continue

            if not file_path:
                logger.error(
                    f"PDF file not found for document {document_id}. Checked: " + 
                    ", ".join(str(c) for c in candidates)
                )
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="PDF file not found on server. Please re-upload this document from the Admin panel."
                )

            async with aiofiles.open(str(file_path), "rb") as f:
                file_content = await f.read()

        job_id = str(uuid.uuid4())

        if is_b2c:
            await db.b2c_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": {"ocr_status": "processing", "ocr_job_id": job_id}}
            )
        else:
            await db.mongo_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": {"ocr_status": "processing", "ocr_job_id": job_id}}
            )

        processing_result = {
            "job_id": job_id,
            "status": "processing",
            "progress": 20,
            "extracted_questions": 0,
            "extracted_images": 0,
            "output_folder": f"extracted_{document_id}_{int(datetime.utcnow().timestamp())}",
            "timestamp": datetime.utcnow()
        }

        await cache.set(f"pdf_job:{job_id}", processing_result, 3600, "admin")

    except HTTPException as exc:
        observe_ocr_job(
            job_type="document",
            status=f"error_{exc.status_code}",
            duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
        )
        raise
    except Exception as exc:
        observe_ocr_job(
            job_type="document",
            status="error_500",
            duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
        )
        logger.error(f"Failed to prepare OCR job for {document_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start OCR processing: {exc}"
        )

    async def execute_pipeline() -> PDFProcessingResult:
        return await run_document_ocr_pipeline(
            document=document,
            file_content=file_content,
            job_id=job_id,
            processing_result=processing_result,
            current_user=current_user,
            db=db,
            cache=cache
        )

    async def execute_with_semaphore() -> PDFProcessingResult:
        semaphore = getattr(request.app.state, "ocr_semaphore", None)
        if semaphore:
            async with semaphore:
                return await execute_pipeline()
        return await execute_pipeline()

    if async_mode:
        tasks = getattr(request.app.state, "ocr_tasks", None)

        async def background_runner():
            try:
                await execute_with_semaphore()
                observe_ocr_job(
                    job_type="document",
                    status="success_async",
                    duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
                )
            except HTTPException as exc:
                logger.error(f"Background OCR job {job_id} failed with HTTP {exc.status_code}: {exc.detail}")
                observe_ocr_job(
                    job_type="document",
                    status=f"error_{exc.status_code}",
                    duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
                )
            except Exception as exc:
                observe_ocr_job(
                    job_type="document",
                    status="error_500",
                    duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
                )
                logger.error(f"Background OCR job {job_id} failed: {exc}", exc_info=True)

        task = asyncio.create_task(background_runner())
        if isinstance(tasks, dict):
            tasks[job_id] = task

            def _cleanup(_):
                tasks.pop(job_id, None)

            task.add_done_callback(_cleanup)

        observe_ocr_job(
            job_type="document",
            status="queued",
            duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=jsonable_encoder(PDFProcessingResult(**processing_result))
        )

    try:
        result = await execute_with_semaphore()
        observe_ocr_job(
            job_type="document",
            status="success",
            duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
        )
        return result
    except HTTPException as exc:
        observe_ocr_job(
            job_type="document",
            status=f"error_{exc.status_code}",
            duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
        )
        raise
    except Exception as exc:
        observe_ocr_job(
            job_type="document",
            status="error_500",
            duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
        )
        logger.error(f"OCR processing failed for {document_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF: {exc}"
        )


@router.post("/direct-ocr")
@limiter.limit("6/minute")
async def perform_direct_ocr(
    request: Request,
    file: UploadFile = File(...),
    subject: Optional[str] = Form("General"),
    difficulty: Optional[str] = Form("medium"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Direct OCR processing for authenticated users (no document persistence)."""
    ocr_started_at = datetime.utcnow()
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )

        file_content = await file.read()

        async def _run_ocr() -> Dict[str, Any]:
            return await call_sarvam_ocr(
                file_content,
                gateway_context=_build_ai_gateway_context(
                    current_user=current_user,
                    db=db,
                    region_scope="direct",
                    is_b2c=is_b2c_admin(current_user),
                ),
            )

        semaphore = getattr(request.app.state, "ocr_semaphore", None)
        if semaphore:
            async with semaphore:
                ocr_result = await _run_ocr()
        else:
            ocr_result = await _run_ocr()

        result = {
            "success": True,
            "filename": file.filename,
            "subject": subject or "General",
            "difficulty": difficulty or "medium",
            "pages": ocr_result.get("pages", []),
            "metadata": {
                "processed_by": current_user.get("user_id"),
                "processed_at": datetime.utcnow().isoformat()
            }
        }
        observe_ocr_job(
            job_type="direct",
            status="success",
            duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
        )
        return result

    except HTTPException as exc:
        observe_ocr_job(
            job_type="direct",
            status=f"error_{exc.status_code}",
            duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
        )
        raise
    except Exception as exc:
        observe_ocr_job(
            job_type="direct",
            status="error_500",
            duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
        )
        logger.error(f"Direct OCR processing failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {exc}"
        )


@router.get("/status/{job_id}", response_model=PDFProcessingResult)
@limiter.limit("60/minute")
async def get_processing_status(
    request: Request,
    job_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    cache: CacheManager = Depends(get_cache)
):
    """Get PDF processing job status"""
    try:
        # Get cached status
        cached_result = await cache.get(f"pdf_job:{job_id}", "admin")

        if not cached_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )

        return PDFProcessingResult(**cached_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get status error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get job status"
        )

@router.get("/documents", response_model=DocumentListResponse)
@limiter.limit("60/minute")
async def get_documents(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    document_type: Optional[str] = Query(None),
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """Get list of uploaded documents with pagination"""
    try:
        # Check if B2C admin
        is_b2c = is_b2c_admin(current_user)
        
        # Build base filter scoped by tenant (admin) and role
        user_type = current_user.get("user_type")
        filter_query: Dict[str, Any] = {}
        
        if is_b2c:
            # B2C admin sees all documents in B2C database (no admin_id filter)
            pass
        elif user_type == "admin":
            try:
                filter_query["admin_id"] = BsonObjectId(current_user.get("admin_id", current_user["user_id"]))
            except Exception:
                pass
        else:
            # Tutor: filter by their admin_id and (optionally) by teacher mapping
            admin_id = current_user.get("admin_id")
            if admin_id:
                try:
                    filter_query["admin_id"] = BsonObjectId(admin_id)
                except Exception:
                    pass
        if document_type:
            filter_query["document_type"] = document_type

        # For tutors, only show docs mapped to them or open to all (no teacher restriction)
        if user_type == "tutor":
            tutor_id = current_user.get("tutor_id")
            filter_query = {
                "$and": [
                    filter_query,
                    {"$or": [
                        {"teacher_ids": {"$in": [tutor_id]}},
                        {"teacher_ids": []},
                        {"teacher_ids": None},
                        {"teacher_ids": {"$exists": False}}
                    ]}
                ]
            }

        # Get data from appropriate database
        if is_b2c:
            # B2C admin - query STOODY-b2c database
            total = len(await db.b2c_find("documents", filter_query))
            skip = (page - 1) * limit
            documents = await db.b2c_find(
                "documents",
                filter_query,
                skip=skip,
                limit=limit,
                sort=[("uploaded_at", -1)]
            )
        else:
            # Regular admin/tutor - query tenant database
            total = len(await db.mongo_find("documents", filter_query))
            skip = (page - 1) * limit
            documents = await db.mongo_find(
                "documents",
                filter_query,
                skip=skip,
                limit=limit,
                sort=[("uploaded_at", -1)]  # Sort by upload date, newest first
            )

        # Format response and check file availability
        # Collect unique uploader IDs to look up their names
        uploader_ids = list(set(doc.get("uploaded_by") for doc in documents if doc.get("uploaded_by")))
        uploader_names = {}

        # Look up uploader names from users, admins, or tutors collections
        if uploader_ids:
            try:
                for uid in uploader_ids:
                    if is_b2c:
                        user = await db.b2c_find_one("users", {"_id": BsonObjectId(uid)})
                        if user:
                            uploader_names[uid] = user.get("full_name") or user.get("username") or "B2C User"
                        else:
                            admin = await db.b2c_find_one("admins", {"_id": BsonObjectId(uid)})
                            uploader_names[uid] = admin.get("name") or "Admin" if admin else "Admin"
                    else:
                        admin = await db.mongo_find_one("admins", {"_id": BsonObjectId(uid)})
                        if admin:
                            uploader_names[uid] = admin.get("name") or admin.get("full_name") or "Admin"
                        else:
                            tutor = await db.mongo_find_one("tutors", {"_id": BsonObjectId(uid)})
                            if tutor:
                                uploader_names[uid] = tutor.get("name") or tutor.get("full_name") or tutor.get("username") or "Teacher"
                            else:
                                uploader_names[uid] = "Admin"
            except Exception as e:
                logger.warning(f"Could not look up uploader names: {e}")

        document_list = []
        for doc in documents:
            file_exists = _document_file_exists(doc)

            # Get uploader display name
            uploader_id = doc.get("uploaded_by", "")
            uploaded_by_name = uploader_names.get(uploader_id, "Admin")

            document_list.append(DocumentMetadata(
                document_id=doc["document_id"],
                title=doc["title"],
                document_type=doc["document_type"],
                subject=doc["subject"],
                difficulty=doc["difficulty"],
                course_plan=doc.get("course_plan"),
                standard=doc.get("standard"),
                section=doc.get("section"),
                teacher_ids=doc.get("teacher_ids"),
                file_path="",
                filename=doc["filename"],
                uploaded_by=doc["uploaded_by"],
                uploaded_by_name=uploaded_by_name,
                uploaded_at=doc["uploaded_at"],
                ocr_status=doc["ocr_status"],
                ocr_job_id=doc.get("ocr_job_id"),
                extracted_questions_count=doc.get("extracted_questions_count", 0),
                extracted_images_count=doc.get("extracted_images_count", 0),
                pages_count=doc.get("pages_count", 0),
                total_points=doc.get("total_points"),
                total_minutes=doc.get("total_minutes"),
                file_exists=file_exists,
                is_active=doc.get("is_active", True),
                instructions=doc.get("instructions"),
                exam_mode=doc.get("exam_mode"),
                exam_template_path=doc.get("exam_template_path"),
                answer_sheet_path=None,
                answer_sheet_filename=doc.get("answer_sheet_filename"),
                answer_sheet_uploaded_at=doc.get("answer_sheet_uploaded_at"),
                answer_sheet_pages_count=doc.get("answer_sheet_pages_count"),
                answer_sheet_ocr_status=doc.get("answer_sheet_ocr_status"),
                answer_sheet_ocr_job_id=doc.get("answer_sheet_ocr_job_id"),
                answer_sheet_ocr_completed_at=doc.get("answer_sheet_ocr_completed_at"),
                answer_sheet_processed_regions_count=doc.get("answer_sheet_processed_regions_count"),
                answer_sheet_mapped_answers_count=doc.get("answer_sheet_mapped_answers_count"),
                has_answer_sheet=bool(doc.get("answer_sheet_path")),
                exam_finalized=doc.get("exam_finalized"),
                exam_finalized_at=doc.get("exam_finalized_at"),
                exam_sync_summary=doc.get("exam_sync_summary"),
                orientation_applied=doc.get("orientation_applied"),
                exam_template_orientation_applied=doc.get("exam_template_orientation_applied"),
                tally_num_questions=doc.get("tally_num_questions"),
                tally_max_marks_per_question=doc.get("tally_max_marks_per_question"),
                tally_marking_scheme=doc.get("tally_marking_scheme"),
                tally_validate_paper_set=doc.get("tally_validate_paper_set"),
                tally_expected_paper_set=doc.get("tally_expected_paper_set"),
            ))

        return DocumentListResponse(
            documents=document_list,
            total=total,
            page=page,
            limit=limit
        )

    except Exception as e:
        logger.error(f"Get documents error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents"
        )

@router.get("/student/practice-sets")
@limiter.limit("30/minute")
async def get_student_practice_sets(
    request: Request,
    plan_type: Optional[str] = Query(None, description="Filter by course plan type"),
    subject: Optional[str] = Query(None, description="Filter by subject"),
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Get practice sets available for the current student based on their profile"""
    try:
        user_type = current_user.get("user_type", "student")
        is_b2c = current_user.get("is_b2c", False) or user_type == "b2c_user"

        # Handle B2C users - query from B2C database
        if is_b2c:
            # Get B2C user profile from B2C database
            b2c_user = await db.b2c_find_one("users", {"_id": BsonObjectId(current_user["user_id"])})
            
            if not b2c_user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="B2C user profile not found"
                )
            
            # Check if onboarding is complete
            if not b2c_user.get("onboarding_complete"):
                return {
                    "success": True,
                    "data": {
                        "practice_sets": [],
                        "total": 0,
                        "onboarding_required": True
                    }
                }
            
            # Get user's plan details
            user_exam_type = b2c_user.get("exam_type")
            user_class_level = b2c_user.get("class_level")
            user_standard = b2c_user.get("standard")
            user_subjects = b2c_user.get("subjects", [])
            user_plan_types = b2c_user.get("plan_types", [])
            
            # Get B2C admin ID for content filtering
            b2c_admin = await db.b2c_find_one("admins", {}, {"_id": 1})
            b2c_admin_id = b2c_admin["_id"] if b2c_admin else None
            
            # Build filter for B2C practice sets
            filter_query = {
                "document_type": "Practice Sets",
                "ocr_status": "completed",
                "is_active": {"$ne": False}
            }
            
            if b2c_admin_id:
                try:
                    filter_query["admin_id"] = BsonObjectId(b2c_admin_id)
                except:
                    filter_query["admin_id"] = b2c_admin_id
            
            # Apply plan type filter
            if plan_type:
                filter_query["course_plan"] = plan_type
            elif user_plan_types:
                filter_query["course_plan"] = {"$in": user_plan_types}
            elif user_exam_type:
                filter_query["course_plan"] = user_exam_type
            
            # Apply subject filter
            if subject:
                filter_query["subject"] = subject
            elif user_subjects:
                filter_query["subject"] = {"$in": user_subjects}
            
            # Apply standard filter
            if user_standard:
                filter_query["standard"] = user_standard
            
            logger.info(f"B2C user {current_user['user_id']} practice sets query: {filter_query}")
            
            # Get practice sets from B2C database
            practice_sets = await db.b2c_find(
                "documents",
                filter_query,
                sort=[("uploaded_at", -1)]
            )
            
            logger.info(f"B2C practice sets found: {len(practice_sets)}")
            
            # Format response
            practice_sets_list = []
            user_id = current_user["user_id"]
            
            for doc in practice_sets:
                doc_id = doc.get("document_id") or str(doc.get("_id"))
                
                # Check if B2C user has attempted/completed this practice set
                sessions = await db.b2c_find(
                    "practice_sessions",
                    {
                        "student_id": user_id,
                        "document_id": doc_id
                    },
                    sort=[("started_at", -1)],
                    limit=10
                )
                
                has_attempted = len(sessions) > 0
                completed = any(s.get("is_completed", False) for s in sessions)
                
                practice_sets_list.append({
                    "document_id": doc_id,
                    "title": doc.get("title"),
                    "subject": doc.get("subject"),
                    "difficulty": doc.get("difficulty"),
                    "course_plan": doc.get("course_plan"),
                    "standard": doc.get("standard"),
                    "extracted_questions_count": doc.get("extracted_questions_count", 0),
                    "completed": completed,
                    "attempted": has_attempted,
                    "instructions": doc.get("instructions"),
                    "session_count": len(sessions)
                })

            return {
                "success": True,
                "data": {
                    "practice_sets": practice_sets_list,
                    "total": len(practice_sets_list)
                }
            }

        if user_type == "admin":
            # Admin can see all practice sets (including non-validated for testing)
            filter_query = {"document_type": "Practice Sets"}

            # Filter by plan type if specified
            if plan_type:
                filter_query["course_plan"] = plan_type

            # Filter by subject if specified
            if subject:
                filter_query["subject"] = subject

            # Get practice sets that match the criteria
            practice_sets = await db.mongo_find(
                "documents",
                filter_query,
                sort=[("uploaded_at", -1)]
            )

            # Format response
            practice_sets_list = []
            user_id = current_user["user_id"]

            for doc in practice_sets:
                doc_id = doc["document_id"]

                # Check if admin has attempted/completed this practice set
                # For practice sets, we check practice_sessions collection by document_id
                sessions = await db.mongo_find(
                    "practice_sessions",
                    {
                        "student_id": user_id,
                        "document_id": doc_id
                    },
                    sort=[("started_at", -1)],
                    limit=10
                )

                # Consider completed if any session for THIS specific practice set is completed
                has_attempted = len(sessions) > 0
                completed = any(s.get("is_completed", False) for s in sessions)

                practice_sets_list.append({
                    "document_id": doc_id,
                    "title": doc["title"],
                    "subject": doc["subject"],
                    "difficulty": doc["difficulty"],
                    "course_plan": doc.get("course_plan"),
                    "standard": doc.get("standard"),
                    "extracted_questions_count": doc.get("extracted_questions_count", 0),
                    "instructions": doc.get("instructions"),
                    "completed": completed,
                    "attempted": has_attempted,
                    "session_count": len(sessions)
                })

            return {
                "success": True,
                "data": {
                    "practice_sets": practice_sets_list,
                    "total": len(practice_sets_list)
                }
            }


        # Student - get their profile and filter by access
        student_profile = await db.mongo_find_one("students", {"_id": BsonObjectId(current_user["user_id"])})

        if not student_profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student profile not found"
            )

        student_grade = student_profile.get("grade")
        student_subjects = student_profile.get("subjects", [])
        student_plan_types = student_profile.get("plan_types", [])
        student_section = student_profile.get("section")  # Section A-F
        student_teacher_ids = student_profile.get("teacher_ids", [])  # Array of teacher IDs

        # Build filter for practice sets - only check if OCR is completed
        filter_query = {
            "document_type": "Practice Sets",
            "ocr_status": "completed",  # Only show practice sets that have been processed with OCR
            # is_active: {$ne: False} matches True, None, or missing field (default active)
            "is_active": {"$ne": False}
        }

        # Get admin_id from student for filtering admin-specific content
        admin_id = current_user.get("admin_id")
        if admin_id:
            # Documents may store admin_id as ObjectId or string; support both
            try:
                admin_oid = BsonObjectId(admin_id)
                admin_filter = {"$in": [admin_oid, admin_id]}
            except Exception:
                admin_filter = admin_id
            filter_query["admin_id"] = admin_filter

        # Filter by plan type if specified in query
        if plan_type:
            filter_query["course_plan"] = plan_type
        elif student_plan_types and len(student_plan_types) > 0:
            # If student has specific plan types assigned, filter by them
            filter_query["course_plan"] = {"$in": student_plan_types}
        # If student has no plan types assigned, show all plan types from their admin

        # Filter by subject if specified in query (explicit filter from frontend dropdown)
        if subject:
            filter_query["subject"] = subject
        elif student_subjects and len(student_subjects) > 0:
            # If student has specific subjects assigned, only show content for those subjects
            filter_query["subject"] = {"$in": student_subjects}

        # Build $and conditions array for section and teacher_ids filtering
        and_conditions = []

        # Filter by student's grade if available - EXACT match
        # Both student.grade and document.standard come from admin settings, so they match exactly
        if student_grade:
            filter_query["standard"] = student_grade

        # Filter by student's section if available (only show docs for their section or docs without section restriction)
        if student_section:
            and_conditions.append({
                "$or": [
                    {"section": student_section},
                    {"section": None},
                    {"section": {"$exists": False}}
                ]
            })

        # Filter by student's teacher_ids if available (only show docs mapped to their teachers or docs without teacher restriction)
        if student_teacher_ids:
            # Document must either have overlapping teacher_ids OR have empty/null teacher_ids
            and_conditions.append({
                "$or": [
                    {"teacher_ids": {"$in": student_teacher_ids}},
                    {"teacher_ids": []},
                    {"teacher_ids": None},
                    {"teacher_ids": {"$exists": False}}
                ]
            })

        # If we have additional AND conditions, wrap the filter_query
        if and_conditions:
            # Combine existing filter_query with new AND conditions
            and_conditions.insert(0, filter_query)
            filter_query = {"$and": and_conditions}

        # Log filter query for debugging
        logger.info(f"Student profile - Grade: {student_grade}, Subjects: {student_subjects}, Plan Types: {student_plan_types}, Section: {student_section}, Teacher IDs: {student_teacher_ids}")
        logger.info(f"Practice sets filter query: {filter_query}")

        # Get practice sets that match the criteria
        practice_sets = await db.mongo_find(
            "documents",
            filter_query,
            sort=[("uploaded_at", -1)]  # Sort by upload date, newest first
        )

        logger.info(f"Found {len(practice_sets)} practice sets matching filter")

        # Get actual question counts per document via aggregation (avoids stale extracted_questions_count)
        doc_ids = [doc["document_id"] for doc in practice_sets]
        question_counts: Dict[str, int] = {}
        if doc_ids:
            count_pipeline = [
                {"$match": {"document_id": {"$in": doc_ids}}},
                {"$group": {"_id": "$document_id", "count": {"$sum": 1}}}
            ]
            count_results = await db.mongo_aggregate("questions", count_pipeline)
            for r in count_results:
                question_counts[r["_id"]] = r["count"]

        # Format response - only include necessary fields for security
        practice_sets_list = []
        user_id = current_user["user_id"]

        for doc in practice_sets:
            doc_id = doc["document_id"]

            # Check if student has attempted/completed this practice set
            # For practice sets, we check practice_sessions collection by document_id
            sessions = await db.mongo_find(
                "practice_sessions",
                {
                    "student_id": user_id,
                    "document_id": doc_id
                },
                sort=[("started_at", -1)],
                limit=10
            )

            # Consider completed if any session for THIS specific practice set is completed
            has_attempted = len(sessions) > 0
            completed = any(s.get("is_completed", False) for s in sessions)

            # Get latest session stats if available
            latest_session = None
            if sessions:
                latest = sessions[0]
                accuracy_rate = 0.0
                if latest.get("questions_attempted", 0) > 0:
                    accuracy_rate = (latest.get("correct_answers", 0) / latest["questions_attempted"]) * 100

                latest_session = {
                    "questions_attempted": latest.get("questions_attempted", 0),
                    "correct_answers": latest.get("correct_answers", 0),
                    "accuracy_rate": round(accuracy_rate, 1),
                    "started_at": latest.get("started_at").isoformat() if latest.get("started_at") else None,
                    "is_completed": latest.get("is_completed", False)
                }

            practice_sets_list.append({
                "document_id": doc_id,
                "title": doc["title"],
                "subject": doc.get("subject", ""),
                "difficulty": doc.get("difficulty", ""),
                "course_plan": doc.get("course_plan"),
                "standard": doc.get("standard"),
                "extracted_questions_count": question_counts.get(doc_id, doc.get("extracted_questions_count", 0)),
                "instructions": doc.get("instructions"),
                "completed": completed,
                "attempted": has_attempted,
                "session_count": len(sessions),
                "latest_session": latest_session
            })

        return {
            "success": True,
            "data": {
                "practice_sets": practice_sets_list,
                "total": len(practice_sets_list)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get student practice sets error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get practice sets"
        )

@router.get("/student/available-options")
@limiter.limit("30/minute")
async def get_student_available_options(
    request: Request,
    document_type: Optional[str] = Query(None, description="Document type (Practice Sets or Test Series)"),
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Get available course plans, subjects, and other options for the student based on admin's uploaded content"""
    try:
        admin_id = current_user.get("admin_id") if current_user.get("user_type") == "student" else current_user.get("user_id")

        # Build base filter (support ObjectId or string in Mongo)
        try:
            admin_oid = BsonObjectId(admin_id)
            admin_filter = {"$in": [admin_oid, admin_id]}
        except Exception:
            admin_filter = admin_id

        # is_active: {$ne: False} matches True, None, or missing field (default active)
        filter_query = {"admin_id": admin_filter, "is_active": {"$ne": False}}
        if document_type:
            filter_query["document_type"] = document_type

        # Get all documents for this admin
        documents = await db.mongo_find("documents", filter_query)

        # Extract unique values for each field
        course_plans = set()
        subjects = set()
        standards = set()

        for doc in documents:
            if doc.get("course_plan"):
                course_plans.add(doc["course_plan"])
            if doc.get("subject"):
                subjects.add(doc["subject"])
            if doc.get("standard"):
                standards.add(doc["standard"])

        return {
            "success": True,
            "data": {
                "course_plans": sorted(list(course_plans)),
                "subjects": sorted(list(subjects)),
                "standards": sorted(list(standards))
            }
        }

    except Exception as e:
        logger.error(f"Get available options error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get available options"
        )

    try:
        # Build filter
        filter_query = {}
        if document_type:
            filter_query["document_type"] = document_type

        # Get total count
        total = len(await db.mongo_find("documents", filter_query))

        # Get paginated documents
        skip = (page - 1) * limit
        documents = await db.mongo_find(
            "documents",
            filter_query,
            skip=skip,
            limit=limit,
            sort=[("uploaded_at", -1)]  # Sort by upload date, newest first
        )

        # Format response and check file availability
        document_list = []
        for doc in documents:
            file_exists = _document_file_exists(doc)

            document_list.append(DocumentMetadata(
                document_id=doc["document_id"],
                title=doc["title"],
                document_type=doc["document_type"],
                subject=doc["subject"],
                difficulty=doc["difficulty"],
                course_plan=doc.get("course_plan"),
                standard=doc.get("standard"),
                file_path="",
                filename=doc["filename"],
                uploaded_by=doc["uploaded_by"],
                uploaded_at=doc["uploaded_at"],
                ocr_status=doc["ocr_status"],
                ocr_job_id=doc.get("ocr_job_id"),
                extracted_questions_count=doc.get("extracted_questions_count", 0),
                extracted_images_count=doc.get("extracted_images_count", 0),
                file_exists=file_exists,
                is_active=doc.get("is_active", True),
                instructions=doc.get("instructions"),
                exam_mode=doc.get("exam_mode"),
                exam_template_path=doc.get("exam_template_path"),
                answer_sheet_path=None,
                answer_sheet_filename=None,
                answer_sheet_uploaded_at=None,
                answer_sheet_pages_count=None,
                answer_sheet_ocr_status=None,
                answer_sheet_ocr_job_id=None,
                answer_sheet_ocr_completed_at=None,
                answer_sheet_processed_regions_count=None,
                answer_sheet_mapped_answers_count=None,
                has_answer_sheet=False,
                exam_finalized=doc.get("exam_finalized"),
                exam_finalized_at=doc.get("exam_finalized_at"),
                exam_sync_summary=doc.get("exam_sync_summary"),
                orientation_applied=doc.get("orientation_applied"),
                exam_template_orientation_applied=doc.get("exam_template_orientation_applied"),
                tally_num_questions=doc.get("tally_num_questions"),
                tally_max_marks_per_question=doc.get("tally_max_marks_per_question"),
                tally_marking_scheme=doc.get("tally_marking_scheme"),
                tally_validate_paper_set=doc.get("tally_validate_paper_set"),
                tally_expected_paper_set=doc.get("tally_expected_paper_set"),
            ))

        return DocumentListResponse(
            documents=document_list,
            total=total,
            page=page,
            limit=limit
        )

    except Exception as e:
        logger.error(f"Get documents error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents"
        )

@router.get("/documents/{document_id}/file")
@limiter.limit("30/minute")
async def get_document_file(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """Serve PDF file for viewing"""
    from fastapi.responses import FileResponse, Response

    try:
        logger.info(f"Attempting to fetch document with ID: {document_id}")

        # Get document metadata - try both main and B2C database
        document = await db.mongo_find_one("documents", {"document_id": document_id})
        if not document:
            document = await db.b2c_find_one("documents", {"document_id": document_id})
            
        if not document:
            # Debug: Log what's in the database
            all_docs = await db.mongo_find("documents", {}, limit=10)
            available_ids = [d.get('document_id', 'NO_ID') for d in all_docs]
            logger.error(f"Document '{document_id}' not found. Available IDs: {available_ids}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' not found in database. Available: {available_ids}"
            )

        # For tutors, ensure they are allowed to access this document
        if current_user.get("user_type") == "tutor":
            tutor_id = current_user.get("tutor_id")
            teacher_ids = document.get("teacher_ids")
            if teacher_ids and tutor_id not in teacher_ids:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Tutor not authorized for this document")

        stored_path = document.get("file_path", "")
        logger.info(f"Document found. File path: {stored_path}")

        # Check if this is an S3 path
        if stored_path.startswith("s3://"):
            logger.info(f"Fetching PDF from S3: {stored_path}")
            
            # Import S3 download function
            from utils.s3_storage import download_file as s3_download
            
            # Download from S3
            file_data = await s3_download(stored_path)
            
            if not file_data:
                logger.error(f"Failed to download PDF from S3: {stored_path}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="PDF file not found in S3"
                )
            
            # Return as response
            return Response(
                content=file_data,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"inline; filename=\"{document.get('filename', 'document.pdf')}\""
                }
            )

        # Local file handling. Use the same path candidates as the document list
        # so legacy absolute paths and repo-relative uploads resolve consistently.
        file_path = _resolve_document_file_path(document)
        if not file_path:
            checked = ", ".join(str(candidate) for candidate in _document_file_candidates(document))
            logger.error(f"File does not exist for document {document_id}. Checked: {checked}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="PDF file not found on server. Please re-upload this document from the Admin panel."
            )

        # Return file response
        logger.info(f"Serving PDF file: {document['filename']}")
        return FileResponse(
            path=str(file_path),
            media_type="application/pdf",
            filename=document["filename"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document file error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document file: {str(e)}"
        )


@router.get("/documents/{document_id}/answer-sheet/file")
@limiter.limit("30/minute")
async def get_document_answer_sheet_file(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """Serve uploaded answer sheet PDF for viewing."""
    from fastapi.responses import FileResponse, Response

    try:
        document = await db.mongo_find_one("documents", {"document_id": document_id})
        if not document:
            document = await db.b2c_find_one("documents", {"document_id": document_id})

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        if current_user.get("user_type") == "tutor":
            tutor_id = current_user.get("tutor_id")
            teacher_ids = document.get("teacher_ids")
            if teacher_ids and tutor_id not in teacher_ids:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Tutor not authorized for this document")

        stored_path = str(document.get("answer_sheet_path") or "")
        if not stored_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No answer sheet uploaded for this document"
            )

        filename = document.get("answer_sheet_filename") or f"{document_id}_answer_sheet.pdf"

        if stored_path.startswith("s3://"):
            file_data = await download_file(stored_path)
            if not file_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Answer sheet PDF file not found in S3"
                )
            return Response(
                content=file_data,
                media_type="application/pdf",
                headers={"Content-Disposition": f"inline; filename=\"{filename}\""}
            )

        file_path = _resolve_answer_sheet_file_path(document)
        if not file_path:
            checked = ", ".join(str(candidate) for candidate in _answer_sheet_file_candidates(document))
            logger.error(f"Answer sheet file does not exist for document {document_id}. Checked: {checked}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Answer sheet PDF file not found on server. Please re-upload this answer sheet from the Admin panel."
            )

        return FileResponse(
            path=str(file_path),
            media_type="application/pdf",
            filename=filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get answer sheet file error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve answer sheet file: {str(e)}"
        )

@router.post("/documents/{document_id}/recalculate-points")
@limiter.limit("30/minute")
async def recalculate_document_points(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """Recalculate total_points for a Test Series document based on question points"""
    try:
        # Get existing document
        document = await db.mongo_find_one("documents", {"document_id": document_id})
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        if document.get("document_type") != "Test Series":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only Test Series documents have total points"
            )

        # Block recalculation if document is finalized for exam
        if document.get("exam_finalized"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot recalculate points for a finalized exam document",
            )

        # Get all questions for this document
        questions = await db.mongo_find("questions", {"pdf_source": document_id})
        total_points = sum(q.get("points", 4.0) for q in questions)  # Default 4 marks per question

        # Update document's total_points
        await db.mongo_update_one(
            "documents",
            {"document_id": document_id},
            {"$set": {"total_points": total_points}}
        )

        logger.info(f"Recalculated total_points for {document_id}: {total_points}")

        return {
            "message": "Total points recalculated successfully",
            "document_id": document_id,
            "total_points": total_points,
            "question_count": len(questions)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recalculate points error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to recalculate points: {str(e)}"
        )

async def _notify_students_content_activated(
    db: DatabaseManager,
    doc: Dict[str, Any],
    current_user: Dict[str, Any],
) -> None:
    """
    When a document is activated (is_active false→true), find all matching
    students and create a notification for each.
    """
    from api.v1.notifications_async import create_notifications_batch

    admin_id = doc.get("admin_id")
    if not admin_id:
        return

    # Build student filter matching the same criteria students use to see content
    student_filter: Dict[str, Any] = {"is_active": {"$ne": False}}

    doc_standard = doc.get("standard")
    if doc_standard:
        student_filter["grade"] = doc_standard

    doc_subject = doc.get("subject")
    if doc_subject:
        student_filter["subjects"] = doc_subject

    doc_plan = doc.get("course_plan")
    if doc_plan:
        student_filter["plan_types"] = doc_plan

    doc_section = doc.get("section")
    if doc_section:
        student_filter["section"] = doc_section

    # If document is restricted to specific teachers, only notify their students
    doc_teacher_ids = doc.get("teacher_ids")
    if doc_teacher_ids and isinstance(doc_teacher_ids, list) and len(doc_teacher_ids) > 0:
        student_filter["teacher_ids"] = {"$in": doc_teacher_ids}

    matching_students = await db.mongo_find(
        "students", student_filter, projection={"_id": 1}, limit=5000
    )
    if not matching_students:
        return

    recipient_ids = [str(s["_id"]) for s in matching_students]

    doc_type = doc.get("document_type", "Content")
    title_map = {
        "Practice Sets": "New Practice Set Available",
        "Test Series": "New Test Assigned",
        "Chapter Notes": "New Notes Available",
    }
    category_map = {
        "Practice Sets": "practice",
        "Test Series": "test",
        "Chapter Notes": "notes",
    }

    creator_name = current_user.get("name") or current_user.get("full_name", "")

    await create_notifications_batch(
        db=db,
        admin_id=admin_id,
        recipient_ids=recipient_ids,
        notif_type="assignment",
        category=category_map.get(doc_type, "content"),
        title=title_map.get(doc_type, "New Content Available"),
        message=f"{doc.get('title', doc_type)} — {doc.get('subject', '')}",
        metadata={
            "document_id": doc.get("document_id", ""),
            "document_type": doc_type,
            "subject": doc.get("subject", ""),
            "title": doc.get("title", ""),
        },
        created_by=current_user.get("user_id", ""),
        created_by_name=creator_name,
    )


@router.patch("/documents/{document_id}/metadata")
@limiter.limit("30/minute")
async def update_document_metadata(
    request: Request,
    document_id: str,
    metadata: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """Update document metadata (e.g., total_points)"""
    try:
        # Get existing document
        existing_doc = await db.mongo_find_one("documents", {"document_id": document_id})
        if not existing_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        # Block metadata edits if document is finalized for exam
        if existing_doc.get("exam_finalized"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot modify metadata of a finalized exam document",
            )

        # Update allowed fields
        update_data = {}

        # String fields that can be updated
        string_fields = ["title", "subject", "course_plan", "standard", "section"]
        for field in string_fields:
            if field in metadata and metadata[field] is not None:
                value = str(metadata[field]).strip()
                if value:  # Only update if non-empty
                    update_data[field] = value

        # Instructions field (allow clearing with empty string)
        if "instructions" in metadata:
            value = str(metadata["instructions"]).strip() if metadata["instructions"] else ""
            update_data["instructions"] = value if value else None

        # Document type with validation
        if "document_type" in metadata:
            doc_type = metadata["document_type"]
            valid_types = ["Practice Sets", "Test Series", "Chapter Notes"]
            if doc_type not in valid_types:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid document type. Must be one of: {valid_types}"
                )
            update_data["document_type"] = doc_type

        # Difficulty with validation
        if "difficulty" in metadata:
            difficulty = metadata["difficulty"]
            valid_difficulties = ["easy", "medium", "hard"]
            if difficulty not in valid_difficulties:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid difficulty. Must be one of: {valid_difficulties}"
                )
            update_data["difficulty"] = difficulty

        # Teacher IDs (array field)
        if "teacher_ids" in metadata:
            teacher_ids = metadata["teacher_ids"]
            if isinstance(teacher_ids, list):
                update_data["teacher_ids"] = teacher_ids

        # Numeric fields
        if "total_points" in metadata:
            total_points = metadata["total_points"]
            if total_points < 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Total points must be greater than or equal to 0"
                )
            update_data["total_points"] = total_points

        if "total_minutes" in metadata:
            total_minutes = metadata["total_minutes"]
            if total_minutes <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Total minutes must be greater than 0"
                )
            update_data["total_minutes"] = total_minutes

        if "is_active" in metadata:
            update_data["is_active"] = bool(metadata["is_active"])

        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid fields to update"
            )

        logger.info(f"Updating document {document_id} with metadata: {update_data}")

        # Update in MongoDB
        result = await db.mongo_update_one(
            "documents",
            {"document_id": document_id},
            {"$set": update_data}
        )

        # mongo_update_one returns False if no changes or not found
        # But we already verified document exists above, so just log and continue
        logger.info(f"Update result for {document_id}: {result}")

        # --- Notification: content activated (false → true) ---
        was_inactive = not existing_doc.get("is_active", False)
        now_active = update_data.get("is_active") is True
        if was_inactive and now_active:
            try:
                await _notify_students_content_activated(db, existing_doc, current_user)
            except Exception as notif_err:
                # Notification failure must never block the main operation
                logger.warning(f"Notification side-effect failed: {notif_err}")

        return {
            "message": "Document metadata updated successfully",
            "document_id": document_id,
            "updated_fields": update_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update document metadata error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update document metadata: {str(e)}"
        )


@router.post("/documents/{document_id}/duplicate")
@limiter.limit("10/minute")
async def duplicate_document(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Duplicate a document with different metadata settings.
    Creates a new document entry that references the same file but with updated metadata.
    Questions are also duplicated to the new document.
    """
    # Parse metadata from request body
    try:
        metadata = await request.json()
    except Exception:
        metadata = {}

    try:
        # Get existing document
        existing_doc = await db.mongo_find_one("documents", {"document_id": document_id})
        if not existing_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        # Generate new document_id
        new_document_id = metadata.get("new_document_id")
        if not new_document_id:
            import uuid
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            new_document_id = f"{document_id}_copy_{timestamp}"

        # Check if new_document_id already exists
        existing_new = await db.mongo_find_one("documents", {"document_id": new_document_id})
        if existing_new:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document with ID {new_document_id} already exists"
            )

        # Create new document with updated metadata
        new_doc = existing_doc.copy()
        del new_doc["_id"]  # Remove MongoDB _id
        new_doc["document_id"] = new_document_id
        new_doc["uploaded_at"] = datetime.utcnow().isoformat()

        # Update metadata fields from request
        update_fields = ["title", "subject", "course_plan", "standard", "section",
                        "difficulty", "document_type", "teacher_ids", "total_minutes"]
        for field in update_fields:
            if field in metadata and metadata[field] is not None:
                new_doc[field] = metadata[field]

        # Insert new document
        await db.mongo_insert_one("documents", new_doc)
        logger.info(f"Duplicated document {document_id} to {new_document_id}")

        # Duplicate questions if they exist
        questions = await db.mongo_find("questions", {"document_id": document_id})
        questions_duplicated = 0
        if questions:
            for q in questions:
                new_q = q.copy()
                del new_q["_id"]
                new_q["document_id"] = new_document_id
                # Update question fields from new document metadata
                if "subject" in metadata:
                    new_q["subject"] = metadata["subject"]
                if "course_plan" in metadata:
                    new_q["course_plan"] = metadata["course_plan"]
                if "standard" in metadata:
                    new_q["standard"] = metadata["standard"]
                await db.mongo_insert_one("questions", new_q)
                questions_duplicated += 1

        return {
            "message": "Document duplicated successfully",
            "original_document_id": document_id,
            "new_document_id": new_document_id,
            "questions_duplicated": questions_duplicated
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Duplicate document error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to duplicate document: {str(e)}"
        )


@router.get("/documents/{document_id}/questions")
@limiter.limit("60/minute")
async def get_document_questions(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Get all questions extracted from a specific document"""
    try:
        # Check if B2C admin or B2C user
        user_type = current_user.get("user_type")
        is_b2c = user_type in ["b2c_admin", "b2c_user"]
        
        # Verify document exists in appropriate database
        if is_b2c:
            document = await db.b2c_find_one("documents", {"document_id": document_id})
        else:
            document = await db.mongo_find_one("documents", {"document_id": document_id})
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        # ── Access control: verify the user belongs to the same school as the document ──
        # Each role stores the school admin reference differently:
        #   - student  → admin_id in JWT is the school admin who created them
        #   - tutor    → admin_id in JWT is the school admin (from created_by)
        #   - admin    → user_id in JWT IS the school admin
        #   - b2c_*    → no admin_id check needed (single-tenant B2C)
        from config_async import DEBUG_MODE as _DEBUG_MODE

        document_admin_id = document.get("admin_id")
        document_admin_id_str = str(document_admin_id) if document_admin_id is not None else None

        if user_type == "student":
            # Students: compare their admin_id (the school they belong to) with the document owner
            student_admin_id = str(current_user.get("admin_id")) if current_user.get("admin_id") is not None else None
            if student_admin_id != document_admin_id_str:
                if _DEBUG_MODE:
                    logger.warning(
                        f"DEBUG_MODE: allowing student {current_user.get('user_id')} with admin_id={student_admin_id} "
                        f"to access document owned by admin_id={document_admin_id_str}"
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="You don't have access to this document"
                    )

            # Students can only access completed OCR documents (unless in DEBUG_MODE)
            if document.get("ocr_status") != "completed":
                if _DEBUG_MODE:
                    logger.warning(
                        f"DEBUG_MODE: allowing access to document {document_id} with ocr_status={document.get('ocr_status')}"
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="This document is not yet available"
                    )

        elif user_type == "tutor":
            # Tutors: compare their admin_id (the school admin who created them) with the document owner.
            # A tutor's admin_id comes from the created_by field set during tutor creation.
            tutor_admin_id = str(current_user.get("admin_id")) if current_user.get("admin_id") is not None else None
            if tutor_admin_id != document_admin_id_str:
                if _DEBUG_MODE:
                    logger.warning(
                        f"DEBUG_MODE: allowing tutor {current_user.get('user_id')} with admin_id={tutor_admin_id} "
                        f"to access document owned by admin_id={document_admin_id_str}"
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="You don't have access to this document"
                    )

        elif user_type == "admin":
            # Admins: their own user_id IS the admin_id that owns documents
            admin_id = str(current_user.get("user_id")) if current_user.get("user_id") is not None else None
            if admin_id != document_admin_id_str:
                if _DEBUG_MODE:
                    logger.warning(
                        f"DEBUG_MODE: allowing admin {admin_id} to access document owned by admin_id={document_admin_id_str}"
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="You don't have access to this document"
                    )

        # B2C admins/users can access all B2C documents (single-tenant, no admin_id check needed)

        # Get questions for this document from appropriate database
        if is_b2c:
            questions = await db.b2c_find("questions", {"document_id": document_id})
        else:
            questions = await db.mongo_find("questions", {"document_id": document_id})

        # Worked-answer mappings are an admin/tutor review surface. Do not attach
        # them for student/B2C learner reads from this shared route.
        include_worked_answers = user_type in ["admin", "tutor", "b2c_admin"]
        mappings_by_question_id: Dict[str, Dict[str, Any]] = {}
        if include_worked_answers:
            if is_b2c:
                answer_mappings = await db.b2c_find("answer_question_mappings", {"document_id": document_id})
            else:
                answer_mappings = await db.mongo_find("answer_question_mappings", {"document_id": document_id})

            for mapping in answer_mappings:
                question_id = str(mapping.get("question_id") or mapping.get("question_region_id") or "")
                if question_id and mapping.get("answer_text"):
                    mappings_by_question_id[question_id] = _serialize_answer_mapping(mapping)

        # Convert ObjectId to string for JSON serialization and map field names
        serialized_questions = []
        for q in questions:
            # Auto-clean orphaned images from the question
            from utils.image_validator import clean_question_images
            cleaned_q, removed_count = await clean_question_images(q, db, is_b2c)

            # If orphaned images were found and removed, update the database
            if removed_count > 0:
                if is_b2c:
                    await db.b2c_update_one(
                        "questions",
                        {"id": q.get("id")},
                        {"$set": {
                            "images": cleaned_q.get("images", []),
                            "question_figures": cleaned_q.get("question_figures", []),
                            "auto_cleaned_at": datetime.utcnow()
                        }}
                    )
                else:
                    await db.mongo_update_one(
                        "questions",
                        {"id": q.get("id")},
                        {"$set": {
                            "images": cleaned_q.get("images", []),
                            "question_figures": cleaned_q.get("question_figures", []),
                            "auto_cleaned_at": datetime.utcnow()
                        }}
                    )
                logger.info(f"Auto-cleaned {removed_count} orphaned images from question {q.get('id')} during retrieval")

            question_dict = {}
            for key, value in cleaned_q.items():
                if isinstance(value, BsonObjectId):
                    question_dict[key] = str(value)
                elif isinstance(value, datetime):
                    question_dict[key] = value.isoformat()
                else:
                    question_dict[key] = value

            # Map backend field names to frontend expected names
            if "text" in question_dict:
                question_dict["question_text"] = question_dict["text"]

            # === ENHANCED: Load base64 image data for question_figures ===
            enriched_figures = []
            for fig_ref in question_dict.get("question_figures", []) or []:
                try:
                    fig_id = fig_ref.get("id") if isinstance(fig_ref, dict) else fig_ref
                    base64_data = None
                    
                    # First check if base64Data is already embedded in the figure reference
                    if isinstance(fig_ref, dict) and fig_ref.get("base64Data"):
                        base64_data = fig_ref["base64Data"]
                    else:
                        # Try to get base64Data from images collection
                        if is_b2c:
                            img_doc = await db.b2c_find_one("images", {"_id": fig_id})
                        else:
                            img_doc = await db.mongo_find_one("images", {"_id": fig_id})
                        
                        if img_doc:
                            # Check if base64Data is stored in the document
                            if img_doc.get("base64Data"):
                                base64_data = img_doc["base64Data"]
                            # If not, try to read from file_path and convert to base64
                            elif img_doc.get("file_path"):
                                import os
                                import base64
                                file_path = img_doc["file_path"]
                                if os.path.exists(file_path):
                                    try:
                                        with open(file_path, "rb") as f:
                                            image_bytes = f.read()
                                            base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
                                            content_type = img_doc.get("content_type", "image/jpeg")
                                            if not content_type.startswith("image/"):
                                                content_type = "image/jpeg"
                                            base64_data = f"data:{content_type};base64,{base64_encoded}"
                                    except Exception as file_err:
                                        logger.error(f"Failed to read image file {file_path}: {file_err}")
                    
                    enriched_figures.append({
                        "id": fig_id,
                        "url": f"/api/v1/images/{fig_id}",
                        "base64Data": base64_data,
                        "description": (fig_ref.get("description", "") if isinstance(fig_ref, dict) else ""),
                        "type": "diagram"
                    })
                except Exception as fig_err:
                    logger.error(f"Error processing figure: {fig_err}")
            
            question_dict["question_figures"] = enriched_figures
            
            # === ENHANCED: Load base64 image data for option images ===
            enriched_images = []
            for img_ref in question_dict.get("images", []) or []:
                try:
                    img_id = img_ref.get("id") if isinstance(img_ref, dict) else img_ref
                    base64_data = None
                    
                    # First check if base64Data is already embedded
                    if isinstance(img_ref, dict) and img_ref.get("base64Data"):
                        base64_data = img_ref["base64Data"]
                    else:
                        # Try to get from images collection
                        if is_b2c:
                            img_doc = await db.b2c_find_one("images", {"_id": img_id})
                        else:
                            img_doc = await db.mongo_find_one("images", {"_id": img_id})
                        
                        if img_doc:
                            if img_doc.get("base64Data"):
                                base64_data = img_doc["base64Data"]
                            elif img_doc.get("file_path"):
                                import os
                                import base64
                                file_path = img_doc["file_path"]
                                if os.path.exists(file_path):
                                    try:
                                        with open(file_path, "rb") as f:
                                            image_bytes = f.read()
                                            base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
                                            content_type = img_doc.get("content_type", "image/jpeg")
                                            if not content_type.startswith("image/"):
                                                content_type = "image/jpeg"
                                            base64_data = f"data:{content_type};base64,{base64_encoded}"
                                    except Exception as file_err:
                                        logger.error(f"Failed to read option image file {file_path}: {file_err}")
                    
                    enriched_images.append({
                        "id": img_id,
                        "url": f"/api/v1/images/{img_id}",
                        "base64Data": base64_data,
                        "description": (img_ref.get("description", "") if isinstance(img_ref, dict) else ""),
                        "type": img_ref.get("type", "option") if isinstance(img_ref, dict) else "option"
                    })
                except Exception as img_err:
                    logger.error(f"Error processing image: {img_err}")

            question_dict["images"] = enriched_images

            question_id = str(question_dict.get("id") or "")
            if include_worked_answers:
                question_dict["mapped_worked_answer"] = mappings_by_question_id.get(question_id)

            serialized_questions.append(question_dict)

        return {
            "document_id": document_id,
            "document_title": document["title"],
            "questions_count": len(serialized_questions),
            "answer_sheet_ocr_status": document.get("answer_sheet_ocr_status"),
            "answer_sheet_mapped_answers_count": document.get("answer_sheet_mapped_answers_count"),
            "has_answer_sheet": bool(document.get("answer_sheet_path")),
            "answer_solution_mode": document.get("answer_solution_mode"),
            "generated_solutions_status": document.get("generated_solutions_status"),
            "generated_solutions_count": document.get("generated_solutions_count"),
            "questions": serialized_questions
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document questions error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document questions"
        )


@router.get("/documents/{document_id}/images")
@limiter.limit("60/minute")
async def get_document_images(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Get all images extracted from a specific document"""
    try:
        # Check if B2C admin or B2C user
        user_type = current_user.get("user_type")
        is_b2c = user_type in ["b2c_admin", "b2c_user"]

        # Verify document exists in appropriate database
        if is_b2c:
            document = await db.b2c_find_one("documents", {"document_id": document_id})
        else:
            document = await db.mongo_find_one("documents", {"document_id": document_id})
            
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        # Access control
        if user_type == "student":
            student_admin_id = str(current_user.get("admin_id")) if current_user.get("admin_id") is not None else None
            document_admin_id = document.get("admin_id")
            document_admin_id_str = str(document_admin_id) if document_admin_id is not None else None

            from config_async import DEBUG_MODE as _DEBUG_MODE
            if student_admin_id != document_admin_id_str and not _DEBUG_MODE:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have access to this document"
                )
        elif not is_b2c:
            # Regular admin
            admin_id = str(current_user.get("user_id")) if current_user.get("user_id") is not None else None
            document_admin_id = document.get("admin_id")
            document_admin_id_str = str(document_admin_id) if document_admin_id is not None else None

            from config_async import DEBUG_MODE as _DEBUG_MODE
            if admin_id != document_admin_id_str and not _DEBUG_MODE:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have access to this document"
                )

        # Get images for this document
        if is_b2c:
            images = await db.b2c_find("images", {"source_pdf": document["filename"]})
        else:
            images = await db.mongo_find("images", {"source_pdf": document["filename"]})

        serialized_images = []
        for img in images:
            img_dict = {}
            for key, value in img.items():
                if isinstance(value, BsonObjectId):
                    img_dict[key] = str(value)
                elif isinstance(value, datetime):
                    img_dict[key] = value.isoformat()
                else:
                    img_dict[key] = value
            
            # Ensure url is present
            if "url" not in img_dict and "_id" in img_dict:
                img_dict["url"] = f"/api/v1/images/{img_dict['_id']}"
                
            serialized_images.append(img_dict)

        return {
            "document_id": document_id,
            "document_title": document["title"],
            "images_count": len(serialized_images),
            "images": serialized_images
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document images error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document images: {str(e)}"
        )

@router.post("/questions")
@limiter.limit("30/minute")
async def create_question(
    request: Request,
    question_id: str = Form(...),
    question_text: str = Form(...),
    correct_answer: str = Form(...),
    subject: str = Form(...),
    difficulty: str = Form(...),
    document_type: str = Form(...),
    course_plan: str = Form(...),
    standard: str = Form(...),
    question_type: str = Form(default="mcq"),  # mcq or integer
    evaluation_mode: str = Form(default="auto"),
    document_id: Optional[str] = Form(None),
    options_data: str = Form(default="[]"),  # JSON string of options metadata (optional for integer type)
    question_image: Optional[UploadFile] = File(None),
    option_images: List[UploadFile] = File(default=[]),
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """Create a new question with optional image uploads"""
    try:
        import uuid
        import json

        # Block creation if document is finalized for exam
        if document_id:
            is_b2c = is_b2c_admin(current_user)
            _doc = await (db.b2c_find_one("documents", {"document_id": document_id}) if is_b2c
                          else db.mongo_find_one("documents", {"document_id": document_id}))
            if _doc and _doc.get("exam_finalized"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cannot add questions to a finalized exam document",
                )

        # DCR documents are objective-only.
        if document_id and _doc and _doc.get("exam_mode") == "dcr" and question_type == "subjective":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="DCR exam documents only allow Objective questions",
            )

        # Generate unique question ID
        full_question_id = f"QST{question_id}"

        # Parse options metadata
        options_metadata = json.loads(options_data) if options_data else []
        normalized_evaluation_mode = str(evaluation_mode or "auto").strip().lower().replace("-", "_").replace(" ", "_")
        if normalized_evaluation_mode not in {"auto", "standard", "stem", "objective_stem", "case_study", "business_case", "mba_case"}:
            normalized_evaluation_mode = "auto"

        # Prepare question document
        question_doc = {
            "id": full_question_id,
            "text": question_text,  # Standard field name used by MCQ service
            "question_text": question_text,  # Alias for compatibility
            "question_type": question_type,  # Store question type (mcq or integer)
            "evaluation_mode": normalized_evaluation_mode,
            "options": [],  # Will be populated below (empty for integer type)
            "correct_answer": correct_answer,
            "subject": subject,
            "difficulty": difficulty,
            "document_type": document_type,
            "course_plan": course_plan,
            "standard": standard,
            "document_id": document_id,
            "created_by": current_user.get("user_id"),
            "created_at": datetime.utcnow(),
            "images": [],
            "question_figures": []
        }

        # Handle question image if provided
        if question_image and question_image.filename:
            logger.info(f"Uploading question image: {question_image.filename}")
            image_data = await question_image.read()

            # Convert to base64 for save_image_to_disk function and storage
            image_base64 = base64.b64encode(image_data).decode('utf-8')

            # Save to disk (split_composite=False for manually uploaded images)
            image_results = await save_image_to_disk(
                image_base64=image_base64,
                image_id=f"{full_question_id}_question",
                pdf_filename=document_id or full_question_id,
                db=db,
                user_id=current_user.get("user_id"),
                split_composite=False
            )

            # Add to question_figures with base64 data for frontend display
            for img_result in image_results:
                question_doc["question_figures"].append({
                    "id": img_result["id"],
                    "filename": img_result["filename"],
                    "path": img_result["path"],
                    "base64Data": image_base64,
                    "description": "",
                    "type": "diagram",
                    "metadata": {
                        "source": "manual_upload",
                        "uploadedAt": datetime.utcnow().isoformat()
                    }
                })

        # Process options with images
        option_image_index = 0
        for i, opt_meta in enumerate(options_metadata):
            if opt_meta.get("type") == "text":
                question_doc["options"].append(opt_meta.get("content", ""))
            elif opt_meta.get("type") == "image":
                # Get the corresponding image file
                if option_image_index < len(option_images):
                    opt_image = option_images[option_image_index]
                    option_image_index += 1

                    if opt_image and opt_image.filename:
                        logger.info(f"Uploading option {i} image: {opt_image.filename}")
                        image_data = await opt_image.read()

                        # Convert to base64 for save_image_to_disk function and storage
                        image_base64 = base64.b64encode(image_data).decode('utf-8')

                        # Save to disk (split_composite=False for manually uploaded images)
                        image_results = await save_image_to_disk(
                            image_base64=image_base64,
                            image_id=f"{full_question_id}_option_{i}",
                            pdf_filename=document_id or full_question_id,
                            db=db,
                            user_id=current_user.get("user_id"),
                            split_composite=False
                        )

                        # Add to images array with base64 data for frontend display
                        for img_result in image_results:
                            question_doc["images"].append({
                                "id": img_result["id"],
                                "filename": img_result["filename"],
                                "path": img_result["path"],
                                "base64Data": image_base64,
                                "description": f"Option {chr(65 + i)}",
                                "type": "option",
                                "option_index": i,
                                "metadata": {
                                    "source": "manual_upload",
                                    "uploadedAt": datetime.utcnow().isoformat()
                                }
                            })

                        # Store image reference in options
                        question_doc["options"].append(f"[IMAGE:{img_result['id']}]")
                    else:
                        question_doc["options"].append("[Image option]")
                else:
                    question_doc["options"].append("[Image option]")

        # Insert question into MongoDB
        await db.mongo_insert_one("questions", question_doc)

        logger.info(f"Created question {full_question_id} with {len(question_doc['question_figures'])} question images and {len(question_doc['images'])} option images")

        return {
            "message": "Question created successfully",
            "question_id": full_question_id
        }

    except Exception as e:
        logger.error(f"Create question error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create question: {str(e)}"
        )

@router.put("/questions/{question_id}")
@limiter.limit("30/minute")
async def update_question(
    request: Request,
    question_id: str,
    question_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """Update a question"""
    try:
        logger.info(f"📝 Update question request received for question_id={question_id}")
        logger.info(f"   Update data keys: {list(question_data.keys())}")
        logger.info(f"   User: {current_user.get('user_id')}")

        # Check if B2C admin
        user_type = current_user.get("user_type")
        is_b2c = user_type == "b2c_admin"

        # Get existing question from appropriate database
        if is_b2c:
            existing_question = await db.b2c_find_one("questions", {"id": question_id})
        else:
            existing_question = await db.mongo_find_one("questions", {"id": question_id})
        if not existing_question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Question {question_id} not found"
            )

        # Block edits if parent document is finalized for exam
        _q_doc_id = existing_question.get("document_id")
        if _q_doc_id:
            _parent_doc = await (db.b2c_find_one("documents", {"document_id": _q_doc_id}) if is_b2c
                                 else db.mongo_find_one("documents", {"document_id": _q_doc_id}))
            if _parent_doc and _parent_doc.get("exam_finalized"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cannot edit questions in a finalized exam document",
                )

        # Update fields
        update_data = {}
        if "text" in question_data:
            update_data["text"] = question_data["text"]
        if "options" in question_data:
            update_data["options"] = question_data["options"]
        if "correct_answer" in question_data:
            update_data["correct_answer"] = question_data["correct_answer"]
        if "subject" in question_data:
            update_data["subject"] = question_data["subject"]
        if "difficulty" in question_data:
            update_data["difficulty"] = question_data["difficulty"]
        if "document_type" in question_data:
            update_data["document_type"] = question_data["document_type"]
        evaluation_mode = question_data.get("evaluation_mode") or question_data.get("evaluationMode")
        if evaluation_mode is not None:
            normalized_evaluation_mode = str(evaluation_mode or "auto").strip().lower().replace("-", "_").replace(" ", "_")
            if normalized_evaluation_mode not in {"auto", "standard", "stem", "objective_stem", "case_study", "business_case", "mba_case"}:
                normalized_evaluation_mode = "auto"
            update_data["evaluation_mode"] = normalized_evaluation_mode
        # Helper to process and save new images
        async def process_new_images(images_list, id_prefix):
            processed_images = []
            for i, img in enumerate(images_list):
                # Check if this is a new image upload (has base64Data)
                if img.get("base64Data"):
                    try:
                        logger.info(f"Processing new image upload for question {question_id}")
                        # Generate a unique ID if the current one is temporary or missing
                        img_id = img.get("id")
                        if not img_id or img_id.startswith("img_") or "temp" in img_id:
                            img_id = f"{question_id}_{id_prefix}_{i}_{int(datetime.utcnow().timestamp())}"
                        
                        # Save to disk
                        saved_results = await save_image_to_disk(
                            image_base64=img["base64Data"],
                            image_id=img_id,
                            pdf_filename=existing_question.get("document_id") or existing_question.get("pdf_source") or question_id,
                            db=db,
                            user_id=current_user.get("user_id"),
                            split_composite=False, # Don't split manual uploads
                            is_b2c=is_b2c
                        )
                        
                        # Add saved images to the list
                        for saved_img in saved_results:
                            # Preserve description and type from the frontend object
                            saved_img["description"] = img.get("description", "")
                            saved_img["type"] = img.get("type", "diagram")
                            # IMPORTANT: Include base64Data so frontend can display it
                            saved_img["base64Data"] = img["base64Data"]
                            processed_images.append(saved_img)
                            
                    except Exception as e:
                        logger.error(f"Failed to save new image: {str(e)}")
                        # If save fails, we might want to skip it or let validation fail
                        # For now, we'll skip adding it to processed_images
                else:
                    # Existing image (no base64Data), keep as is
                    processed_images.append(img)
            return processed_images

        if "images" in question_data:
            # Process any new images first
            question_data["images"] = await process_new_images(question_data["images"], "opt")

            # Validate images before updating
            from utils.image_validator import validate_images_list
            valid_images, invalid_image_ids = await validate_images_list(question_data["images"], db, is_b2c)

            if invalid_image_ids:
                logger.warning(f"Question {question_id} update attempted with {len(invalid_image_ids)} invalid images. These will be filtered out: {invalid_image_ids}")

            update_data["images"] = valid_images

        # Support question_figures (diagram images) - separate from option images
        if "question_figures" in question_data:
            # Process any new images first
            question_data["question_figures"] = await process_new_images(question_data["question_figures"], "fig")

            # Validate question figures before updating
            from utils.image_validator import validate_images_list
            valid_figures, invalid_figure_ids = await validate_images_list(question_data["question_figures"], db, is_b2c)

            if invalid_figure_ids:
                logger.warning(f"Question {question_id} update attempted with {len(invalid_figure_ids)} invalid question figures. These will be filtered out: {invalid_figure_ids}")

            update_data["question_figures"] = valid_figures

        # Support enhanced_options (options with images/metadata)
        if "enhanced_options" in question_data:
            update_data["enhanced_options"] = question_data["enhanced_options"]

        if "points" in question_data:
            update_data["points"] = question_data["points"]
        if "penalty" in question_data:
            # Validate penalty max 50
            penalty = question_data["penalty"]
            if penalty > 50:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Penalty cannot exceed 50 points"
                )
            update_data["penalty"] = penalty
        
        # Support question_type (mcq or integer) - accept both snake_case and camelCase
        question_type = question_data.get("question_type") or question_data.get("questionType")
        if question_type:
            if question_type not in ["mcq", "integer", "subjective"]:
                question_type = "mcq"  # Default to MCQ
            if _parent_doc and _parent_doc.get("exam_mode") == "dcr" and question_type == "subjective":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="DCR exam documents only allow Objective questions",
                )
            update_data["question_type"] = question_type

        # Add updated timestamp
        update_data["updated_at"] = datetime.utcnow()
        update_data["updated_by"] = current_user.get("user_id")

        # Update in MongoDB (use appropriate database based on user type)
        if is_b2c:
            success = await db.b2c_update_one(
                "questions",
                {"id": question_id},
                {"$set": update_data}
            )
        else:
            success = await db.mongo_update_one(
                "questions",
                {"id": question_id},
                {"$set": update_data}
            )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No changes were made or question not found"
            )

        # If points were updated, recalculate document's total_points
        if "points" in update_data:
            # Use document_id consistently (not pdf_source)
            document_id = existing_question.get("document_id") or existing_question.get("pdf_source")
            if document_id:
                if is_b2c:
                    document = await db.b2c_find_one("documents", {"document_id": document_id})
                else:
                    document = await db.mongo_find_one("documents", {"document_id": document_id})
                if document and document.get("document_type") == "Test Series":
                    # Get all questions for this document using document_id
                    if is_b2c:
                        all_questions = await db.b2c_find("questions", {"document_id": document_id})
                    else:
                        all_questions = await db.mongo_find("questions", {"document_id": document_id})

                    # Fallback to pdf_source if document_id didn't find any
                    if not all_questions:
                        if is_b2c:
                            all_questions = await db.b2c_find("questions", {"pdf_source": document_id})
                        else:
                            all_questions = await db.mongo_find("questions", {"pdf_source": document_id})

                    total_points = sum(q.get("points", 4.0) for q in all_questions)  # Default 4 marks per question

                    # Update document's total_points
                    if is_b2c:
                        await db.b2c_update_one(
                            "documents",
                            {"document_id": document_id},
                            {"$set": {"total_points": total_points}}
                        )
                    else:
                        await db.mongo_update_one(
                            "documents",
                            {"document_id": document_id},
                            {"$set": {"total_points": total_points}}
                        )
                    logger.info(f"Updated document {document_id} total_points to {total_points}")

        return {
            "message": "Question updated successfully",
            "question_id": question_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update question error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update question: {str(e)}"
        )

@router.patch("/documents/{document_id}/questions/bulk-update")
@limiter.limit("10/minute")
async def bulk_update_questions(
    request: Request,
    document_id: str,
    update_data: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """Bulk update points and/or penalty for all questions in a document"""
    try:
        user_type = current_user.get("user_type")
        is_b2c = user_type == "b2c_admin"

        # Block bulk updates if document is finalized for exam
        _parent_doc = await (db.b2c_find_one("documents", {"document_id": document_id}) if is_b2c
                             else db.mongo_find_one("documents", {"document_id": document_id}))
        if _parent_doc and _parent_doc.get("exam_finalized"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot modify questions in a finalized exam document",
            )

        # Build the $set update
        set_fields = {}
        if "points" in update_data:
            pts = update_data["points"]
            if not isinstance(pts, (int, float)) or pts < 0:
                raise HTTPException(status_code=400, detail="points must be >= 0")
            set_fields["points"] = pts
        if "penalty" in update_data:
            pen = update_data["penalty"]
            if not isinstance(pen, (int, float)) or pen < 0:
                raise HTTPException(status_code=400, detail="penalty must be >= 0")
            set_fields["penalty"] = pen

        if not set_fields:
            raise HTTPException(status_code=400, detail="Provide at least one of: points, penalty")

        query = {"document_id": document_id}
        if is_b2c:
            # B2C has no update_many, loop through questions
            all_qs = await db.b2c_find("questions", query)
            modified = 0
            for q in all_qs:
                await db.b2c_update_one("questions", {"id": q["id"]}, {"$set": set_fields})
                modified += 1
        else:
            result = await db.mongo_update_many("questions", query, {"$set": set_fields})
            modified = result.modified_count if result else 0

        # Recalculate total_points on the document if points were changed
        if "points" in set_fields:
            if is_b2c:
                all_qs = await db.b2c_find("questions", query)
            else:
                all_qs = await db.mongo_find("questions", query)
            total = sum(q.get("points", set_fields.get("points", 4)) for q in all_qs)
            if is_b2c:
                await db.b2c_update_one("documents", {"document_id": document_id}, {"$set": {"total_points": total}})
            else:
                await db.mongo_update_one("documents", {"document_id": document_id}, {"$set": {"total_points": total}})

        return {
            "success": True,
            "message": f"Updated {modified} questions",
            "modified_count": modified,
            "updated_fields": set_fields
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk update questions error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to bulk update: {str(e)}")


@router.delete("/questions/{question_id}")
@limiter.limit("30/minute")
async def delete_question(
    request: Request,
    question_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """Delete a question and all its associated images and metadata"""
    try:
        # Check if B2C admin or B2C user
        user_type = current_user.get("user_type")
        is_b2c = user_type in ["b2c_admin", "b2c_user"]

        # Get the question first
        if is_b2c:
            question = await db.b2c_find_one("questions", {"id": question_id})
        else:
            question = await db.mongo_find_one("questions", {"id": question_id})

        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Question {question_id} not found"
            )

        # Block deletion if parent document is finalized for exam
        _q_doc_id = question.get("document_id")
        if _q_doc_id:
            _parent_doc = await (db.b2c_find_one("documents", {"document_id": _q_doc_id}) if is_b2c
                                 else db.mongo_find_one("documents", {"document_id": _q_doc_id}))
            if _parent_doc and _parent_doc.get("exam_finalized"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cannot delete questions from a finalized exam document",
                )

        # Delete associated images
        deleted_images_count = 0
        if question.get("images"):
            for image in question["images"]:
                image_id = image.get("id")
                if image_id:
                    # Delete from database
                    if is_b2c:
                        result = await db.b2c_delete_one("images", {"_id": image_id})
                    else:
                        result = await db.mongo_delete_one("images", {"_id": image_id})
                    
                    if result:
                        deleted_images_count += 1

                    # Delete file from disk
                    try:
                        file_path = image.get("path")
                        if file_path and os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete image file {image_id}: {str(e)}")

        # Delete question figures
        if question.get("question_figures"):
            for figure in question["question_figures"]:
                figure_id = figure.get("id")
                if figure_id:
                    # Delete from database
                    if is_b2c:
                        result = await db.b2c_delete_one("images", {"_id": figure_id})
                    else:
                        result = await db.mongo_delete_one("images", {"_id": figure_id})
                    
                    if result:
                        deleted_images_count += 1

                    # Delete file from disk
                    try:
                        file_path = figure.get("path")
                        if file_path and os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete figure file {figure_id}: {str(e)}")

        # Delete the question from MongoDB
        if is_b2c:
            result = await db.b2c_delete_one("questions", {"id": question_id})
        else:
            result = await db.mongo_delete_one("questions", {"id": question_id})

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Question {question_id} not found"
            )

        logger.info(f"Deleted question {question_id} and {deleted_images_count} associated images")

        return {
            "message": "Question deleted successfully",
            "question_id": question_id,
            "deleted_images": deleted_images_count
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete question error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete question: {str(e)}"
        )

@router.get("/documents/{document_id}/images")
@limiter.limit("60/minute")
async def get_document_images(
    request: Request,
    document_id: str,
    include_orphaned: bool = Query(False, description="Include images that don't exist on disk"),
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get all images extracted from a specific document.
    By default, filters out orphaned images (missing from filesystem).
    """
    try:
        from utils.image_validator import validate_image_exists

        # Verify document exists
        document = await db.mongo_find_one("documents", {"document_id": document_id})
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        # Get images for this document
        images = await db.mongo_find("images", {"source_pdf": document["filename"]})

        # Convert ObjectId to string and optionally filter orphaned images
        serialized_images = []
        orphaned_count = 0

        for img in images:
            image_id = str(img.get("_id", ""))

            # Check if image exists (unless include_orphaned is True)
            if not include_orphaned:
                exists = await validate_image_exists(image_id, db)
                if not exists:
                    orphaned_count += 1
                    logger.debug(f"Skipping orphaned image {image_id}")
                    continue

            image_dict = {}
            for key, value in img.items():
                if isinstance(value, BsonObjectId):
                    image_dict[key] = str(value)
                elif isinstance(value, datetime):
                    image_dict[key] = value.isoformat()
                else:
                    image_dict[key] = value
            serialized_images.append(image_dict)

        return {
            "document_id": document_id,
            "document_title": document["title"],
            "images_count": len(serialized_images),
            "total_in_db": len(images),
            "orphaned_count": orphaned_count,
            "images": serialized_images
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document images error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document images"
        )

@router.post("/documents/{document_id}/clean-orphaned-images")
@limiter.limit("10/minute")
async def clean_document_orphaned_images(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Clean orphaned image references from all questions in a document.
    Removes image references that don't exist in database or filesystem.
    """
    try:
        from utils.image_validator import get_orphaned_images_in_document, clean_question_images

        # Verify document exists
        document = await db.mongo_find_one("documents", {"document_id": document_id})
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        # Find all orphaned images first
        orphaned_by_question = await get_orphaned_images_in_document(document_id, db)

        if not orphaned_by_question:
            return {
                "message": "No orphaned images found",
                "document_id": document_id,
                "questions_cleaned": 0,
                "total_images_removed": 0,
                "details": []
            }

        # Clean each affected question
        questions_cleaned = 0
        total_images_removed = 0
        details = []

        for question_id, orphaned_ids in orphaned_by_question.items():
            # Get question
            question = await db.mongo_find_one("questions", {"id": question_id})
            if not question:
                continue

            # Clean orphaned references
            cleaned_question, removed_count = await clean_question_images(question, db)

            if removed_count > 0:
                # Update question in database
                await db.mongo_update_one(
                    "questions",
                    {"id": question_id},
                    {"$set": {
                        "images": cleaned_question.get("images", []),
                        "question_figures": cleaned_question.get("question_figures", []),
                        "cleaned_at": datetime.utcnow(),
                        "cleaned_by": current_user.get("user_id")
                    }}
                )

                questions_cleaned += 1
                total_images_removed += removed_count

                details.append({
                    "question_id": question_id,
                    "removed_images": orphaned_ids,
                    "removed_count": removed_count
                })

                logger.info(f"Cleaned {removed_count} orphaned images from question {question_id}")

        return {
            "message": f"Successfully cleaned {total_images_removed} orphaned image references",
            "document_id": document_id,
            "questions_cleaned": questions_cleaned,
            "total_images_removed": total_images_removed,
            "details": details
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clean orphaned images error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clean orphaned images: {str(e)}"
        )

@router.post("/questions/{question_id}/clean-orphaned-images")
@limiter.limit("20/minute")
async def clean_question_orphaned_images(
    request: Request,
    question_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Clean orphaned image references from a specific question.
    Removes image references that don't exist in database or filesystem.
    """
    try:
        from utils.image_validator import clean_question_images, get_orphaned_images_in_question

        # Get question
        question = await db.mongo_find_one("questions", {"id": question_id})
        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Question {question_id} not found"
            )

        # Get orphaned images first
        orphaned_ids = await get_orphaned_images_in_question(question_id, db)

        if not orphaned_ids:
            return {
                "message": "No orphaned images found",
                "question_id": question_id,
                "removed_count": 0,
                "orphaned_images": []
            }

        # Clean orphaned references
        cleaned_question, removed_count = await clean_question_images(question, db)

        if removed_count > 0:
            # Update question in database
            await db.mongo_update_one(
                "questions",
                {"id": question_id},
                {"$set": {
                    "images": cleaned_question.get("images", []),
                    "question_figures": cleaned_question.get("question_figures", []),
                    "cleaned_at": datetime.utcnow(),
                    "cleaned_by": current_user.get("user_id")
                }}
            )

            logger.info(f"Cleaned {removed_count} orphaned images from question {question_id}")

        return {
            "message": f"Successfully removed {removed_count} orphaned image references",
            "question_id": question_id,
            "removed_count": removed_count,
            "orphaned_images": orphaned_ids
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clean question orphaned images error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clean orphaned images: {str(e)}"
        )

@router.get("/documents/{document_id}/orphaned-images")
@limiter.limit("30/minute")
async def get_document_orphaned_images(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get all orphaned image references in a document without cleaning them.
    Useful for inspection before cleanup.
    """
    try:
        from utils.image_validator import get_orphaned_images_in_document

        # Verify document exists
        document = await db.mongo_find_one("documents", {"document_id": document_id})
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        # Find all orphaned images
        orphaned_by_question = await get_orphaned_images_in_document(document_id, db)

        total_orphaned = sum(len(ids) for ids in orphaned_by_question.values())

        return {
            "document_id": document_id,
            "document_title": document.get("title", ""),
            "total_orphaned_images": total_orphaned,
            "affected_questions": len(orphaned_by_question),
            "orphaned_by_question": orphaned_by_question
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get orphaned images error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get orphaned images: {str(e)}"
        )

@router.delete("/documents/{document_id}")
@limiter.limit("10/minute")
async def delete_document(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """Delete document and all associated data (cascading delete)"""
    try:
        # Check if B2C admin or B2C user
        user_type = current_user.get("user_type")
        is_b2c = user_type in ["b2c_admin", "b2c_user"]

        # Get document metadata
        if is_b2c:
            document = await db.b2c_find_one("documents", {"document_id": document_id})
        else:
            document = await db.mongo_find_one("documents", {"document_id": document_id})

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        # Delete PDF file
        from pathlib import Path
        backend_dir = Path(os.getcwd())
        stored_path = document["file_path"].replace("\\", "/")
        file_path = backend_dir / stored_path
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted PDF file: {file_path}")

        answer_sheet_path = str(document.get("answer_sheet_path") or "").replace("\\", "/")
        if answer_sheet_path and not answer_sheet_path.startswith("s3://"):
            answer_sheet_file_path = backend_dir / answer_sheet_path
            if answer_sheet_file_path.exists():
                answer_sheet_file_path.unlink()
                logger.info(f"Deleted answer sheet PDF file: {answer_sheet_file_path}")

        # Delete all questions associated with this document
        if is_b2c:
            questions = await db.b2c_find("questions", {"document_id": document_id})
        else:
            questions = await db.mongo_find("questions", {"document_id": document_id})

        logger.info(f"Found {len(questions)} questions to delete for document {document_id}")

        # Delete questions from MongoDB
        try:
            if is_b2c:
                q_result = await db.b2c_delete_many("questions", {"document_id": document_id})
            else:
                q_result = await db.mongo_delete_many("questions", {"document_id": document_id})
            logger.info(f"Deleted {len(questions)} questions from MongoDB for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to delete questions from MongoDB: {str(e)}")
            raise

        # Delete all images associated with this document
        if is_b2c:
            images = await db.b2c_find("images", {"source_pdf": document["filename"]})
        else:
            images = await db.mongo_find("images", {"source_pdf": document["filename"]})

        logger.info(f"Found {len(images)} images to delete for document {document_id}")

        for image in images:
            # Delete image file
            if "file_path" in image and os.path.exists(image["file_path"]):
                try:
                    os.remove(image["file_path"])
                    logger.debug(f"Deleted image file: {image['file_path']}")
                except Exception as e:
                    logger.warning(f"Failed to delete image file {image['file_path']}: {str(e)}")

        # Delete images from MongoDB
        try:
            if is_b2c:
                img_result = await db.b2c_delete_many("images", {"source_pdf": document["filename"]})
            else:
                img_result = await db.mongo_delete_many("images", {"source_pdf": document["filename"]})
            logger.info(f"Deleted {len(images)} images from MongoDB for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to delete images from MongoDB: {str(e)}")
            raise

        # Delete document metadata
        try:
            if is_b2c:
                doc_result = await db.b2c_delete_one("documents", {"document_id": document_id})
            else:
                doc_result = await db.mongo_delete_one("documents", {"document_id": document_id})
            logger.info(f"Deleted document {document_id} from MongoDB")
        except Exception as e:
            logger.error(f"Failed to delete document from MongoDB: {str(e)}")
            raise

        return {
            "message": f"Document {document_id} and all associated data deleted successfully",
            "deleted_questions": len(questions),
            "deleted_images": len(images)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete document error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


# =============================================================================
# REGION-BASED OCR ENDPOINTS
# Manual question segmentation and region-based OCR processing
# =============================================================================

VALID_REGION_SCOPES = {"question", "answer"}


def _normalise_region_scope(region_scope: str) -> str:
    scope = (region_scope or "question").strip().lower()
    if scope not in VALID_REGION_SCOPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="region_scope must be either 'question' or 'answer'"
        )
    return scope


def _document_regions_filter(document_id: str, region_scope: str) -> Dict[str, Any]:
    if region_scope == "question":
        return {
            "document_id": document_id,
            "$or": [
                {"region_scope": "question"},
                {"region_scope": {"$exists": False}},
                {"region_scope": None},
            ],
        }
    return {"document_id": document_id, "region_scope": region_scope}


class QuestionRegion(BaseModel):
    """Represents a bounding box for a question region on a PDF page"""
    id: str
    pageNumber: int
    x: float  # Percentage (0-100)
    y: float  # Percentage (0-100)
    width: float  # Percentage (0-100)
    height: float  # Percentage (0-100)
    order: int
    label: str
    hasSubQuestions: bool = False
    notes: Optional[str] = None
    ocrStatus: Optional[str] = None  # pending, processing, completed, error
    extractedText: Optional[str] = None
    createdAt: str
    updatedAt: str

class DocumentRegionsRequest(BaseModel):
    """Request body for saving document regions"""
    regions: List[QuestionRegion]
    excludedPages: List[int] = Field(default_factory=list)

class DocumentRegionsResponse(BaseModel):
    """Response for document regions"""
    documentId: str
    regions: List[QuestionRegion]
    excludedPages: List[int] = Field(default_factory=list)
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None
    createdBy: Optional[str] = None

class RegionOCRRequest(BaseModel):
    """Request for processing specific regions with OCR"""
    regionIds: Optional[List[str]] = None  # If None, process all regions
    replaceExisting: bool = True
    regionScope: str = "question"
    documentAnchorText: Optional[str] = None


class GenerateSolutionsRequest(BaseModel):
    """Request for generating worked solutions from extracted MCQ questions."""
    confirmQuestionsReviewed: bool = False
    replaceExisting: bool = True
    batchSize: int = Field(default=8, ge=1, le=20)

class RegionOCRResult(BaseModel):
    """Result of OCR processing for a single region"""
    regionId: str
    success: bool
    extractedText: Optional[str] = None
    extractedOptions: Optional[List[str]] = None
    extractedImages: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

class RegionOCRResponse(BaseModel):
    """Response for region-based OCR processing"""
    success: bool
    documentId: str
    processedRegions: int
    successfulRegions: int
    failedRegions: int
    results: List[RegionOCRResult]


@router.get("/documents/{document_id}/regions")
async def get_document_regions(
    document_id: str,
    region_scope: str = Query("question"),
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get all saved question regions for a document.
    
    Returns the list of bounding box regions that have been manually drawn
    on the PDF pages for question segmentation.
    """
    try:
        region_scope = _normalise_region_scope(region_scope)
        is_b2c = is_b2c_admin(current_user)
        
        # Verify document exists
        if is_b2c:
            document = await db.b2c_find_one("documents", {"document_id": document_id})
        else:
            document = await db.mongo_find_one("documents", {"document_id": document_id})
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        if region_scope == "answer" and not document.get("answer_sheet_path"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No answer sheet uploaded for this document"
            )
        
        # Get regions for this document
        regions_filter = _document_regions_filter(document_id, region_scope)
        if is_b2c:
            regions_doc = await db.b2c_find_one("document_regions", regions_filter)
        else:
            regions_doc = await db.mongo_find_one("document_regions", regions_filter)
        
        if not regions_doc:
            return {
                "documentId": document_id,
                "regionScope": region_scope,
                "regions": [],
                "excludedPages": [],
                "createdAt": None,
                "updatedAt": None,
                "createdBy": None
            }
        
        return {
            "documentId": document_id,
            "regionScope": region_scope,
            "regions": regions_doc.get("regions", []),
            "excludedPages": regions_doc.get("excluded_pages", []),
            "createdAt": regions_doc.get("created_at"),
            "updatedAt": regions_doc.get("updated_at"),
            "createdBy": regions_doc.get("created_by")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document regions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document regions: {str(e)}"
        )


@router.get("/documents/{document_id}/answer-mappings")
async def get_document_answer_mappings(
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """Return question-to-worked-answer mappings for admin/tutor review."""
    try:
        is_b2c = is_b2c_admin(current_user)

        if is_b2c:
            document = await db.b2c_find_one("documents", {"document_id": document_id})
        else:
            document = await db.mongo_find_one("documents", {"document_id": document_id})

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        if not is_b2c:
            from config_async import DEBUG_MODE as _DEBUG_MODE

            document_admin_id = document.get("admin_id")
            document_admin_id_str = str(document_admin_id) if document_admin_id is not None else None
            user_type = current_user.get("user_type")

            if user_type == "admin":
                admin_id = str(current_user.get("user_id")) if current_user.get("user_id") is not None else None
                if admin_id != document_admin_id_str and not _DEBUG_MODE:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="You don't have access to this document"
                    )
            elif user_type == "tutor":
                tutor_admin_id = str(current_user.get("admin_id")) if current_user.get("admin_id") is not None else None
                if tutor_admin_id != document_admin_id_str and not _DEBUG_MODE:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="You don't have access to this document"
                    )

        if is_b2c:
            questions = await db.b2c_find("questions", {"document_id": document_id})
            mappings = await db.b2c_find("answer_question_mappings", {"document_id": document_id})
            answer_regions_doc = await db.b2c_find_one(
                "document_regions",
                _document_regions_filter(document_id, "answer")
            )
        else:
            questions = await db.mongo_find("questions", {"document_id": document_id})
            mappings = await db.mongo_find("answer_question_mappings", {"document_id": document_id})
            answer_regions_doc = await db.mongo_find_one(
                "document_regions",
                _document_regions_filter(document_id, "answer")
            )

        mappings_by_question_id: Dict[str, Dict[str, Any]] = {}
        for mapping in mappings:
            question_id = str(mapping.get("question_id") or mapping.get("question_region_id") or "")
            if question_id:
                mappings_by_question_id[question_id] = _serialize_answer_mapping(mapping)

        def _question_sort_key(question: Dict[str, Any]) -> tuple:
            region = question.get("region_metadata") or {}
            return (
                int(question.get("page_number") or region.get("page") or 0),
                float(region.get("y") or 0),
                str(question.get("id") or ""),
            )

        answer_regions = (answer_regions_doc or {}).get("regions", []) or []
        extracted_answer_count = len(
            [
                region
                for region in answer_regions
                if str(region.get("extractedText") or "").strip()
            ]
        )
        generated_solution_count = len(
            [
                mapping
                for mapping in mappings
                if mapping.get("source") == "ai_generated" and mapping.get("answer_text")
            ]
        )
        answer_count = extracted_answer_count or generated_solution_count

        rows: List[Dict[str, Any]] = []
        for index, question in enumerate(sorted(questions, key=_question_sort_key), start=1):
            question_id = str(question.get("id") or "")
            mapping = mappings_by_question_id.get(question_id)
            rows.append({
                "question_index": index,
                "question_id": question_id,
                "question_text": question.get("text") or question.get("question_text") or "",
                "correct_answer": question.get("correct_answer"),
                "mapped_answer": mapping,
            })

        mapped_count = len(
            [
                mapping
                for mapping in mappings_by_question_id.values()
                if mapping.get("answer_text") and not mapping.get("manual_review_required")
            ]
        )

        return {
            "documentId": document_id,
            "hasAnswerSheet": bool(document.get("answer_sheet_path")),
            "questionOcrStatus": document.get("ocr_status"),
            "answerSheetOcrStatus": document.get("answer_sheet_ocr_status") or "not_processed",
            "answerSolutionMode": document.get("answer_solution_mode") or ("upload" if document.get("answer_sheet_path") else "none"),
            "generatedSolutionsStatus": document.get("generated_solutions_status"),
            "generatedSolutionCount": generated_solution_count,
            "questionCount": len(questions),
            "answerCount": answer_count,
            "mappedCount": mapped_count,
            "mappings": rows,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting answer mappings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get answer mappings: {str(e)}"
        )


@router.post("/documents/{document_id}/generate-solutions")
async def generate_document_solutions(
    document_id: str,
    generation_request: GenerateSolutionsRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    """Generate worked explanations from extracted MCQ question data."""
    started_at = datetime.utcnow()
    is_b2c = is_b2c_admin(current_user)
    try:
        if not generation_request.confirmQuestionsReviewed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Confirm that OCR has identified the questions/options correctly before generating solutions.",
            )

        if is_b2c:
            document = await db.b2c_find_one("documents", {"document_id": document_id})
        else:
            document = await db.mongo_find_one("documents", {"document_id": document_id})

        if not document:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

        if not is_b2c:
            from config_async import DEBUG_MODE as _DEBUG_MODE

            document_admin_id = document.get("admin_id")
            document_admin_id_str = str(document_admin_id) if document_admin_id is not None else None
            user_type = current_user.get("user_type")
            if user_type == "admin":
                admin_id = str(current_user.get("user_id")) if current_user.get("user_id") is not None else None
                if admin_id != document_admin_id_str and not _DEBUG_MODE:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this document")
            elif user_type == "tutor":
                tutor_admin_id = str(current_user.get("admin_id")) if current_user.get("admin_id") is not None else None
                if tutor_admin_id != document_admin_id_str and not _DEBUG_MODE:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this document")

        if document.get("document_type") != "Test Series" or document.get("exam_mode"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Auto-generated solutions are only available for online Test Series documents.",
            )
        if str(document.get("question_type") or "mcq").lower() != "mcq":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Auto-generated solutions require objective questions.",
            )
        if document.get("ocr_status") != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question paper OCR must be completed before generating solutions.",
            )

        if is_b2c:
            questions = await db.b2c_find("questions", {"document_id": document_id})
        else:
            questions = await db.mongo_find("questions", {"document_id": document_id})

        if not questions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No extracted questions found. Run question paper OCR first.",
            )

        missing_correct = [
            str(question.get("id") or index + 1)
            for index, question in enumerate(questions)
            if not _normalise_correct_answer_label(question.get("correct_answer"))
        ]
        if missing_correct:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"All questions must have a correct answer before generating solutions. Missing: {', '.join(missing_correct[:10])}",
            )

        def _question_sort_key(question: Dict[str, Any]) -> tuple:
            region = question.get("region_metadata") or {}
            return (
                int(question.get("page_number") or region.get("page") or 0),
                float(region.get("y") or 0),
                str(question.get("id") or ""),
            )

        sorted_questions = sorted(questions, key=_question_sort_key)

        status_update = {
            "answer_solution_mode": "auto",
            "generated_solutions_status": "processing",
            "generated_solutions_started_at": started_at,
            "generated_solutions_error": None,
        }
        if is_b2c:
            await db.b2c_update_one("documents", {"document_id": document_id}, {"$set": status_update})
        else:
            await db.mongo_update_one("documents", {"document_id": document_id}, {"$set": status_update})

        if not generation_request.replaceExisting:
            if is_b2c:
                existing_mappings = await db.b2c_find("answer_question_mappings", {"document_id": document_id})
            else:
                existing_mappings = await db.mongo_find("answer_question_mappings", {"document_id": document_id})
            existing_question_ids = {
                str(mapping.get("question_id") or "")
                for mapping in existing_mappings
                if mapping.get("answer_text")
            }
            sorted_questions = [
                question
                for question in sorted_questions
                if str(question.get("id") or "") not in existing_question_ids
            ]

        generated = 0
        manual_review = 0
        failed = 0
        batch_size = max(1, min(20, int(generation_request.batchSize or 8)))
        for offset in range(0, len(sorted_questions), batch_size):
            batch = sorted_questions[offset:offset + batch_size]
            batch_result = await generate_worked_solution_batch(
                questions=batch,
                document=document,
                gateway_context=_build_ai_gateway_context(
                    current_user=current_user,
                    db=db,
                    document_id=document_id,
                    region_scope="generated_solution_batch",
                    is_b2c=is_b2c,
                ),
            )
            solutions_by_id = {
                str(solution.get("question_id") or ""): solution
                for solution in batch_result.get("solutions", [])
            }
            for question in batch:
                question_id = str(question.get("id") or "")
                solution = solutions_by_id.get(question_id)
                if not solution:
                    failed += 1
                    continue
                answer_text = str(solution.get("answer_text") or "").strip()
                if not answer_text:
                    failed += 1
                    continue
                confidence = solution.get("confidence")
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = 0.5
                review_required = bool(solution.get("manual_review_required")) or not bool(solution.get("correct_option_verified"))
                if review_required:
                    manual_review += 1
                mapping = {
                    "mapping_id": f"{document_id}:{question_id}:generated",
                    "document_id": document_id,
                    "question_region_id": question_id,
                    "question_id": question_id,
                    "answer_region_id": f"generated:{question_id}",
                    "answer_text": answer_text,
                    "mapping_strategy": "ai_generated_solution",
                    "confidence": max(0.0, min(1.0, confidence)),
                    "manual_review_required": review_required,
                    "source": "ai_generated",
                    "correct_option_verified": bool(solution.get("correct_option_verified")),
                    "generation_notes": solution.get("notes") or "",
                    "generator_provider": batch_result.get("provider"),
                    "generator_model": batch_result.get("model"),
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                }
                query = {"document_id": document_id, "question_id": question_id}
                if is_b2c:
                    await db.b2c_update_one("answer_question_mappings", query, {"$set": mapping}, upsert=True)
                else:
                    await db.mongo_update_one("answer_question_mappings", query, {"$set": mapping}, upsert=True)
                generated += 1

        completed_update = {
            "answer_solution_mode": "auto",
            "generated_solutions_status": "completed",
            "generated_solutions_completed_at": datetime.utcnow(),
            "generated_solutions_count": generated,
            "generated_solutions_manual_review_count": manual_review,
            "generated_solutions_failed_count": failed,
            "answer_sheet_mapped_answers_count": generated - manual_review,
        }
        if is_b2c:
            await db.b2c_update_one("documents", {"document_id": document_id}, {"$set": completed_update})
        else:
            await db.mongo_update_one("documents", {"document_id": document_id}, {"$set": completed_update})

        return {
            "success": True,
            "documentId": document_id,
            "processedQuestions": len(sorted_questions),
            "generated": generated,
            "manualReviewRequired": manual_review,
            "failed": failed,
            "batchSize": batch_size,
        }

    except HTTPException:
        raise
    except Exception as e:
        error_update = {
            "generated_solutions_status": "error",
            "generated_solutions_error": str(e),
        }
        if is_b2c:
            await db.b2c_update_one("documents", {"document_id": document_id}, {"$set": error_update})
        else:
            await db.mongo_update_one("documents", {"document_id": document_id}, {"$set": error_update})
        logger.error(f"Generate solutions failed for {document_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate solutions: {e}",
        )


@router.post("/documents/{document_id}/regions")
async def save_document_regions(
    document_id: str,
    request: DocumentRegionsRequest,
    region_scope: str = Query("question"),
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Save question regions for a document.
    
    Stores the bounding box regions that have been manually drawn on the PDF pages.
    This replaces any existing regions for the document.
    """
    try:
        region_scope = _normalise_region_scope(region_scope)
        is_b2c = is_b2c_admin(current_user)
        
        # Verify document exists
        if is_b2c:
            document = await db.b2c_find_one("documents", {"document_id": document_id})
        else:
            document = await db.mongo_find_one("documents", {"document_id": document_id})
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        if region_scope == "answer" and not document.get("answer_sheet_path"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No answer sheet uploaded for this document"
            )
        
        now = datetime.utcnow().isoformat()
        
        excluded_pages = sorted({
            int(page)
            for page in (request.excludedPages or [])
            if isinstance(page, int) and page > 0
        })
        excluded_page_set = set(excluded_pages)

        # Convert regions to dict format and never persist regions for deleted pages
        regions_data = [region.dict() for region in request.regions]
        regions_data = [
            region
            for region in regions_data
            if int(region.get("pageNumber", 0) or 0) not in excluded_page_set
        ]
        
        # Prepare regions document
        regions_doc = {
            "document_id": document_id,
            "region_scope": region_scope,
            "regions": regions_data,
            "excluded_pages": excluded_pages,
            "created_by": current_user.get("user_id"),
            "updated_at": now
        }
        
        # Upsert regions document
        regions_filter = _document_regions_filter(document_id, region_scope)
        if is_b2c:
            existing = await db.b2c_find_one("document_regions", regions_filter)
            if existing:
                await db.b2c_update_one(
                    "document_regions",
                    {"_id": existing["_id"]},
                    {"$set": regions_doc}
                )
            else:
                regions_doc["created_at"] = now
                await db.b2c_insert_one("document_regions", regions_doc)
        else:
            existing = await db.mongo_find_one("document_regions", regions_filter)
            if existing:
                await db.mongo_update_one(
                    "document_regions",
                    {"_id": existing["_id"]},
                    {"$set": regions_doc}
                )
            else:
                regions_doc["created_at"] = now
                await db.mongo_insert_one("document_regions", regions_doc)
        
        logger.info(f"Saved {len(request.regions)} {region_scope} regions for document {document_id}")
        
        return {
            "success": True,
            "message": f"Saved {len(request.regions)} regions",
            "documentId": document_id,
            "regionScope": region_scope,
            "regionsCount": len(request.regions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving document regions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save document regions: {str(e)}"
        )


@router.delete("/documents/{document_id}/regions")
async def delete_document_regions(
    document_id: str,
    region_scope: str = Query("question"),
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Delete all regions for a document.
    """
    try:
        region_scope = _normalise_region_scope(region_scope)
        is_b2c = is_b2c_admin(current_user)
        regions_filter = _document_regions_filter(document_id, region_scope)
        
        if is_b2c:
            result = await db.b2c_delete_one("document_regions", regions_filter)
        else:
            result = await db.mongo_delete_one("document_regions", regions_filter)
        
        return {
            "success": True,
            "message": f"Deleted {region_scope} regions for document {document_id}"
        }
        
    except Exception as e:
        logger.error(f"Error deleting document regions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document regions: {str(e)}"
        )


async def extract_region_from_pdf(
    pdf_content: bytes,
    page_number: int,
    bbox: Dict[str, float],
    region_id: str,
    *,
    region_scope: str = "question",
    gateway_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract a specific region from a PDF page and process it with OCR.

    Uses Sarvam AI Document Intelligence to extract text from the region, and
    attaches only images that intersect the selected region on the original PDF
    page. The original-page intersection avoids duplicated image resources from
    clipped region PDFs.

    Args:
        pdf_content: Raw PDF bytes
        page_number: Page number (1-indexed)
        bbox: Bounding box as percentages {x, y, width, height}
        region_id: ID of the region being processed

    Returns:
        Dict with extracted text, images list, and region screenshot
    """
    try:
        try:
            crop_result = RegionCropService().crop(
                pdf_content=pdf_content,
                page_number=page_number,
                bbox=bbox,
                region_id=region_id,
                region_scope=region_scope,
            )
        except ValueError as crop_err:
            return {"success": False, "error": str(crop_err)}

        region_pdf_bytes = crop_result["region_pdf_bytes"]
        region_img_base64 = crop_result["region_png_base64"]
        region_embedded_images = crop_result.get("embedded_images", [])
        layout_report = LayoutPreflightService().analyze(
            region_id=region_id,
            text_items=crop_result.get("text_items", []),
            embedded_images=region_embedded_images,
        )

        # OCR the cropped region PDF
        logger.info(f"Calling OCR for region {region_id} (PDF size: {len(region_pdf_bytes)} bytes)")

        try:
            ocr_result = await call_sarvam_ocr(
                region_pdf_bytes,
                gateway_context=gateway_context,
            )
        except Exception as ocr_err:
            logger.error(f"Sarvam OCR failed for region {region_id}: {ocr_err}")
            return {
                "success": False,
                "error": f"OCR failed for region: {ocr_err}",
                "regionImageBase64": region_img_base64,
                "layoutReport": layout_report,
                "cropMetadata": crop_result.get("crop_metadata", {}),
            }

        # Extract text and images from OCR result
        extracted_text = ""
        provider_images = []

        for page_data in ocr_result.get("pages", []):
            page_markdown = page_data.get("markdown", "")
            extracted_text += page_markdown + "\n"

            for img in page_data.get("images", []):
                if img.get("image_base64"):
                    provider_images.append({
                        "id": img.get("id", f"img-{len(provider_images)}"),
                        "base64": img.get("image_base64"),
                        "top_left_x": img.get("top_left_x", 0),
                        "top_left_y": img.get("top_left_y", 0),
                        "bottom_right_x": img.get("bottom_right_x", 0),
                        "bottom_right_y": img.get("bottom_right_y", 0),
                        "source": "ocr_provider"
                    })

        extracted_text = extracted_text.strip()

        # Prefer images proven to intersect this manual segment in the original
        # page. Only fall back to provider images when the OCR markdown explicitly
        # references them; unreferenced provider images from clipped PDFs can be
        # page-resource duplicates.
        import re
        referenced_ids = set(re.findall(r'!\[[^\]]*\]\(([^)]+)\)', extracted_text))
        candidate_images = list(region_embedded_images)

        if not candidate_images:
            for img in provider_images:
                img_id = img.get("id")
                if img_id and img_id in referenced_ids:
                    candidate_images.append(img)

        unique_figures = []
        seen_image_keys = set()
        for img in candidate_images:
            img_id = img.get("id")
            image_key = img.get("base64", "")[:128] or img_id
            if not image_key or image_key in seen_image_keys:
                continue
            seen_image_keys.add(image_key)
            img["referencedInMarkdown"] = img_id in referenced_ids
            unique_figures.append(img)

        logger.info(
            f"Region {region_id}: {len(extracted_text)} chars, "
            f"{len(provider_images)} OCR images, {len(region_embedded_images)} "
            f"region-matched PDF images, {len(unique_figures)} retained figures"
        )

        return {
            "success": True,
            "extractedText": extracted_text,
            "extractedImages": unique_figures,
            "regionImageBase64": region_img_base64,
            "ocrResult": ocr_result,
            "layoutReport": layout_report,
            "cropMetadata": crop_result.get("crop_metadata", {}),
            "textItems": crop_result.get("text_items", []),
            "_regionPdfBytes": region_pdf_bytes,
        }
                
    except ImportError as e:
        logger.error(f"Missing dependency for region extraction: {e}")
        return {
            "success": False,
            "error": f"Missing dependency: {e}. Install PyMuPDF with 'pip install pymupdf'"
        }
    except Exception as e:
        logger.error(f"Error extracting region {region_id}: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/documents/{document_id}/regions/process-ocr")
@limiter.limit("5/minute")
async def process_regions_ocr(
    request: Request,
    document_id: str,
    ocr_request: RegionOCRRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """
    Process OCR on specific regions of a document.
    
    This endpoint:
    1. Retrieves saved regions for the document
    2. For each region, extracts that portion of the PDF page
    3. Sends each region to Sarvam AI OCR
    4. Parses the OCR result and creates questions
    5. Returns the results
    """
    ocr_started_at = datetime.utcnow()
    try:
        region_scope = _normalise_region_scope(ocr_request.regionScope)
        is_answer_scope = region_scope == "answer"
        is_b2c = is_b2c_admin(current_user)
        
        # Get document
        if is_b2c:
            document = await db.b2c_find_one("documents", {"document_id": document_id})
        else:
            document = await db.mongo_find_one("documents", {"document_id": document_id})
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        status_field = "answer_sheet_ocr_status" if is_answer_scope else "ocr_status"
        if document.get(status_field) == "processing":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="OCR processing already in progress"
            )

        if is_answer_scope and not document.get("answer_sheet_path"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No answer sheet uploaded for this document"
            )
        
        # Get regions
        regions_filter = _document_regions_filter(document_id, region_scope)
        if is_b2c:
            regions_doc = await db.b2c_find_one("document_regions", regions_filter)
        else:
            regions_doc = await db.mongo_find_one("document_regions", regions_filter)
        
        if not regions_doc or not regions_doc.get("regions"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No {region_scope} regions defined for this document. Please draw regions first."
            )
        
        excluded_pages = {
            int(page)
            for page in (regions_doc.get("excluded_pages", []) or [])
            if isinstance(page, int) and page > 0
        }
        all_regions = [
            region
            for region in regions_doc.get("regions", [])
            if int(region.get("pageNumber", 0) or 0) not in excluded_pages
        ]
        
        # Filter regions if specific IDs provided
        if ocr_request.regionIds:
            regions_to_process = [r for r in all_regions if r['id'] in ocr_request.regionIds]
        else:
            regions_to_process = all_regions
        
        if not regions_to_process:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No matching regions found to process"
            )

        processing_all_regions = not ocr_request.regionIds
        
        # Load PDF content
        from pathlib import Path as _Path
        backend_dir = _Path(os.getcwd())
        file_path = document.get("answer_sheet_path" if is_answer_scope else "file_path", "")
        
        pdf_content = None
        
        # Check if S3 storage
        if file_path.startswith("s3://"):
            try:
                from utils.s3_storage import download_file as s3_download_file
                pdf_content = await s3_download_file(file_path)
            except Exception as s3_err:
                logger.error(f"Failed to download PDF from S3: {s3_err}")
        else:
            # Try local file
            resolved_path = _resolve_answer_sheet_file_path(document) if is_answer_scope else _resolve_document_file_path(document)
            local_path = resolved_path or (backend_dir / file_path if not _Path(file_path).is_absolute() else _Path(file_path))
            if local_path.exists():
                async with aiofiles.open(str(local_path), "rb") as f:
                    pdf_content = await f.read()
        
        if not pdf_content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Answer sheet PDF file not found" if is_answer_scope else "PDF file not found"
            )
        
        # Update document status to processing
        processing_status_update = {status_field: "processing"}
        if is_answer_scope:
            processing_status_update["answer_sheet_document_anchor_text"] = (
                ocr_request.documentAnchorText.strip()
                if ocr_request.documentAnchorText
                else None
            )
        if is_b2c:
            await db.b2c_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": processing_status_update}
            )
        else:
            await db.mongo_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": processing_status_update}
            )

        if ocr_request.replaceExisting and not is_answer_scope:
            await delete_existing_ocr_outputs(
                document=document,
                current_user=current_user,
                db=db,
                question_ids=None if processing_all_regions else [r["id"] for r in regions_to_process],
                delete_images=processing_all_regions,
            )
        
        # Process each region
        results = []
        successful = 0
        failed = 0
        total_images_saved = 0  # Track total images saved
        question_regions_for_answer: List[Dict[str, Any]] = []
        questions_by_id_for_answer: Dict[str, Dict[str, Any]] = {}
        answer_region_order_by_id: Dict[str, int] = {}

        if is_answer_scope:
            question_regions_filter = _document_regions_filter(document_id, "question")
            if is_b2c:
                question_regions_doc = await db.b2c_find_one("document_regions", question_regions_filter)
                question_docs_for_answer = await db.b2c_find("questions", {"document_id": document_id})
            else:
                question_regions_doc = await db.mongo_find_one("document_regions", question_regions_filter)
                question_docs_for_answer = await db.mongo_find("questions", {"document_id": document_id})

            question_regions_for_answer = sorted(
                (question_regions_doc or {}).get("regions", []) or [],
                key=_region_sort_key,
            )
            questions_by_id_for_answer = {
                str(question.get("id")): question
                for question in question_docs_for_answer
                if question.get("id")
            }
            answer_region_order_by_id = {
                str(region.get("id")): index
                for index, region in enumerate(sorted(all_regions, key=_region_sort_key))
                if region.get("id")
            }
        
        for region in regions_to_process:
            region_id = region['id']
            
            logger.info(f"Processing region {region_id} on page {region['pageNumber']}")
            
            # Update region status
            region['ocrStatus'] = 'processing'
            
            # Extract and OCR the region
            extraction_result = await extract_region_from_pdf(
                pdf_content=pdf_content,
                page_number=region['pageNumber'],
                bbox={
                    'x': region['x'],
                    'y': region['y'],
                    'width': region['width'],
                    'height': region['height']
                },
                region_id=region_id,
                region_scope=region_scope,
                gateway_context=_build_ai_gateway_context(
                    current_user=current_user,
                    db=db,
                    document_id=document_id,
                    region_id=region_id,
                    region_scope=region_scope,
                    is_b2c=is_b2c,
                ),
            )
            
            if extraction_result['success']:
                successful += 1
                region['ocrStatus'] = 'completed'
                region['extractedText'] = extraction_result.get('extractedText', '')
                layout_report = extraction_result.get("layoutReport", {})
                crop_metadata = extraction_result.get("cropMetadata", {})
                text_items = extraction_result.get("textItems", [])
                region['layoutStatus'] = 'completed'
                region['layoutRisks'] = layout_report.get("layout_risks", [])
                region['manualReviewRequired'] = False
                region['cropMetadata'] = crop_metadata
                
                # Parse extracted text to extract options if present
                extracted_text = extraction_result.get('extractedText', '')
                if is_answer_scope:
                    region['documentAnchorText'] = (
                        ocr_request.documentAnchorText.strip()
                        if ocr_request.documentAnchorText
                        else None
                    )
                    question_context = _resolve_question_context_for_answer_region(
                        answer_region=region,
                        answer_text=extracted_text,
                        answer_region_order=answer_region_order_by_id.get(str(region_id), -1),
                        question_regions=question_regions_for_answer,
                        questions_by_id=questions_by_id_for_answer,
                    )
                    try:
                        structured_answer = await extract_worked_answer_with_gpt(
                            ocr_result=extraction_result.get("ocrResult", {"pages": []}),
                            raw_answer_text=extracted_text,
                            question_context=question_context,
                            document_anchor_text=ocr_request.documentAnchorText,
                            gateway_context=_build_ai_gateway_context(
                                current_user=current_user,
                                db=db,
                                document_id=document_id,
                                region_id=region_id,
                                region_scope=region_scope,
                                is_b2c=is_b2c,
                            ),
                            layout_report=layout_report,
                        )
                        if structured_answer.get("answer_text"):
                            extracted_text = structured_answer["answer_text"]
                            region['extractedText'] = extracted_text
                        region['answerExtractionMetadata'] = {
                            "question_id": question_context.get("question_id"),
                            "question_label": question_context.get("question_label"),
                            "question_match_strategy": question_context.get("match_strategy"),
                            "answer_parser_provider": structured_answer.get("provider"),
                            "answer_parser_model": structured_answer.get("model"),
                            "answer_parser_confidence": structured_answer.get("confidence"),
                            "answer_parser_notes": structured_answer.get("notes"),
                        }
                        region['manualReviewRequired'] = bool(structured_answer.get("manual_review_required"))
                    except Exception as answer_parse_err:
                        logger.warning(
                            "Answer structuring failed for region %s; falling back to raw OCR text: %s",
                            region_id,
                            answer_parse_err,
                        )
                        region['manualReviewRequired'] = True
                        region['answerExtractionMetadata'] = {
                            "question_id": question_context.get("question_id"),
                            "question_label": question_context.get("question_label"),
                            "question_match_strategy": question_context.get("match_strategy"),
                            "answer_parser_error": str(answer_parse_err),
                        }
                    results.append({
                        "regionId": region_id,
                        "success": True,
                        "extractedText": extracted_text,
                        "extractedOptions": None,
                        "extractedImages": None,
                        "layoutReport": layout_report,
                        "manualReviewRequired": bool(region.get("manualReviewRequired")),
                        "answerExtractionMetadata": region.get("answerExtractionMetadata"),
                        "error": None
                    })
                    continue

                document_type = document.get("document_type", "Chapter Notes")
                skip_option_extraction = document_type == "Practice Sets"
                parsed_question = None

                try:
                    parsed_questions = await extract_questions_with_gpt(
                        extraction_result.get("ocrResult", {"pages": []}),
                        document.get("subject", "General"),
                        document.get("difficulty", "medium"),
                            skip_option_extraction=skip_option_extraction,
                            document_anchor_text=ocr_request.documentAnchorText,
                            gateway_context=_build_ai_gateway_context(
                                current_user=current_user,
                                db=db,
                                document_id=document_id,
                                region_id=region_id,
                                region_scope=region_scope,
                                is_b2c=is_b2c,
                            ),
                            layout_report=layout_report,
                        )
                    if parsed_questions:
                        parsed_question = parsed_questions[0]
                        extracted_text = parsed_question.text or extracted_text
                        if len(parsed_questions) > 1:
                            logger.info(
                                "Region %s produced %s parsed questions; using the first for this segment",
                                region_id,
                                len(parsed_questions),
                            )
                except Exception as parse_err:
                    logger.warning(
                        "Question extraction failed for region %s; falling back to raw OCR text: %s",
                        region_id,
                        parse_err,
                    )

                options = list(parsed_question.options) if parsed_question and not skip_option_extraction else []
                
                # Simple parsing for MCQ options (only if not Practice Sets)
                import re
                if not skip_option_extraction and not options:
                    option_pattern = re.compile(r'^([A-D])\)\s*(.+)$', re.MULTILINE)
                    for match in option_pattern.finditer(extracted_text):
                        options.append(match.group(2).strip())

                layout_corrections: List[Dict[str, Any]] = []
                validator = ExtractionValidator()
                expected_option_count = (
                    4
                    if not skip_option_extraction
                    and str(document.get("question_type", "mcq")).lower() == "mcq"
                    else None
                )
                validation_result = validator.validate_question(
                    question_text=parsed_question.text if parsed_question else extracted_text,
                    options=options,
                    layout_report=layout_report,
                    expected_option_count=expected_option_count,
                    has_figure=bool(extraction_result.get("extractedImages")),
                )

                if (
                    not skip_option_extraction
                    and "staggered_options" in layout_report.get("layout_risks", [])
                ):
                    normalization = OptionLayoutNormalizer().correct(
                        text_items=text_items,
                        layout_report=layout_report,
                    )
                    normalized_options = normalization.get("options_by_label", {})
                    if normalized_options and not normalization.get("manual_review_required"):
                        options = [normalized_options[label] for label in sorted(normalized_options)]
                        layout_corrections = normalization.get("corrections", [])
                        validation_result = validator.validate_question(
                            question_text=parsed_question.text if parsed_question else extracted_text,
                            options=options,
                            layout_report=layout_report,
                            expected_option_count=expected_option_count,
                            has_figure=bool(extraction_result.get("extractedImages")),
                        )
                    elif normalization.get("manual_review_required"):
                        validation_result["manual_review_required"] = True
                        validation_result.setdefault("reasons", []).append("ambiguous_option_layout")

                if (
                    not skip_option_extraction
                    and not validation_result.get("valid")
                    and extracted_text.strip()
                    and parsed_question is not None
                ):
                    retry_reason = ",".join(validation_result.get("reasons", []))
                    try:
                        retry_questions = await extract_questions_with_gpt(
                            extraction_result.get("ocrResult", {"pages": []}),
                            document.get("subject", "General"),
                            document.get("difficulty", "medium"),
                            skip_option_extraction=skip_option_extraction,
                            document_anchor_text=ocr_request.documentAnchorText,
                            gateway_context=_build_ai_gateway_context(
                                current_user=current_user,
                                db=db,
                                document_id=document_id,
                                region_id=region_id,
                                region_scope=region_scope,
                                is_b2c=is_b2c,
                            ),
                            layout_report=layout_report,
                            retry_reason=retry_reason,
                        )
                        if retry_questions:
                            retry_question = retry_questions[0]
                            retry_options = list(retry_question.options or [])
                            retry_validation = validator.validate_question(
                                question_text=retry_question.text,
                                options=retry_options,
                                layout_report=layout_report,
                                expected_option_count=expected_option_count,
                                has_figure=bool(extraction_result.get("extractedImages")),
                            )
                            if retry_validation.get("valid"):
                                parsed_question = retry_question
                                extracted_text = retry_question.text or extracted_text
                                options = retry_options
                                validation_result = retry_validation
                    except Exception as retry_err:
                        logger.warning(
                            "Layout-aware parser retry failed for region %s: %s",
                            region_id,
                            retry_err,
                        )

                if (
                    not skip_option_extraction
                    and not validation_result.get("valid")
                    and extraction_result.get("_regionPdfBytes")
                ):
                    fallback_reason = "validation_failed_after_layout_retry"
                    try:
                        vision_ocr_result = await call_gpt_vision_ocr_validation_fallback(
                            extraction_result["_regionPdfBytes"],
                            gateway_context=_build_ai_gateway_context(
                                current_user=current_user,
                                db=db,
                                document_id=document_id,
                                region_id=region_id,
                                region_scope=region_scope,
                                is_b2c=is_b2c,
                            ),
                            fallback_reason=fallback_reason,
                        )
                        vision_text = _ocr_pages_to_plain_text(vision_ocr_result)
                        if vision_text.strip():
                            vision_questions = await extract_questions_with_gpt(
                                vision_ocr_result,
                                document.get("subject", "General"),
                                document.get("difficulty", "medium"),
                                skip_option_extraction=skip_option_extraction,
                                document_anchor_text=ocr_request.documentAnchorText,
                                gateway_context=_build_ai_gateway_context(
                                    current_user=current_user,
                                    db=db,
                                    document_id=document_id,
                                    region_id=region_id,
                                    region_scope=region_scope,
                                    is_b2c=is_b2c,
                                ),
                                layout_report=layout_report,
                                retry_reason=fallback_reason,
                            )
                            if vision_questions:
                                vision_question = vision_questions[0]
                                vision_options = list(vision_question.options or [])
                                vision_validation = validator.validate_question(
                                    question_text=vision_question.text,
                                    options=vision_options,
                                    layout_report=layout_report,
                                    expected_option_count=expected_option_count,
                                    has_figure=bool(extraction_result.get("extractedImages")),
                                )
                                if vision_validation.get("valid"):
                                    extraction_result["ocrResult"] = vision_ocr_result
                                    parsed_question = vision_question
                                    extracted_text = vision_question.text or vision_text
                                    options = vision_options
                                    validation_result = vision_validation
                                else:
                                    validation_result.setdefault("reasons", []).append("vision_fallback_validation_failed")
                        else:
                            validation_result.setdefault("reasons", []).append("vision_fallback_empty")
                    except Exception as vision_err:
                        logger.warning(
                            "Validation-triggered GPT Vision fallback failed for region %s: %s",
                            region_id,
                            vision_err,
                        )
                        validation_result.setdefault("reasons", []).append("vision_fallback_failed")

                region['manualReviewRequired'] = bool(validation_result.get("manual_review_required"))
                region['validationStatus'] = "completed"
                region['validationReasons'] = validation_result.get("reasons", [])
                if layout_corrections:
                    region['layoutCorrections'] = layout_corrections
                
                # ============================================
                # SAVE THE EXTRACTED IMAGES (CROPPED FIGURES)
                # These are individually cropped diagrams from OCR
                # ============================================
                question_figures = []
                page_images = []
                
                # Get the extracted images from OCR result (these are cropped figures like direct OCR)
                extracted_images = extraction_result.get('extractedImages', [])
                region_image_base64 = extraction_result.get('regionImageBase64', '')
                
                if extracted_images:
                    # We have cropped figures from OCR - save each one
                    logger.info(f"📸 Region {region_id} has {len(extracted_images)} extracted images from OCR")
                    
                    for idx, img_data in enumerate(extracted_images):
                        img_base64 = img_data.get('base64', '')
                        img_id = img_data.get('id', f'img-{idx}')
                        
                        if img_base64:
                            try:
                                # Save each cropped figure using the same function as regular OCR
                                saved_images = await save_image_to_disk(
                                    image_base64=img_base64,
                                    image_id=f"region-{region_id}-{img_id}",
                                    pdf_filename=document.get("filename", "unknown.pdf"),
                                    db=db,
                                    user_id=current_user.get("user_id"),
                                    split_composite=False,
                                    is_b2c=is_b2c
                                )
                                
                                if saved_images:
                                    logger.info(f"✅ Saved cropped figure {img_id} for region {region_id}")
                                    total_images_saved += len(saved_images)
                                    
                                    # Add to question_figures (these are the actual diagrams)
                                    for saved_img in saved_images:
                                        image_obj = {
                                            'id': saved_img['id'],
                                            'filename': saved_img['filename'],
                                            'path': saved_img['path'],
                                            'base64Data': img_base64,
                                            'description': '',
                                            'type': 'diagram',
                                            'bbox': {
                                                'top_left_x': img_data.get('top_left_x', 0),
                                                'top_left_y': img_data.get('top_left_y', 0),
                                                'bottom_right_x': img_data.get('bottom_right_x', 0),
                                                'bottom_right_y': img_data.get('bottom_right_y', 0)
                                            },
                                            'metadata': {
                                                'source': 'manual_segmentation_ocr',
                                                'page': region['pageNumber'],
                                                'extractedAt': datetime.utcnow().isoformat()
                                            }
                                        }
                                        question_figures.append(image_obj)
                            except Exception as img_err:
                                logger.error(f"Failed to save cropped figure {img_id} for region {region_id}: {img_err}")
                
                else:
                    figure_hint_pattern = r'\b(figure|fig\.?|diagram|graph|plot|chart|image|shown below|given below|following diagram|following figure)\b'
                    should_save_region_snapshot = bool(region_image_base64) and (
                        len(extracted_text.strip()) < 50 or
                        bool(re.search(figure_hint_pattern, extracted_text, re.IGNORECASE))
                    )

                    if not should_save_region_snapshot:
                        logger.info(f"Region {region_id}: text extracted successfully, no figures needed")
                    else:
                        # Fallback for vector/embedded diagrams that the OCR provider
                        # and PyMuPDF image extraction do not expose as separate images.
                        logger.info(f"Region {region_id}: saving region screenshot as figure fallback")

                        try:
                            saved_images = await save_image_to_disk(
                                image_base64=region_image_base64,
                                image_id=f"region-{region_id}-full",
                                pdf_filename=document.get("filename", "unknown.pdf"),
                                db=db,
                                user_id=current_user.get("user_id"),
                                split_composite=False,
                                is_b2c=is_b2c
                            )

                            if saved_images:
                                total_images_saved += len(saved_images)

                                for saved_img in saved_images:
                                    image_obj = {
                                        'id': saved_img['id'],
                                        'filename': saved_img['filename'],
                                        'path': saved_img['path'],
                                        'base64Data': region_image_base64,
                                        'description': '',
                                        'type': 'region_screenshot',
                                        'bbox': {
                                            'x': region['x'],
                                            'y': region['y'],
                                            'width': region['width'],
                                            'height': region['height']
                                        },
                                        'metadata': {
                                            'source': 'manual_segmentation_fallback',
                                            'page': region['pageNumber'],
                                            'extractedAt': datetime.utcnow().isoformat()
                                        }
                                    }
                                    question_figures.append(image_obj)
                        except Exception as img_err:
                            logger.error(f"Failed to save region screenshot for {region_id}: {img_err}")
                
                results.append({
                    "regionId": region_id,
                    "success": True,
                    "extractedText": extracted_text,
                    "extractedOptions": options if options else None,
                    "extractedImages": [{"id": img['id'], "path": img['path']} for img in question_figures],
                    "layoutReport": layout_report,
                    "validation": validation_result,
                    "manualReviewRequired": bool(validation_result.get("manual_review_required")),
                    "error": None
                })
                
                # Create question in database
                question_text = parsed_question.text if parsed_question else extracted_text
                ocr_result_metadata = extraction_result.get("ocrResult", {})
                
                # Only parse question structure if not Practice Sets
                if not parsed_question and not skip_option_extraction:
                    question_match = re.search(r'QUESTION:\s*(.+?)(?:OPTIONS:|FIGURES:|$)', extracted_text, re.DOTALL)
                    if question_match:
                        question_text = question_match.group(1).strip()
                
                question_doc = {
                    "id": region_id,
                    "text": question_text,  # For Practice Sets, this includes the full text with options
                    "subject": document.get("subject", "General"),
                    "difficulty": document.get("difficulty", "medium"),
                    "question_type": document.get("question_type", "mcq"),
                    "document_type": document.get("document_type", "Practice Sets"),
                    "extracted_at": datetime.utcnow(),
                    "pdf_source": document.get("filename", ""),
                    "document_id": document_id,
                    "images": page_images,  # Option images (empty for manual segmentation)
                    "question_figures": question_figures,  # The region image as the question figure
                    "options": options,  # Will be empty for Practice Sets
                    "enhanced_options": [
                        {
                            "id": f"{region_id}_opt_{i}",
                            "type": "text",
                            "content": opt,
                            "label": chr(65 + i)
                        }
                        for i, opt in enumerate(options)
                    ] if options else [],  # Empty for Practice Sets
                    "correct_answer": parsed_question.correct_answer if parsed_question else None,
                    "is_region_based": True,
                    "options_inline": skip_option_extraction,  # Flag indicating options are in text
                    "metadata": parsed_question.metadata if parsed_question else {},
                    "extraction_metadata": {
                        "ocr_provider": ocr_result_metadata.get("_ocr_provider", "unknown"),
                        "ocr_model": ocr_result_metadata.get("_ocr_model", MISTRAL_OCR_MODEL),
                        "ocr_fallback_reason": ocr_result_metadata.get("_fallback_reason"),
                        "parser_provider": "groq" if GROQ_API_KEY else "openai",
                        "parser_model": GROQ_MODEL if GROQ_API_KEY else OCR_FALLBACK_MODEL,
                        "layout_risks": layout_report.get("layout_risks", []),
                        "layout_corrections": [
                            correction.get("correction")
                            for correction in layout_corrections
                            if correction.get("correction")
                        ],
                        "layout_report": layout_report,
                        "validation": validation_result,
                        "extraction_confidence": layout_report.get("option_layout", {}).get("confidence"),
                        "manual_review_required": bool(validation_result.get("manual_review_required")),
                    },
                    "region_metadata": {
                        "page": region['pageNumber'],
                        "x": region['x'],
                        "y": region['y'],
                        "width": region['width'],
                        "height": region['height'],
                        "crop": crop_metadata,
                        "layout_risks": layout_report.get("layout_risks", []),
                        "manual_review_required": bool(validation_result.get("manual_review_required")),
                    },
                    "points": parsed_question.points if parsed_question and parsed_question.points else 4.0,
                    "penalty": parsed_question.penalty if parsed_question and parsed_question.penalty else 1.0,
                    "created_by": current_user.get("user_id"),
                    "created_at": datetime.utcnow()
                }
                
                if is_b2c:
                    await db.b2c_insert_one("questions", question_doc)
                else:
                    await db.mongo_insert_one("questions", question_doc)
                
            else:
                failed += 1
                region['ocrStatus'] = 'error'
                results.append({
                    "regionId": region_id,
                    "success": False,
                    "extractedText": None,
                    "extractedOptions": None,
                    "extractedImages": None,
                    "error": extraction_result.get('error', 'Unknown error')
                })

        answer_question_mappings: List[Dict[str, Any]] = []
        if is_answer_scope:
            question_regions_filter = _document_regions_filter(document_id, "question")
            if is_b2c:
                question_regions_doc = await db.b2c_find_one("document_regions", question_regions_filter)
            else:
                question_regions_doc = await db.mongo_find_one("document_regions", question_regions_filter)
            answer_question_mappings = await AnswerQuestionMappingService().map_region_order(
                db=db,
                is_b2c=is_b2c,
                document_id=document_id,
                question_regions=(question_regions_doc or {}).get("regions", []),
                answer_regions=all_regions,
            )
        
        # Update regions with OCR status
        regions_update_filter = {"_id": regions_doc["_id"]} if regions_doc and regions_doc.get("_id") else regions_filter
        if is_b2c:
            await db.b2c_update_one(
                "document_regions",
                regions_update_filter,
                {"$set": {"regions": all_regions, "updated_at": datetime.utcnow().isoformat()}}
            )
        else:
            await db.mongo_update_one(
                "document_regions",
                regions_update_filter,
                {"$set": {"regions": all_regions, "updated_at": datetime.utcnow().isoformat()}}
            )
        
        # Update document status
        final_status = "completed" if failed == 0 else ("error" if successful == 0 else "completed")
        if is_answer_scope:
            document_status_update = {
                "answer_sheet_ocr_status": final_status,
                "answer_sheet_ocr_completed_at": datetime.utcnow(),
                "answer_sheet_processed_regions_count": successful,
                "answer_sheet_mapped_answers_count": len(
                    [
                        mapping
                        for mapping in answer_question_mappings
                        if not mapping.get("manual_review_required")
                    ]
                ),
                "answer_sheet_document_anchor_text": (
                    ocr_request.documentAnchorText.strip()
                    if ocr_request.documentAnchorText
                    else None
                ),
            }
        else:
            document_status_update = {
                "ocr_status": final_status,
                "extracted_questions_count": successful,
                "extracted_images_count": total_images_saved,
                "ocr_completed_at": datetime.utcnow()
            }
        
        if is_b2c:
            await db.b2c_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": document_status_update}
            )
        else:
            await db.mongo_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": document_status_update}
            )
        
        logger.info(
            f"{region_scope.title()} region OCR completed for {document_id}: "
            f"{successful} regions, {total_images_saved} images, {failed} failed"
        )
        
        result = {
            "success": True,
            "documentId": document_id,
            "regionScope": region_scope,
            "processedRegions": len(regions_to_process),
            "successfulRegions": successful,
            "failedRegions": failed,
            "mappedAnswers": len([
                mapping
                for mapping in answer_question_mappings
                if not mapping.get("manual_review_required")
            ]) if is_answer_scope else None,
            "results": results
        }
        observe_ocr_job(
            job_type="region",
            status="success",
            duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
        )
        return result
        
    except HTTPException as exc:
        observe_ocr_job(
            job_type="region",
            status=f"error_{exc.status_code}",
            duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
        )
        raise
    except Exception as e:
        observe_ocr_job(
            job_type="region",
            status="error_500",
            duration_seconds=(datetime.utcnow() - ocr_started_at).total_seconds(),
        )
        logger.error(f"Region OCR processing error: {str(e)}", exc_info=True)
        
        # Reset status on error
        try:
            is_b2c = is_b2c_admin(current_user)
            error_region_scope = _normalise_region_scope(getattr(ocr_request, "regionScope", "question"))
            error_status_field = "answer_sheet_ocr_status" if error_region_scope == "answer" else "ocr_status"
            if is_b2c:
                await db.b2c_update_one(
                    "documents",
                    {"document_id": document_id},
                    {"$set": {error_status_field: "error"}}
                )
            else:
                await db.mongo_update_one(
                    "documents",
                    {"document_id": document_id},
                    {"$set": {error_status_field: "error"}}
                )
        except:
            pass
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process OCR: {str(e)}"
        )
