"""
Async PDF Processing API for SkillBot
PDF upload and OCR processing endpoints with Mistral AI OCR (primary) and GPT Vision (fallback)
"""

import logging
import base64
import asyncio
import uuid
import os
from typing import Optional, Dict, Any, List
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


async def call_sarvam_ocr(file_content: bytes) -> Dict[str, Any]:
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
    # --- Mistral OCR primary, GPT Vision fallback ---
    if MISTRAL_API_KEY:
        try:
            result = await call_mistral_ocr(file_content)
            print("[OCR] Provider: Mistral AI", flush=True)
            return result
        except Exception as mistral_err:
            print(f"[OCR] Mistral OCR failed ({type(mistral_err).__name__}: {mistral_err}), falling back to GPT Vision...", flush=True)
            logger.warning(f"Mistral OCR failed: {mistral_err}")
    else:
        print("[OCR] MISTRAL_API_KEY not set, skipping Mistral...", flush=True)

    # GPT Vision fallback
    try:
        result = await call_gpt_vision_ocr(file_content)
        print("[OCR] Provider: GPT Vision (fallback)", flush=True)
        return result
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
    skip_option_extraction: bool = False
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

    extraction_prompt = (
        "You are a question paper parser. Extract ONLY the questions from the text below.\n\n"
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
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=extract_model,
                messages=[{"role": "user", "content": extraction_prompt}],
                max_completion_tokens=16384,
            )
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
                retry_response = await client.chat.completions.create(
                    model=extract_model,
                    messages=[{"role": "user", "content": retry_prompt}],
                    max_completion_tokens=4096,
                )
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

        # Use real image IDs from the page if this question references a figure
        # Assign images 1:1 in order: 1st figure-question gets 1st image, etc.
        if has_image and q_page in page_image_ids:
            cursor = page_image_cursor.get(q_page, 0)
            all_page_imgs = page_image_ids[q_page]
            if cursor < len(all_page_imgs):
                img_refs = [all_page_imgs[cursor]]
                page_image_cursor[q_page] = cursor + 1
            else:
                img_refs = []
        else:
            img_refs = []

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
        ocr_result = await call_sarvam_ocr(file_content)

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
            skip_option_extraction=skip_option_extraction
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
                    for page in ocr_result.get("pages", []):
                        if page.get("index") == page_index:
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
                                                'page': page_index,
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
                                                    'page': page_index,
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

class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]
    total: int
    page: int
    limit: int

@router.post("/upload")
@limiter.limit("10/minute")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
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
            "question_type": question_type if question_type in ["mcq", "subjective"] else "mcq",  # Default question type for extracted questions
            "instructions": instructions.strip() if instructions else None,  # Paper instructions
            "is_active": False,  # Default to inactive until admin enables
            "is_s3": is_s3_enabled()  # Track storage location
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
            "message": "Document uploaded successfully. Use 'Segment' to define question regions before processing OCR.",
            "document_id": document_id,
            "file_path": relative_path,
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
    current_user: Dict[str, Any] = Depends(get_current_user)
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
            return await call_sarvam_ocr(file_content)

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

        # Format response and check file existence
        from pathlib import Path

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
            # Check if physical file exists on disk
            file_path = Path(doc["file_path"])
            file_exists = file_path.exists()

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
                file_path=doc["file_path"],
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
                is_active=doc.get("is_active", True)
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

        # Format response and check file existence
        from pathlib import Path
        document_list = []
        for doc in documents:
            # Check if physical file exists on disk
            file_path = Path(doc["file_path"])
            file_exists = file_path.exists()

            document_list.append(DocumentMetadata(
                document_id=doc["document_id"],
                title=doc["title"],
                document_type=doc["document_type"],
                subject=doc["subject"],
                difficulty=doc["difficulty"],
                course_plan=doc.get("course_plan"),
                standard=doc.get("standard"),
                file_path=doc["file_path"],
                filename=doc["filename"],
                uploaded_by=doc["uploaded_by"],
                uploaded_at=doc["uploaded_at"],
                ocr_status=doc["ocr_status"],
                ocr_job_id=doc.get("ocr_job_id"),
                extracted_questions_count=doc.get("extracted_questions_count", 0),
                extracted_images_count=doc.get("extracted_images_count", 0),
                file_exists=file_exists
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

        # Local file handling
        from pathlib import Path
        backend_dir = Path(os.getcwd())
        # Convert stored path to use forward slashes, then to Path
        stored_path = stored_path.replace("\\", "/")
        file_path = backend_dir / stored_path
        logger.info(f"Full file path: {file_path}")

        if not file_path.exists():
            logger.error(f"File does not exist at path: {file_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"PDF file not found on server at: {file_path}"
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

            serialized_questions.append(question_dict)

        return {
            "document_id": document_id,
            "document_title": document["title"],
            "questions_count": len(serialized_questions),
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

        # Generate unique question ID
        full_question_id = f"QST{question_id}"

        # Parse options metadata
        options_metadata = json.loads(options_data) if options_data else []

        # Prepare question document
        question_doc = {
            "id": full_question_id,
            "text": question_text,  # Standard field name used by MCQ service
            "question_text": question_text,  # Alias for compatibility
            "question_type": question_type,  # Store question type (mcq or integer)
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

class DocumentRegionsResponse(BaseModel):
    """Response for document regions"""
    documentId: str
    regions: List[QuestionRegion]
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None
    createdBy: Optional[str] = None

class RegionOCRRequest(BaseModel):
    """Request for processing specific regions with OCR"""
    regionIds: Optional[List[str]] = None  # If None, process all regions

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
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get all saved question regions for a document.
    
    Returns the list of bounding box regions that have been manually drawn
    on the PDF pages for question segmentation.
    """
    try:
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
        
        # Get regions for this document
        if is_b2c:
            regions_doc = await db.b2c_find_one("document_regions", {"document_id": document_id})
        else:
            regions_doc = await db.mongo_find_one("document_regions", {"document_id": document_id})
        
        if not regions_doc:
            return {
                "documentId": document_id,
                "regions": [],
                "createdAt": None,
                "updatedAt": None,
                "createdBy": None
            }
        
        return {
            "documentId": document_id,
            "regions": regions_doc.get("regions", []),
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


@router.post("/documents/{document_id}/regions")
async def save_document_regions(
    document_id: str,
    request: DocumentRegionsRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Save question regions for a document.
    
    Stores the bounding box regions that have been manually drawn on the PDF pages.
    This replaces any existing regions for the document.
    """
    try:
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
        
        now = datetime.utcnow().isoformat()
        
        # Convert regions to dict format
        regions_data = [region.dict() for region in request.regions]
        
        # Prepare regions document
        regions_doc = {
            "document_id": document_id,
            "regions": regions_data,
            "created_by": current_user.get("user_id"),
            "updated_at": now
        }
        
        # Upsert regions document
        if is_b2c:
            existing = await db.b2c_find_one("document_regions", {"document_id": document_id})
            if existing:
                await db.b2c_update_one(
                    "document_regions",
                    {"document_id": document_id},
                    {"$set": regions_doc}
                )
            else:
                regions_doc["created_at"] = now
                await db.b2c_insert_one("document_regions", regions_doc)
        else:
            existing = await db.mongo_find_one("document_regions", {"document_id": document_id})
            if existing:
                await db.mongo_update_one(
                    "document_regions",
                    {"document_id": document_id},
                    {"$set": regions_doc}
                )
            else:
                regions_doc["created_at"] = now
                await db.mongo_insert_one("document_regions", regions_doc)
        
        logger.info(f"Saved {len(request.regions)} regions for document {document_id}")
        
        return {
            "success": True,
            "message": f"Saved {len(request.regions)} regions",
            "documentId": document_id,
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
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database)
):
    """
    Delete all regions for a document.
    """
    try:
        is_b2c = is_b2c_admin(current_user)
        
        if is_b2c:
            result = await db.b2c_delete_one("document_regions", {"document_id": document_id})
        else:
            result = await db.mongo_delete_one("document_regions", {"document_id": document_id})
        
        return {
            "success": True,
            "message": f"Deleted regions for document {document_id}"
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
    region_id: str
) -> Dict[str, Any]:
    """
    Extract a specific region from a PDF page and process it with OCR.

    Uses Sarvam AI Document Intelligence to extract text AND images from the region,
    just like the direct OCR pipeline does for full documents.

    Args:
        pdf_content: Raw PDF bytes
        page_number: Page number (1-indexed)
        bbox: Bounding box as percentages {x, y, width, height}
        region_id: ID of the region being processed

    Returns:
        Dict with extracted text, images list, and region screenshot
    """
    try:
        from PIL import Image
        import fitz  # PyMuPDF
        import io
        
        # Open PDF
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        if page_number < 1 or page_number > len(doc):
            return {
                "success": False,
                "error": f"Invalid page number {page_number}"
            }
        
        page = doc[page_number - 1]  # 0-indexed
        page_rect = page.rect
        
        # Convert percentage coordinates to actual coordinates
        x0 = page_rect.width * (bbox['x'] / 100)
        y0 = page_rect.height * (bbox['y'] / 100)
        x1 = x0 + page_rect.width * (bbox['width'] / 100)
        y1 = y0 + page_rect.height * (bbox['height'] / 100)
        
        clip_rect = fitz.Rect(x0, y0, x1, y1)
        
        # Create a new single-page PDF with just the region
        # This allows us to use Sarvam OCR which expects a PDF file
        region_doc = fitz.open()  # Create new empty PDF
        
        # Create a new page with the region dimensions
        region_width = x1 - x0
        region_height = y1 - y0
        new_page = region_doc.new_page(width=region_width, height=region_height)
        
        # Copy the content from the original region to the new page
        # Use show_pdf_page to copy a portion of the original page
        new_page.show_pdf_page(
            fitz.Rect(0, 0, region_width, region_height),  # Target rect on new page
            doc,  # Source document
            page_number - 1,  # Source page (0-indexed)
            clip=clip_rect  # Clip to the region
        )
        
        # Get the region PDF as bytes
        region_pdf_bytes = region_doc.tobytes()
        region_doc.close()
        
        # Also render the region as an image for fallback/reference
        mat = fitz.Matrix(3.0, 3.0)  # 3x zoom for better quality
        pix = page.get_pixmap(matrix=mat, clip=clip_rect)
        region_img_bytes = pix.tobytes("png")
        region_img_base64 = base64.b64encode(region_img_bytes).decode('utf-8')
        
        doc.close()

        # OCR the cropped region PDF
        logger.info(f"Calling OCR for region {region_id} (PDF size: {len(region_pdf_bytes)} bytes)")

        try:
            ocr_result = await call_sarvam_ocr(region_pdf_bytes)
        except Exception as ocr_err:
            logger.error(f"Sarvam OCR failed for region {region_id}: {ocr_err}")
            return {
                "success": False,
                "error": f"OCR failed for region: {ocr_err}",
                "regionImageBase64": region_img_base64
            }

        # Extract text and images from OCR result
        extracted_text = ""
        extracted_images = []

        for page_data in ocr_result.get("pages", []):
            page_markdown = page_data.get("markdown", "")
            extracted_text += page_markdown + "\n"

            for img in page_data.get("images", []):
                if img.get("image_base64"):
                    extracted_images.append({
                        "id": img.get("id", f"img-{len(extracted_images)}"),
                        "base64": img.get("image_base64"),
                        "top_left_x": img.get("top_left_x", 0),
                        "top_left_y": img.get("top_left_y", 0),
                        "bottom_right_x": img.get("bottom_right_x", 0),
                        "bottom_right_y": img.get("bottom_right_y", 0)
                    })

        extracted_text = extracted_text.strip()

        # Filter images: only keep those referenced in the markdown
        import re
        referenced_ids = set(re.findall(r'!\[[^\]]*\]\(([^)]+)\)', extracted_text))
        real_figures = [img for img in extracted_images if img['id'] in referenced_ids]

        logger.info(
            f"Region {region_id}: {len(extracted_text)} chars, "
            f"{len(extracted_images)} raw images, {len(real_figures)} actual figures"
        )

        return {
            "success": True,
            "extractedText": extracted_text,
            "extractedImages": real_figures,
            "regionImageBase64": region_img_base64,
            "ocrResult": ocr_result
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
        
        # Get regions
        if is_b2c:
            regions_doc = await db.b2c_find_one("document_regions", {"document_id": document_id})
        else:
            regions_doc = await db.mongo_find_one("document_regions", {"document_id": document_id})
        
        if not regions_doc or not regions_doc.get("regions"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No regions defined for this document. Please draw question regions first."
            )
        
        all_regions = regions_doc.get("regions", [])
        
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
        
        # Load PDF content
        from pathlib import Path as _Path
        backend_dir = _Path(os.getcwd())
        file_path = document.get("file_path", "")
        
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
            local_path = backend_dir / file_path if not _Path(file_path).is_absolute() else _Path(file_path)
            if local_path.exists():
                async with aiofiles.open(str(local_path), "rb") as f:
                    pdf_content = await f.read()
        
        if not pdf_content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="PDF file not found"
            )
        
        # Update document status to processing
        if is_b2c:
            await db.b2c_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": {"ocr_status": "processing"}}
            )
        else:
            await db.mongo_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": {"ocr_status": "processing"}}
            )
        
        # Process each region
        results = []
        successful = 0
        failed = 0
        total_images_saved = 0  # Track total images saved
        
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
                region_id=region_id
            )
            
            if extraction_result['success']:
                successful += 1
                region['ocrStatus'] = 'completed'
                region['extractedText'] = extraction_result.get('extractedText', '')
                
                # Parse extracted text to extract options if present
                extracted_text = extraction_result.get('extractedText', '')
                options = []
                
                # Check if this is Practice Sets - skip option extraction
                document_type = document.get("document_type", "Chapter Notes")
                skip_option_extraction = document_type == "Practice Sets"
                
                # Simple parsing for MCQ options (only if not Practice Sets)
                import re
                if not skip_option_extraction:
                    option_pattern = re.compile(r'^([A-D])\)\s*(.+)$', re.MULTILINE)
                    for match in option_pattern.finditer(extracted_text):
                        options.append(match.group(2).strip())
                
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
                                    split_composite=True,  # Split if it's a composite image with multiple options
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
                
                elif region_image_base64 and len(extracted_text.strip()) < 50:
                    # Fallback: Save region screenshot ONLY when text extraction failed.
                    # Short text (< 50 chars) means the content is likely a pure diagram/figure
                    # that OCR couldn't read as text. Don't save for text-heavy questions —
                    # that would just duplicate the already-extracted text as an image.
                    logger.info(f"Region {region_id}: minimal text extracted, saving screenshot as figure fallback")

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
                else:
                    logger.info(f"Region {region_id}: text extracted successfully, no figures needed")
                
                results.append({
                    "regionId": region_id,
                    "success": True,
                    "extractedText": extracted_text,
                    "extractedOptions": options if options else None,
                    "extractedImages": [{"id": img['id'], "path": img['path']} for img in question_figures],
                    "error": None
                })
                
                # Create question in database
                question_text = extracted_text
                
                # Only parse question structure if not Practice Sets
                if not skip_option_extraction:
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
                    "correct_answer": None,
                    "is_region_based": True,
                    "options_inline": skip_option_extraction,  # Flag indicating options are in text
                    "region_metadata": {
                        "page": region['pageNumber'],
                        "x": region['x'],
                        "y": region['y'],
                        "width": region['width'],
                        "height": region['height']
                    },
                    "points": 4.0,  # Default 4 marks per question
                    "penalty": 1.0,  # Default 1 mark penalty for wrong answer
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
        
        # Update regions with OCR status
        if is_b2c:
            await db.b2c_update_one(
                "document_regions",
                {"document_id": document_id},
                {"$set": {"regions": all_regions, "updated_at": datetime.utcnow().isoformat()}}
            )
        else:
            await db.mongo_update_one(
                "document_regions",
                {"document_id": document_id},
                {"$set": {"regions": all_regions, "updated_at": datetime.utcnow().isoformat()}}
            )
        
        # Update document status
        final_status = "completed" if failed == 0 else ("error" if successful == 0 else "completed")
        
        if is_b2c:
            await db.b2c_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": {
                    "ocr_status": final_status,
                    "extracted_questions_count": successful,
                    "extracted_images_count": total_images_saved,
                    "ocr_completed_at": datetime.utcnow()
                }}
            )
        else:
            await db.mongo_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": {
                    "ocr_status": final_status,
                    "extracted_questions_count": successful,
                    "extracted_images_count": total_images_saved,
                    "ocr_completed_at": datetime.utcnow()
                }}
            )
        
        logger.info(f"Region OCR completed for {document_id}: {successful} questions, {total_images_saved} images, {failed} failed")
        
        result = {
            "success": True,
            "documentId": document_id,
            "processedRegions": len(regions_to_process),
            "successfulRegions": successful,
            "failedRegions": failed,
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
        except:
            pass
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process OCR: {str(e)}"
        )
