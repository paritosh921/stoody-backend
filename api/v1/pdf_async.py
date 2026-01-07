"""
Async PDF Processing API for SkillBot
PDF upload and OCR processing endpoints with Mistral AI integration
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

from fastapi import APIRouter, Request, HTTPException, Depends, status, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
import aiofiles
from bson import ObjectId as BsonObjectId

from core.database import DatabaseManager
from core.cache import CacheManager
from api.v1.auth_async import get_current_user, get_database, get_cache
from api.v1.student_async import require_student, require_student_or_admin
from config_async import OCR_TIMEOUT_SECONDS
from utils.path_utils import get_relative_path, get_absolute_path
from utils.s3_storage import upload_file as s3_upload_file, is_s3_enabled, get_public_url, download_file

logger = logging.getLogger(__name__)

router = APIRouter()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Mistral OCR API configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"

# IMPORTANT: Grade/Standard matching uses EXACT string matching
# Both student.grade and document.standard should come from the same admin settings
# (configured in School Settings), so they will match exactly
# For example: student.grade="12th Pass" matches document.standard="12th Pass"
# No normalization or fuzzy matching is needed or desired

# Pydantic models
class MistralOCRImage(BaseModel):
    id: str
    top_left_x: int
    top_left_y: int
    bottom_right_x: int
    bottom_right_y: int
    image_base64: Optional[str] = None

class MistralOCRPage(BaseModel):
    index: int
    markdown: str
    images: List[MistralOCRImage]
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
    pages: Optional[List[MistralOCRPage]] = None

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

async def call_mistral_ocr(pdf_base64: str) -> Dict[str, Any]:
    """Call Mistral OCR API with base64 PDF data"""
    import aiohttp

    # Validate API key
    if not MISTRAL_API_KEY:
        logger.error("MISTRAL_API_KEY is not configured in environment variables")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Mistral API key is not configured. Please set MISTRAL_API_KEY in environment variables."
        )

    try:
        document_url = f"data:application/pdf;base64,{pdf_base64}"

        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": document_url
            },
            "include_image_base64": True
        }

        # Log request without base64 data
        logger.info(f"Calling Mistral OCR API (PDF size: {len(pdf_base64)} chars)")

        # Create session with trace disabled to prevent logging base64
        async with aiohttp.ClientSession(trace_configs=[]) as session:
            async with session.post(
                MISTRAL_OCR_URL,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=OCR_TIMEOUT_SECONDS)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Mistral OCR API error: {response.status} - {error_text}")
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Mistral OCR API error: {error_text}"
                    )

                return await response.json()

    except HTTPException:
        # Re-raise HTTPException without wrapping it
        raise
    except asyncio.TimeoutError:
        logger.error("Mistral OCR API timeout")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="OCR processing timeout"
        )
    except aiohttp.ClientError as e:
        logger.error(f"Mistral OCR API client error: {type(e).__name__} - {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR API connection error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Mistral OCR API unexpected error: {type(e).__name__} - {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}"
        )

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

def extract_questions_from_ocr(
    ocr_result: Dict[str, Any],
    subject: str,
    difficulty: str,
    skip_option_extraction: bool = False  # When True, keep options in question text (for Practice Sets)
) -> List[ExtractedQuestion]:
    """Extract questions from Mistral OCR result.
    
    Args:
        ocr_result: The raw OCR result from Mistral
        subject: Subject of the document
        difficulty: Difficulty level
        skip_option_extraction: If True, don't extract options separately - keep them in question text.
                               This is used for Practice Sets where we want the full question with options
                               to be displayed inline.
    """
    questions = []
    
    # ------------------------------
    # Helper: generic LaTeX + text normalisation
    # ------------------------------
    import re

    # Match inline/display LaTeX blocks
    _MATH_BLOCK_RE = re.compile(r"(\$\$[\s\S]*?\$\$|\$[^$\n]+?\$|\\\[[\s\S]*?\\\]|\\\([^)]*?\\\))")

    # Characters typically present in a plain rendered math echo (no natural language)
    _MATHY_TEXT_RE = re.compile(r"^[\s0-9A-Za-z_\^\-\+=*/()\[\]{}|.,:×·%°<>²³⁴⁵⁶⁷⁸⁹⁰⁻⁺±∞≤≥≠≈√⁄~]*$")

    def _dedupe_latex_echo(text: str) -> str:
        """If a string contains LaTeX and an immediate plain-text echo of the
        same formula, prefer keeping the LaTeX only.

        The heuristic is conservative:
        - Detect one or more LaTeX blocks
        - Compute the residual (non-LaTeX) text
        - If residual appears to be purely math-like (no words) keep only the LaTeX blocks
        - Otherwise, return the original text
        """
        if not text:
            return text

        parts = _MATH_BLOCK_RE.split(text)
        if len(parts) == 1:
            # No LaTeX blocks
            return text

        latex_blocks = [p for p in parts if _MATH_BLOCK_RE.match(p)]
        non_latex = "".join(p for p in parts if not _MATH_BLOCK_RE.match(p)).strip()

        # If we have LaTeX and the rest is only mathy symbols/digits (likely OCR echo), drop it
        if latex_blocks and (not non_latex or _MATHY_TEXT_RE.match(non_latex)):
            # Join multiple LaTeX blocks with a space to preserve intent
            return " ".join(latex_blocks).strip()

        # Additional check: if the text starts with LaTeX and contains rendered math symbols, 
        # extract only the LaTeX part
        if latex_blocks and any(char in non_latex for char in '⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺±×÷∞≤≥≠≈√⁄'):
            return " ".join(latex_blocks).strip()

        return text

    def _clean_option_text(text: str) -> str:
        # Normalise whitespace
        t = re.sub(r"\s+", " ", (text or "")).strip()
        # Remove duplicated plain echo when LaTeX is present
        t = _dedupe_latex_echo(t)
        return t
    # Parse pages from OCR result
    pages = ocr_result.get("pages", [])

    for page in pages:
        markdown_content = page.get("markdown", "")

        # Log a sample of the markdown to understand structure
        logger.info(f"Page {page.get('index', 0)} markdown sample (first 1500 chars):\n{markdown_content[:1500]}")

        # Count potential questions in markdown for debugging
        potential_questions = len(re.findall(r'(?:^|\n)(?:#{1,3}\s+)?Q\.?\s*\d+|(?:^|\n)\d+[\.\)]\s+', markdown_content, re.MULTILINE))
        logger.info(f"Potential questions detected in markdown: {potential_questions}")

        # Simple question extraction logic
        # Look for patterns like:
        # - Lines ending with "?"
        # - Numbered items (1., 2., etc.)
        # - MCQ patterns (A), B), C), D))

        lines = markdown_content.split("\n")
        current_question = None
        current_question_text = ""
        current_options = []
        current_image_refs = []
        current_question_images = []  # Images that are part of the question itself (diagrams)
        current_option_images = []  # Images that are MCQ options
        # Track multi-line text option accumulation (A/B/C/D spilling to next lines)
        accumulating_option_idx: Optional[int] = None
        previous_line = ""  # Track previous line to detect option labels before images

        # Compile regex patterns for option label detection
        option_label_pattern = re.compile(r'^\s*\(([A-Da-d]|[ivxIVX]+)\)\s*$')  # (A), (B), (i), (ii), etc.

        # Buffer for images detected before the corresponding question text starts
        pending_images: List[str] = []

        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                previous_line = line_stripped
                continue

            # Log first few lines for debugging
            if line_num < 10:
                logger.debug(f"Line {line_num}: {line_stripped[:100]}")

            # Extract image references from the line (format: ![alt](url) or ![alt])
            # Updated regex to capture both alt text and URL to handle cases like ![](img-0.jpeg)
            image_refs_raw = re.findall(r'!\[([^\]]*)\](?:\(([^)]+)\))?', line)
            # Build image_refs list: prefer alt text if not empty, otherwise use URL
            image_refs = []
            for alt_text, url_text in image_refs_raw:
                img_ref = alt_text.strip() if alt_text.strip() else url_text.strip()
                if img_ref:
                    image_refs.append(img_ref)
            
            # Log image extraction for debugging
            if image_refs:
                logger.info(f"📸 Line {line_num}: Extracted {len(image_refs)} image refs: {image_refs}")

            # NEW LOGIC: Classify images based on previous line
            if image_refs and not current_question:
                # Buffer images appearing before question starts
                for img_ref in image_refs:
                    if img_ref not in pending_images:
                        pending_images.append(img_ref)
                logger.info(f"Buffered {len(image_refs)} images appearing before question start")
            elif image_refs and current_question:
                # Check if previous line was an option label
                is_option_image = option_label_pattern.match(previous_line)

                for img_ref in image_refs:
                    # Avoid duplicates
                    if is_option_image:
                        # This image is an MCQ option (has label like (A), (B) before it)
                        if img_ref not in current_option_images:
                            current_option_images.append(img_ref)
                            logger.info(f"✓ Detected option image: {img_ref} (preceded by label: {previous_line.strip()})")
                        if img_ref not in current_image_refs:
                            current_image_refs.append(img_ref)
                    else:
                        # This image is a question figure (diagram)
                        if img_ref not in current_question_images:
                            current_question_images.append(img_ref)
                            logger.info(f"✓ Detected question figure: {img_ref} (no option label detected)")
                        if img_ref not in current_image_refs:
                            current_image_refs.append(img_ref)

            previous_line = line_stripped

            # Detect question - improved detection with Q. numbering
            # Match patterns: Q. 1, Q 2, Q.14, ## Q. 15, Question 5, etc.
            # Strip markdown heading markers (##, ###) before checking
            line_without_heading = re.sub(r'^#+\s*', '', line_stripped)
            is_question = (
                line_stripped.endswith("?") or
                line_stripped.startswith(("Question", "Problem", "Q.", "Q ")) or
                re.match(r'^\d+[\.\)]\s+', line_stripped) or  # 1. or 1) followed by space
                re.match(r'^Q\.?\s*\d+', line_stripped) or  # Q. 1, Q 2, Q.14
                re.match(r'^Q\.?\s*\d+', line_without_heading) or  # ## Q. 15 after stripping ##
                (line_stripped and line_stripped[0].isdigit() and ('.' in line_stripped or ')' in line_stripped))  # More flexible number detection
            )

            if is_question:
                logger.info(f"Detected new question starting: {line_stripped[:80]}...")
                # Save previous question if exists
                if current_question:
                    # NEW CLASSIFICATION: Use the already-classified image arrays
                    total_images = len(current_image_refs)
                    num_option_images = len(current_option_images)
                    num_question_figures = len(current_question_images)
                    has_text_options = len(current_options) > 0 and any(opt.strip() for opt in current_options)

                    # Determine final classification
                    final_question_images = current_question_images.copy()
                    final_options = current_options
                    is_image_based_mcq = False

                    # If we have option images (detected by label pattern), it's image-based MCQ
                    if num_option_images > 0:
                        # We have option images - check if text options are valid or corrupted
                        valid_text_options = [opt for opt in current_options if opt.strip()]

                        if not valid_text_options:
                            # No valid text options - pure image-based MCQ
                            logger.info(f"📊 Image-based MCQ: {num_option_images} option images, {num_question_figures} question figures")
                            final_options = []  # Empty options - images will be mapped to A, B, C, etc. later
                            is_image_based_mcq = True
                        else:
                            # Both option images AND valid text - unusual mixed case
                            logger.warning(f"⚠️ Mixed question: {num_option_images} option images + {len(valid_text_options)} text options")
                            final_options = valid_text_options
                            is_image_based_mcq = False
                    elif total_images > 0 and not has_text_options:
                        # Fallback: No labels detected, but images exist and no text options
                        # Assume all are option images if 3+
                        if total_images >= 3:
                            logger.info(f"✅ Fallback RULE 1: {total_images} images, no labels, no text → Treating as option images")
                            final_question_images = []
                            final_options = []
                            is_image_based_mcq = True
                        else:
                            logger.info(f"📊 {total_images} question figures (no option labels detected)")
                    else:
                        # Text-based MCQ or question with diagrams
                        logger.info(f"📝 Text-based question: {len(current_options)} text options, {num_question_figures} question figures")

                    logger.info(f"Extracted question: {len(final_options)} options, {len(final_question_images)} question figures, {total_images} total images")
                    
                    # If skip_option_extraction is True, include options in the question text
                    if skip_option_extraction and final_options:
                        # Build question text with options inline
                        inline_question_text = current_question_text + "\n\n"
                        for idx, opt in enumerate(final_options):
                            label = chr(65 + idx)  # A, B, C, D...
                            inline_question_text += f"({label}) {opt}\n"
                        
                        questions.append(ExtractedQuestion(
                            id=str(uuid.uuid4()),
                            text=inline_question_text.strip(),
                            options=[],  # Empty options - they're now in the text
                            metadata={
                                "subject": subject,
                                "difficulty": difficulty,
                                "page": page.get("index", 0),
                                "image_refs": current_image_refs,
                                "question_image_refs": final_question_images,
                                "is_image_based_mcq": is_image_based_mcq,
                                "options_inline": True  # Flag to indicate options are in text
                            }
                        ))
                        logger.info(f"📝 Practice mode: Included {len(final_options)} options inline with question text")
                    else:
                        questions.append(ExtractedQuestion(
                            id=str(uuid.uuid4()),
                            text=current_question_text,
                            options=final_options,
                            metadata={
                                "subject": subject,
                                "difficulty": difficulty,
                                "page": page.get("index", 0),
                                "image_refs": current_image_refs,  # All images (question + option)
                                "question_image_refs": final_question_images,  # Only question figures
                                "is_image_based_mcq": is_image_based_mcq
                            }
                        ))

                # Use the line without heading markers for question text
                current_question = line_without_heading
                current_question_text = line_without_heading
                current_options = []
                current_image_refs = []
                current_question_images = []
                current_option_images = []
                
                # Ingest pending images (found before this question started)
                if pending_images:
                    logger.info(f"Attaching {len(pending_images)} pending images to new question")
                    current_image_refs.extend(pending_images)
                    # Assume pending images are question figures
                    for img in pending_images:
                        if img not in current_question_images:
                            current_question_images.append(img)
                    pending_images = []

                accumulating_option_idx = None

            # Detect text options - ONLY accept lines that start with option label
            # This is consistent with image option detection logic
            elif current_question and not image_refs:
                # Check if this line starts with an option marker (A. / (A) / A) / (1) / (i) etc.)
                # Updated regex to be more inclusive of formats like (a), (1), (i), etc.
                option_match = re.match(r'^\s*(?:\(|\[)?([A-Za-z]|[0-9]{1,2}|[ivxIVX]+)[\.|\)]\s*(.*)', line_stripped)
                if option_match:
                    option_label = option_match.group(1).upper()
                    option_text = option_match.group(2).strip()

                    # Start new option accumulation slot
                    cleaned = _clean_option_text(option_text) if option_text else f"Option {option_label}"
                    logger.debug(
                        f"Detected text option {option_label}: {cleaned[:80] if cleaned else '(empty)'}..."
                    )
                    current_options.append(cleaned)
                    accumulating_option_idx = len(current_options) - 1
                else:
                    # If we're in the middle of an option and the current line looks like a continuation,
                    # append it to the last option (helps for LaTeX broken across lines)
                    if accumulating_option_idx is not None:
                        # Stop accumulating if the line looks like a heading for a new question
                        is_new_question_like = (
                            line_stripped.startswith(("Question", "Problem", "Q.", "Q ")) or
                            re.match(r'^\d+[\.\)]\s+', line_stripped) is not None or
                            re.match(r'^Q\.?\s*\d+', line_stripped) is not None
                        )
                        if not is_new_question_like:
                            addon = line_stripped
                            # Avoid accidental "Answer:" or "Solution:" lines
                            if not re.match(r'^(Answer|Solution)\b', addon, re.IGNORECASE):
                                merged = (current_options[accumulating_option_idx] + " " + addon).strip()
                                current_options[accumulating_option_idx] = _clean_option_text(merged)
                                logger.debug(
                                    f"Appended continuation to option {accumulating_option_idx}: "
                                    f"{current_options[accumulating_option_idx][:120]}..."
                                )
                        else:
                            # Stop accumulating when a new question-ish line appears
                            accumulating_option_idx = None

        # Add last question if exists
        if current_question:
            # NEW CLASSIFICATION: Use the already-classified image arrays
            total_images = len(current_image_refs)
            num_option_images = len(current_option_images)
            num_question_figures = len(current_question_images)
            has_text_options = len(current_options) > 0 and any(opt.strip() for opt in current_options)

            # Determine final classification
            final_question_images = current_question_images.copy()
            final_options = current_options
            is_image_based_mcq = False

            # If we have option images (detected by label pattern), it's image-based MCQ
            if num_option_images > 0:
                # We have option images - check if text options are valid or corrupted
                valid_text_options = [opt for opt in current_options if opt.strip()]

                if not valid_text_options:
                    # No valid text options - pure image-based MCQ
                    logger.info(f"📊 Image-based MCQ (last): {num_option_images} option images, {num_question_figures} question figures")
                    final_options = []  # Empty options - images will be mapped to A, B, C, etc. later
                    is_image_based_mcq = True
                else:
                    # Both option images AND valid text - unusual mixed case
                    logger.warning(f"⚠️ Mixed question (last): {num_option_images} option images + {len(valid_text_options)} text options")
                    final_options = valid_text_options
                    is_image_based_mcq = False
            elif total_images > 0 and not has_text_options:
                # Fallback: No labels detected, but images exist and no text options
                # Assume all are option images if 3+
                if total_images >= 3:
                    logger.info(f"✅ Fallback RULE 1 (last): {total_images} images, no labels, no text → Treating as option images")
                    final_question_images = []
                    final_options = []
                    is_image_based_mcq = True
                else:
                    logger.info(f"📊 {total_images} question figures (last, no option labels detected)")
            else:
                # Text-based MCQ or question with diagrams
                logger.info(f"📝 Text-based question (last): {len(current_options)} text options, {num_question_figures} question figures")

            # Final clean-up for text options (dedupe LaTeX/plain echoes)
            if final_options:
                final_options = [_clean_option_text(o) for o in final_options]

            logger.info(
                f"Extracted last question: {len(final_options)} options, "
                f"{len(final_question_images)} question figures, {total_images} total images"
            )
            
            # If skip_option_extraction is True, include options in the question text
            if skip_option_extraction and final_options:
                # Build question text with options inline
                inline_question_text = current_question_text + "\n\n"
                for idx, opt in enumerate(final_options):
                    label = chr(65 + idx)  # A, B, C, D...
                    inline_question_text += f"({label}) {opt}\n"
                
                questions.append(ExtractedQuestion(
                    id=str(uuid.uuid4()),
                    text=inline_question_text.strip(),
                    options=[],  # Empty options - they're now in the text
                    metadata={
                        "subject": subject,
                        "difficulty": difficulty,
                        "page": page.get("index", 0),
                        "image_refs": current_image_refs,
                        "question_image_refs": final_question_images,
                        "is_image_based_mcq": is_image_based_mcq,
                        "options_inline": True
                    }
                ))
                logger.info(f"📝 Practice mode (last): Included {len(final_options)} options inline with question text")
            else:
                questions.append(ExtractedQuestion(
                    id=str(uuid.uuid4()),
                    text=current_question_text,
                    options=final_options,
                    metadata={
                        "subject": subject,
                        "difficulty": difficulty,
                        "page": page.get("index", 0),
                        "image_refs": current_image_refs,
                        "question_image_refs": final_question_images,  # Only question figures
                        "is_image_based_mcq": is_image_based_mcq
                    }
                ))

    return questions


async def run_document_ocr_pipeline(
    document: Dict[str, Any],
    pdf_base64: str,
    job_id: str,
    processing_result: Dict[str, Any],
    current_user: Dict[str, Any],
    db: DatabaseManager,
    cache: CacheManager
) -> PDFProcessingResult:
    """Run the full OCR extraction pipeline for a stored document."""
    document_id = document["document_id"]
    try:
        logger.info(f"Calling Mistral OCR API for job {job_id}")
        ocr_result = await call_mistral_ocr(pdf_base64)

        processing_result["progress"] = 60
        await cache.set(f"pdf_job:{job_id}", processing_result, 3600, "admin")

        document_type = document.get("document_type", "Chapter Notes")
        logger.info(f"Extracting questions from OCR result for job {job_id}, document_type: {document_type}")
        
        # For Practice Sets, don't extract options separately - keep them in question text
        skip_option_extraction = document_type == "Practice Sets"
        
        extracted_questions = extract_questions_from_ocr(
            ocr_result,
            document.get("subject", "General"),
            document.get("difficulty", "medium"),
            skip_option_extraction=skip_option_extraction
        )
        
        if skip_option_extraction:
            logger.info(f"📝 Practice Sets mode: Options kept inline with question text")

        logger.info(f"Processing extracted images for job {job_id}, document type: {document_type}")

        all_images: List[Dict[str, Any]] = []
        image_base64_map: Dict[str, Dict[str, Any]] = {}

        # Determine if B2C admin
        is_b2c = is_b2c_admin(current_user)

        for page in ocr_result.get("pages", []):
            for img in page.get("images", []):
                if img.get("image_base64"):
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

        logger.info(f"Saved {len(all_images)} images to disk and database")
        logger.info(f"Image base64 map contains {len(image_base64_map)} entries")

        processing_result["progress"] = 80
        processing_result["extracted_questions"] = len(extracted_questions)
        processing_result["extracted_images"] = len(all_images)
        await cache.set(f"pdf_job:{job_id}", processing_result, 3600, "admin")

        logger.info(f"Storing {len(extracted_questions)} questions for {document_type}")

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
                            for mistral_img in page.get("images", []):
                                mistral_img_id = mistral_img.get('id')
                                base_img_id = mistral_img_id.split('.')[0] if '.' in mistral_img_id else mistral_img_id

                                is_referenced = any(
                                    base_img_id in ref or mistral_img_id in ref
                                    for ref in image_refs
                                )

                                if not is_referenced:
                                    logger.debug(f"Skipping non-referenced image {mistral_img_id}")
                                    continue

                                logger.info(f"Including {mistral_img_id} - referenced in question")

                                # Find saved images - check for both exact match and split variants
                                # If image was split, the IDs become img-X-A, img-X-B, etc.
                                matching_saved_images = [
                                    img for img in all_images 
                                    if img['id'] == base_img_id or img['id'].startswith(f"{base_img_id}-")
                                ]
                                
                                img_base64_data = image_base64_map.get(mistral_img_id) or image_base64_map.get(base_img_id, {})

                                if matching_saved_images and img_base64_data:
                                    is_question_figure = any(
                                        base_img_id in ref or mistral_img_id in ref
                                        for ref in question_image_refs
                                    )

                                    is_image_based_mcq = question.metadata.get("is_image_based_mcq", False)
                                    if is_image_based_mcq and not is_question_figure:
                                        is_question_figure = False
                                        logger.info(f"Treating {mistral_img_id} as option image for image-based MCQ")
                                    
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
                                                'source': 'mistral_ocr',
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
                                                    'source': 'mistral_ocr',
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
                                        logger.warning(f"⚠️ Image {mistral_img_id} not found in image_base64_map (keys: {list(image_base64_map.keys())[:10]}...)")

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
            if is_b2c:
                await db.b2c_insert_one("questions", question_doc)
            else:
                await db.mongo_insert_one("questions", question_doc)

            # Store richer metadata so other services can reconstruct the question fully
            import json as _json_for_full
            chromadb_metadata = {
                "document_id": document_id,
                "document_type": document_type,
                "subject": document.get("subject", "General"),
                "difficulty": document.get("difficulty", "medium"),
                # Align with legacy readers that expect pdfSource
                "pdfSource": document_id,
                # Include serialized full data for robust reconstruction paths
                "fullData": _json_for_full.dumps(question_doc, default=str),
                "page": question.metadata.get("page", 0)
                if isinstance(question.metadata.get("page", 0), (int, float)) else 0
            }

            await db.chroma_add(
                [question.id],
                [question.text],
                [chromadb_metadata]
            )

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

        logger.info(f"OCR processing completed for document {document_id}")
        return PDFProcessingResult(**processing_result)
    except Exception as exc:
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
    current_user: Dict[str, Any] = Depends(require_admin),
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

        # Create document metadata
        # Attach tenant context
        try:
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
            "ocr_status": "not_processed",
            "ocr_job_id": None,
            "extracted_questions_count": 0,
            "extracted_images_count": 0,
            "pages_count": pages_count,  # Store page count for Notes display
            "total_points": total_points if document_type == "Test Series" else None,
            "total_minutes": total_minutes if document_type == "Test Series" else None,
            "is_validated": False,
            "is_active": True,  # Default to enabled
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
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Trigger OCR processing on an existing uploaded document."""
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

        pdf_base64 = base64.b64encode(file_content).decode('utf-8')
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

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to prepare OCR job for {document_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start OCR processing: {exc}"
        )

    async def execute_pipeline() -> PDFProcessingResult:
        return await run_document_ocr_pipeline(
            document=document,
            pdf_base64=pdf_base64,
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
            except HTTPException:
                pass
            except Exception as exc:
                logger.error(f"Background OCR job {job_id} failed: {exc}", exc_info=True)

        task = asyncio.create_task(background_runner())
        if isinstance(tasks, dict):
            tasks[job_id] = task

            def _cleanup(_):
                tasks.pop(job_id, None)

            task.add_done_callback(_cleanup)

        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=jsonable_encoder(PDFProcessingResult(**processing_result))
        )

    try:
        return await execute_with_semaphore()
    except HTTPException:
        raise
    except Exception as exc:
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
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )

        file_content = await file.read()
        pdf_base64 = base64.b64encode(file_content).decode("utf-8")

        async def _run_ocr() -> Dict[str, Any]:
            return await call_mistral_ocr(pdf_base64)

        semaphore = getattr(request.app.state, "ocr_semaphore", None)
        if semaphore:
            async with semaphore:
                ocr_result = await _run_ocr()
        else:
            ocr_result = await _run_ocr()

        return {
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

    except HTTPException:
        raise
    except Exception as exc:
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
    current_user: Dict[str, Any] = Depends(require_admin),
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
            # Regular admin/tutor - query skillbot_db
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
        # NOTE: We intentionally do NOT filter by student_subjects here - students should
        # see ALL content uploaded by their admin. The assigned subjects are for 
        # personalization/recommendations, not for restricting access to content.

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
                "subject": doc["subject"],
                "difficulty": doc["difficulty"],
                "course_plan": doc.get("course_plan"),
                "standard": doc.get("standard"),
                "extracted_questions_count": doc.get("extracted_questions_count", 0),
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
    current_user: Dict[str, Any] = Depends(require_admin),
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
        total_points = sum(q.get("points", 1.0) for q in questions)

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
    current_user: Dict[str, Any] = Depends(require_admin),
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

        # Verify the user has access to this document
        # For students, check if document belongs to their admin
        if current_user.get("user_type") == "student":
            # Normalize types for comparison
            student_admin_id = str(current_user.get("admin_id")) if current_user.get("admin_id") is not None else None
            document_admin_id = document.get("admin_id")
            document_admin_id_str = str(document_admin_id) if document_admin_id is not None else None

            # In development mode, allow cross-admin access to simplify testing
            from config_async import DEBUG_MODE as _DEBUG_MODE
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
        elif user_type not in ["b2c_admin", "b2c_user"]:
            # For regular admins, verify they own the document (type-safe)
            admin_id = str(current_user.get("user_id")) if current_user.get("user_id") is not None else None
            document_admin_id = document.get("admin_id")
            document_admin_id_str = str(document_admin_id) if document_admin_id is not None else None

            from config_async import DEBUG_MODE as _DEBUG_MODE
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
        # B2C admins can access all B2C documents (no admin_id check needed)

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
    current_user: Dict[str, Any] = Depends(require_admin),
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

        # Also add to ChromaDB for searchability and MCQ retrieval
        try:
            chromadb_metadata = {
                "document_id": document_id or full_question_id,
                "document_type": document_type,
                "course_plan": course_plan,
                "standard": standard,
                "subject": subject,
                "difficulty": difficulty,
                "source": "manual_creation",
                "created_by": current_user.get("user_id"),
                "created_at": datetime.utcnow().isoformat()
            }

            await db.chroma_add(
                [full_question_id],
                [question_text],
                [chromadb_metadata]
            )
            logger.info(f"Added question {full_question_id} to ChromaDB")
        except Exception as e:
            logger.warning(f"Failed to add question to ChromaDB: {str(e)}")
            # Don't fail the request if ChromaDB insertion fails

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
    current_user: Dict[str, Any] = Depends(require_admin),
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

        # Update in ChromaDB with proper metadata (CRITICAL for categorization)
        try:
            # Get updated question data from appropriate database
            if is_b2c:
                updated_question = await db.b2c_find_one("questions", {"id": question_id})
            else:
                updated_question = await db.mongo_find_one("questions", {"id": question_id})

            # Build updated ChromaDB metadata with all fields
            chromadb_metadata = {
                "document_id": updated_question.get("document_id", question_id),
                "document_type": updated_question.get("document_type", "Chapter Notes"),  # CRITICAL!
                "subject": updated_question.get("subject", "General"),
                "difficulty": updated_question.get("difficulty", "medium"),
                "hasImages": len(updated_question.get("images", [])) > 0 or len(updated_question.get("question_figures", [])) > 0,
                "imageCount": len(updated_question.get("images", [])) + len(updated_question.get("question_figures", [])),
                "source": "manual_edit",
                "updated_by": current_user.get("user_id"),
                "updated_at": datetime.utcnow().isoformat()
            }

            # Update ChromaDB (delete and re-add with updated metadata)
            await db.chroma_delete(ids=[question_id])
            await db.chroma_add(
                [question_id],
                [updated_question.get("text", "")],
                [chromadb_metadata]
            )
            logger.info(f"Updated question {question_id} in ChromaDB with document_type={chromadb_metadata['document_type']}")
        except Exception as e:
            logger.warning(f"Failed to update ChromaDB: {str(e)}")
        # Don't fail the request if ChromaDB update fails

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

                    total_points = sum(q.get("points", 1.0) for q in all_questions)

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

@router.delete("/questions/{question_id}")
@limiter.limit("30/minute")
async def delete_question(
    request: Request,
    question_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
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

        # Also delete from ChromaDB if it exists there
        try:
            await db.chroma_delete(ids=[question_id])
            logger.info(f"Deleted question {question_id} from ChromaDB")
        except Exception as e:
            logger.warning(f"Failed to delete from ChromaDB (may not exist there): {str(e)}")

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
    current_user: Dict[str, Any] = Depends(require_admin),
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
    current_user: Dict[str, Any] = Depends(require_admin),
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
    current_user: Dict[str, Any] = Depends(require_admin),
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
    current_user: Dict[str, Any] = Depends(require_admin),
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
    current_user: Dict[str, Any] = Depends(require_admin),
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

        for question in questions:
            # Delete from ChromaDB
            try:
                await db.chroma_delete(question["id"])
                logger.debug(f"Deleted question {question['id']} from ChromaDB")
            except Exception as e:
                logger.warning(f"Failed to delete question {question['id']} from ChromaDB: {str(e)}")

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
    current_user: Dict[str, Any] = Depends(require_admin),
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
    current_user: Dict[str, Any] = Depends(require_admin),
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
    
    Args:
        pdf_content: Raw PDF bytes
        page_number: Page number (1-indexed)
        bbox: Bounding box as percentages {x, y, width, height}
        region_id: ID of the region being processed
    
    Returns:
        Dict with extracted text and images
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
        
        # Render the clipped region at high resolution
        mat = fitz.Matrix(3.0, 3.0)  # 3x zoom for better OCR quality
        pix = page.get_pixmap(matrix=mat, clip=clip_rect)
        
        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
        
        # Encode as base64 for Mistral OCR
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Close document
        doc.close()
        
        # Send to Mistral OCR as an image
        import aiohttp
        
        if not MISTRAL_API_KEY:
            return {
                "success": False,
                "error": "Mistral API key not configured"
            }
        
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Use Pixtral model for image-based OCR
        payload = {
            "model": "pixtral-12b-2409",  # Pixtral model for image understanding
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Extract all text from this question image. This is a question from an exam or practice paper.

Please extract:
1. The complete question text
2. All options (if this is an MCQ, label them as A), B), C), D))
3. Any mathematical formulas or equations (use LaTeX notation)
4. Descriptions of any diagrams or figures

Format the output as:
QUESTION: [question text]
OPTIONS:
A) [option A text]
B) [option B text]
C) [option C text]
D) [option D text]
FIGURES: [description of any images/diagrams if present]

If this is not an MCQ, just provide the question text and any figure descriptions."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Mistral API error for region {region_id}: {error_text}")
                    return {
                        "success": False,
                        "error": f"OCR API error: {response.status}"
                    }
                
                result = await response.json()
                extracted_text = result['choices'][0]['message']['content'].strip()
                
                logger.info(f"Successfully extracted text for region {region_id}: {len(extracted_text)} chars")
                
                return {
                    "success": True,
                    "extractedText": extracted_text,
                    "imageBase64": img_base64  # Keep for reference
                }
                
    except ImportError as e:
        logger.error(f"Missing dependency for region extraction: {e}")
        return {
            "success": False,
            "error": f"Missing dependency: {e}. Install PyMuPDF with 'pip install pymupdf'"
        }
    except Exception as e:
        logger.error(f"Error extracting region {region_id}: {str(e)}")
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
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """
    Process OCR on specific regions of a document.
    
    This endpoint:
    1. Retrieves saved regions for the document
    2. For each region, extracts that portion of the PDF page
    3. Sends each region to Mistral OCR
    4. Parses the OCR result and creates questions
    5. Returns the results
    """
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
                
                results.append({
                    "regionId": region_id,
                    "success": True,
                    "extractedText": extracted_text,
                    "extractedOptions": options if options else None,
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
                    "document_type": document.get("document_type", "Practice Sets"),
                    "extracted_at": datetime.utcnow(),
                    "pdf_source": document.get("filename", ""),
                    "document_id": document_id,
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
                    "points": 1.0,
                    "penalty": 0.0,
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
                    "ocr_completed_at": datetime.utcnow()
                }}
            )
        
        logger.info(f"Region OCR completed for {document_id}: {successful} successful, {failed} failed")
        
        return {
            "success": True,
            "documentId": document_id,
            "processedRegions": len(regions_to_process),
            "successfulRegions": successful,
            "failedRegions": failed,
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
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
