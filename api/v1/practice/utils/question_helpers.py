"""
Practice Module - Question Helper Utilities
Functions for loading questions and extracting images
"""

import os
import re
import base64 as base64_module
import logging
from typing import Optional, Dict, Any, List

from core.database import DatabaseManager

logger = logging.getLogger(__name__)


async def load_question_doc(
    db: DatabaseManager,
    qid: str,
    is_b2c: bool = False
) -> Dict[str, Any]:
    """Fetch question from Chroma (fullData) with Mongo fallback.

    For B2C users, falls back to B2C database instead of main database.

    Args:
        db: Database manager instance
        qid: Question ID
        is_b2c: Whether user is B2C (uses B2C database)

    Returns:
        Question document dict
    """
    try:
        chroma = await db.chroma_get(ids=[qid])
        metas = chroma.get("metadatas") or []
        if metas and metas[0].get("fullData"):
            import json as _json
            return _json.loads(metas[0]["fullData"]) or {}
    except Exception:
        pass

    # Fallback to MongoDB - use B2C database for B2C users
    if is_b2c:
        return await db.b2c_find_one("questions", {"id": qid}) or {}
    return await db.mongo_find_one("questions", {"id": qid}) or {}


def options_text_from_question(q: Dict[str, Any]) -> str:
    """Extract options as formatted text from question document.

    Handles both simple options array and enhancedOptions with types.

    Args:
        q: Question document

    Returns:
        Formatted options string (e.g., "A. option1\nB. option2")
    """
    opts = q.get("options", []) or []
    if opts:
        return "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(opts)])

    enh = q.get("enhancedOptions") or []
    if enh:
        parts = []
        for i, opt in enumerate(enh):
            label = chr(65 + i)
            if isinstance(opt, dict):
                if opt.get("type") == "text" and opt.get("content"):
                    parts.append(f"{label}. {opt.get('content')}")
                elif opt.get("type") == "image":
                    desc = opt.get("description") or "image option"
                    parts.append(f"{label}. [IMAGE] {desc}")
            else:
                parts.append(f"{label}. {str(opt)}")
        return "\n".join(parts)
    return ""


async def _load_image_by_id(
    db: DatabaseManager,
    img_id: str,
    is_b2c: bool = False
) -> Optional[str]:
    """Load image data from database by ID.

    Tries base64Data first, then reads from file_path if available.

    Args:
        db: Database manager
        img_id: Image document ID
        is_b2c: Whether to use B2C database

    Returns:
        Base64 data URL string or None
    """
    if not img_id or not db:
        return None
    try:
        if is_b2c:
            img_doc = await db.b2c_find_one("images", {"_id": img_id})
        else:
            img_doc = await db.mongo_find_one("images", {"_id": img_id})

        if img_doc:
            if img_doc.get("base64Data"):
                b64 = img_doc["base64Data"]
                logger.info(f"Loaded base64Data from database for image {img_id}")
                return b64
            elif img_doc.get("file_path"):
                file_path = img_doc["file_path"]
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        image_bytes = f.read()
                        base64_encoded = base64_module.b64encode(image_bytes).decode('utf-8')
                        content_type = img_doc.get("content_type", "image/jpeg")
                        if not content_type.startswith("image/"):
                            content_type = "image/jpeg"
                        b64 = f"data:{content_type};base64,{base64_encoded}"
                        logger.info(f"Loaded image from disk for {img_id}: {len(b64)} bytes")
                        return b64
    except Exception as e:
        logger.error(f"Failed to load image {img_id}: {e}")
    return None


def _normalize_b64(b64: str) -> str:
    """Normalize base64 string to data URL format."""
    if b64 and not b64.startswith("data:image"):
        return f"data:image/png;base64,{b64}"
    return b64


async def figure_images_base64(
    q: Dict[str, Any],
    db: DatabaseManager = None,
    is_b2c: bool = False
) -> List[str]:
    """Extract base64 image data for question figures.

    Loads images from disk/database if base64Data is not embedded.
    This ensures question diagram images are always available for LLM evaluation.

    Args:
        q: Question document
        db: Database manager for loading images from disk (optional)
        is_b2c: Whether this is a B2C user (uses B2C database)

    Returns:
        List of base64 data URLs for question figures
    """
    imgs: List[str] = []

    # 1. Extract from question_figures array
    for fig_ref in (q.get("question_figures", []) or []):
        try:
            b64 = None
            fig_id = None

            if isinstance(fig_ref, dict):
                b64 = fig_ref.get("base64Data")
                fig_id = fig_ref.get("id")
            elif isinstance(fig_ref, str):
                fig_id = fig_ref

            if not b64 and fig_id:
                b64 = await _load_image_by_id(db, fig_id, is_b2c)

            if b64:
                imgs.append(_normalize_b64(b64))

        except Exception as e:
            logger.warning(f"Error processing figure: {e}")

    # 2. Extract from images array (diagrams and other embedded images)
    for img_ref in (q.get("images", []) or []):
        try:
            b64 = None
            img_id = None

            if isinstance(img_ref, dict):
                b64 = img_ref.get("base64Data")
                img_id = img_ref.get("id")
            elif isinstance(img_ref, str):
                img_id = img_ref

            if not b64 and img_id:
                b64 = await _load_image_by_id(db, img_id, is_b2c)

            if b64:
                imgs.append(_normalize_b64(b64))
        except Exception:
            pass

    # 3. Check for inline figure in the question text (base64 embedded)
    if q.get("figure") and isinstance(q.get("figure"), str):
        fig = q["figure"]
        if fig.startswith("data:image") or len(fig) > 100:  # Likely base64
            imgs.append(_normalize_b64(fig))

    # 4. Check for questionImage field
    if q.get("questionImage"):
        qimg = q["questionImage"]
        if isinstance(qimg, str):
            if qimg.startswith("data:image") or len(qimg) > 100:
                imgs.append(_normalize_b64(qimg))
            else:
                # It might be an ID
                loaded = await _load_image_by_id(db, qimg, is_b2c)
                if loaded:
                    imgs.append(_normalize_b64(loaded))
        elif isinstance(qimg, dict):
            b64 = qimg.get("base64Data")
            if not b64:
                b64 = await _load_image_by_id(db, qimg.get("id"), is_b2c)
            if b64:
                imgs.append(_normalize_b64(b64))

    logger.info(f"Extracted {len(imgs)} question figure/diagram images for LLM evaluation")
    return imgs


async def option_images_base64(
    q: Dict[str, Any],
    db: DatabaseManager = None,
    is_b2c: bool = False
) -> List[Dict[str, str]]:
    """Extract images from MCQ options.

    Args:
        q: Question document
        db: Database manager
        is_b2c: Whether to use B2C database

    Returns:
        List of dicts with 'option' (A, B, C, D) and 'image' (base64 data URL)
    """
    option_images: List[Dict[str, str]] = []

    # Check enhancedOptions for image options
    enh = q.get("enhancedOptions") or []
    for i, opt in enumerate(enh):
        label = chr(65 + i)  # A, B, C, D...
        if isinstance(opt, dict) and opt.get("type") == "image":
            b64 = opt.get("base64Data")
            if not b64:
                b64 = await _load_image_by_id(db, opt.get("id"), is_b2c)
            if b64:
                option_images.append({
                    "option": label,
                    "image": _normalize_b64(b64)
                })

    # Also check regular options array for image objects
    opts = q.get("options") or []
    for i, opt in enumerate(opts):
        label = chr(65 + i)
        if isinstance(opt, dict):
            if opt.get("type") == "image" or opt.get("image"):
                b64 = opt.get("base64Data") or opt.get("image")
                if not b64:
                    b64 = await _load_image_by_id(db, opt.get("id"), is_b2c)
                if b64:
                    option_images.append({
                        "option": label,
                        "image": _normalize_b64(b64)
                    })

    if option_images:
        logger.info(f"Extracted {len(option_images)} option images: {[o['option'] for o in option_images]}")

    return option_images


def normalize_choice_text(s: str) -> str:
    """Normalize MCQ choice text to uppercase letter.

    Extracts single letter A-Z from input text.

    Args:
        s: Input text

    Returns:
        Uppercase letter or cleaned text
    """
    t = (s or '').upper().strip()
    m = re.search(r"\b([A-Z])\b", t)
    return m.group(1) if m else t


def normalize_numeric_text(s: str) -> str:
    """Normalize numeric text for comparison.

    Handles comma/period variations and time-like formats.

    Args:
        s: Input text

    Returns:
        Normalized numeric string
    """
    t = (s or '').strip().replace(' ', '').replace(',', '.')
    if ':' in t and t.count(':') == 1 and all(part.isdigit() for part in t.split(':')):
        t = t.replace(':', '.')
    return t
