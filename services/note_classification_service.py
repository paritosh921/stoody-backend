"""
Note Classification Service for Stoody

Pipeline: Render strokes → OCR → Classify → Store
- OCR: Mistral (primary, cheap for handwriting) → OpenAI GPT-4o Vision (fallback)
- Classify text-heavy pages: Groq llama-3.1-8b (near-free)
- Classify diagram/equation pages: GPT-4o-mini Vision
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from openai import AsyncOpenAI
from pymongo.errors import DuplicateKeyError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_OCR_MODEL = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_CLASSIFY_MODEL = os.getenv("GROQ_CLASSIFY_MODEL", "llama-3.1-8b-instant")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VISION_CLASSIFY_MODEL = os.getenv("VISION_CLASSIFY_MODEL", "gpt-4o-mini")

DEFAULT_SUBJECTS = [
    "Physics", "Chemistry", "Mathematics", "Language",
    "History", "Hindi", "Unorganised",
]

# Minimum new-stroke % before auto-reclassification
RECLASSIFY_THRESHOLD = 0.20

# Debounce window in seconds
DEBOUNCE_SECONDS = 60


# ---------------------------------------------------------------------------
# Queue management
# ---------------------------------------------------------------------------

async def queue_classification(
    db,
    db_name: str,
    user_id: str,
    pen_mac: str,
    book_type: Optional[str],
    page_number: Optional[int],
    copy_id: Optional[str] = None,
):
    """Upsert into classification_queue with 60s debounce."""
    if not db_name or not user_id or page_number is None:
        return
    try:
        tenant_db = await db.get_tenant_db(db_name)
        if tenant_db is None:
            return
        now = datetime.utcnow()
        queue_key: Dict[str, Any] = {
            "user_id": user_id,
            "pen_mac": (pen_mac or "").upper(),
            "book_type": book_type or "A5",
            "page_number": page_number,
            "db_name": db_name,
        }
        if copy_id:
            queue_key["copy_id"] = copy_id
        queue_update = {
            "$set": {
                "status": "pending",
                "process_after": now + timedelta(seconds=DEBOUNCE_SECONDS),
                "updated_at": now,
                "attempts": 0,
                "error": None,
                "cancel_requested": False,
            },
            "$unset": {
                "started_at": "",
                "completed_at": "",
                "cancelled_at": "",
            },
            "$setOnInsert": {
                "queued_at": now,
                "created_at": now,
            },
        }
        try:
            await tenant_db["classification_queue"].update_one(
                queue_key,
                queue_update,
                upsert=True,
            )
            logger.info(
                "Queued AI classification for user=%s pen=%s book=%s page=%s copy=%s process_after=%s",
                user_id,
                (pen_mac or "").upper(),
                book_type or "A5",
                page_number,
                copy_id or "default",
                (now + timedelta(seconds=DEBOUNCE_SECONDS)).isoformat(),
            )
        except DuplicateKeyError as exc:
            legacy_key = {
                "user_id": user_id,
                "pen_mac": (pen_mac or "").upper(),
                "book_type": book_type or "A5",
                "page_number": page_number,
                "db_name": db_name,
            }
            legacy_doc = await tenant_db["classification_queue"].find_one(legacy_key, {"_id": 1, "copy_id": 1})
            if not legacy_doc:
                raise
            fallback_update = dict(queue_update)
            fallback_set = dict(fallback_update.get("$set", {}))
            if copy_id and not legacy_doc.get("copy_id"):
                fallback_set["copy_id"] = copy_id
            fallback_update["$set"] = fallback_set
            await tenant_db["classification_queue"].update_one(
                {"_id": legacy_doc["_id"]},
                fallback_update,
                upsert=False,
            )
            logger.warning(
                "Recovered classification queue upsert via legacy identity fallback for user=%s pen=%s book=%s page=%s copy=%s: %s",
                user_id,
                (pen_mac or "").upper(),
                book_type or "A5",
                page_number,
                copy_id,
                exc,
            )
    except Exception as e:
        logger.warning(f"Failed to queue classification: {e}")


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

async def process_page(
    tenant_db,
    user_id: str,
    job: Dict[str, Any],
    openai_client: Optional[AsyncOpenAI] = None,
    should_cancel: Optional[Callable[[], Awaitable[bool]]] = None,
):
    """Full pipeline: render → OCR → classify → store."""

    pen_mac = job.get("pen_mac", "")
    book_type = job.get("book_type", "A5")
    page_number = job.get("page_number")
    copy_id = job.get("copy_id")

    page_key: Dict[str, Any] = {
        "user_id": user_id,
        "book_type": book_type,
        "page_number": page_number,
    }
    if copy_id:
        page_key["copy_id"] = copy_id

    async def _cancelled() -> bool:
        return bool(should_cancel and await should_cancel())

    # 1. Check reclassification threshold
    existing = await _find_existing_note_classification(
        tenant_db,
        user_id,
        pen_mac,
        book_type,
        page_number,
        copy_id=copy_id,
    )
    if existing and existing.get("classification_source") == "manual":
        logger.info(
            "Skipping AI classification for manual page user=%s pen=%s book=%s page=%s copy=%s",
            user_id,
            (pen_mac or "").upper(),
            book_type or "A5",
            page_number,
            copy_id or "default",
        )
        return  # never override manual classification
    if await _cancelled():
        return

    current_strokes = await _count_strokes(tenant_db, user_id, pen_mac, book_type, page_number, copy_id=copy_id)
    if existing and not _should_reclassify(existing, current_strokes):
        logger.info(
            "Skipping AI reclassification below threshold for user=%s pen=%s book=%s page=%s copy=%s strokes=%s",
            user_id,
            (pen_mac or "").upper(),
            book_type or "A5",
            page_number,
            copy_id or "default",
            current_strokes,
        )
        return
    if await _cancelled():
        return

    # 2. Fetch strokes → render SVG → PNG
    strokes = await _fetch_page_strokes(tenant_db, user_id, pen_mac, book_type, page_number, copy_id=copy_id)
    if not strokes:
        if await _cancelled():
            return
        await _save_classification(
            tenant_db, page_key, "Unorganised", "Empty Page", 0.5, "", None, 0
        )
        logger.info(
            "AI classification saved empty-page fallback for user=%s pen=%s book=%s page=%s copy=%s",
            user_id,
            (pen_mac or "").upper(),
            book_type or "A5",
            page_number,
            copy_id or "default",
        )
        return

    # Extract session info from stroke batches
    latest_session_id = strokes[-1].get("session_id") if strokes else None
    timestamps = [s.get("timestamp") for s in strokes if s.get("timestamp")]
    first_activity = min(timestamps) if timestamps else None
    last_activity = max(timestamps) if timestamps else None

    from utils.stroke_pdf_generator import build_svg_from_strokes, svg_to_png_bytes

    svg = build_svg_from_strokes(strokes, book_type=book_type)
    png_bytes = svg_to_png_bytes(svg, scale=0.5)
    if not png_bytes:
        if await _cancelled():
            return
        await _save_classification(
            tenant_db, page_key, "Unorganised", "Render Failed", 0.3, "", None, current_strokes
        )
        logger.warning(
            "AI classification render failed fallback for user=%s pen=%s book=%s page=%s copy=%s",
            user_id,
            (pen_mac or "").upper(),
            book_type or "A5",
            page_number,
            copy_id or "default",
        )
        return
    if await _cancelled():
        return

    # 3. Upload thumbnail to S3
    thumbnail_url = await _upload_thumbnail(png_bytes, user_id, pen_mac, book_type, page_number, copy_id=copy_id)

    # 4. OCR (Mistral primary, OpenAI fallback)
    ocr_text = await _ocr_page(png_bytes, openai_client)
    if await _cancelled():
        return

    # 5. Classify
    existing_topics = await _get_existing_topics(tenant_db, user_id, copy_id=copy_id)
    # Build allowed subjects: defaults + any user-created ones from DB
    all_subjects = set(DEFAULT_SUBJECTS)
    for et in existing_topics:
        all_subjects.add(et["subject"])

    if len(ocr_text.strip()) > 50:
        result = await _text_classify(ocr_text, existing_topics, all_subjects)
    else:
        result = await _vision_classify(png_bytes, existing_topics, openai_client, all_subjects)

    subject = result.get("subject", "Unorganised")
    topic = result.get("topic", "General")
    confidence = result.get("confidence", 0.5)

    if subject not in all_subjects:
        subject = "Unorganised"

    # 6. Upsert into note_classifications
    if await _cancelled():
        return
    await _save_classification(
        tenant_db, page_key, subject, topic, confidence,
        ocr_text, thumbnail_url, current_strokes, existing,
        session_id=latest_session_id,
        first_activity=first_activity,
        last_activity=last_activity,
    )
    logger.info(
        "AI classification saved for user=%s pen=%s book=%s page=%s copy=%s subject=%s topic=%s confidence=%.2f",
        user_id,
        (pen_mac or "").upper(),
        book_type or "A5",
        page_number,
        copy_id or "default",
        subject,
        topic,
        float(confidence or 0.0),
    )


# ---------------------------------------------------------------------------
# Stroke fetching
# ---------------------------------------------------------------------------

def _build_user_id_match(user_id: str) -> dict:
    """Build a user_id match clause that handles all storage formats
    (string, ObjectId, username). Mirrors copies_async.py logic."""
    from bson import ObjectId as _OID

    identifiers: list = [user_id, str(user_id)]
    try:
        if _OID.is_valid(user_id):
            identifiers.append(_OID(user_id))
    except Exception:
        pass
    return {"$in": identifiers}


async def _count_strokes(
    tenant_db, user_id: str, pen_mac: str, book_type: str, page_number: int,
    *, copy_id: Optional[str] = None,
) -> int:
    # Prefer canvas_pages (copy-scoped) over legacy strokes
    cp_query: Dict[str, Any] = {
        "user_id": _build_user_id_match(user_id),
        "page_number": page_number,
        "book_type": book_type,
    }
    if copy_id:
        cp_query["copy_id"] = copy_id
    count = 0
    cursor = tenant_db["canvas_pages"].find(cp_query, {"stroke_count": 1, "strokes": 1})
    async for doc in cursor:
        sc = doc.get("stroke_count")
        if sc and isinstance(sc, int):
            count += sc
        elif doc.get("strokes"):
            count += len(doc["strokes"])
    if count > 0:
        return count
    # Fallback to legacy strokes collection
    query: Dict[str, Any] = {
        "user_id": _build_user_id_match(user_id),
        "pen_mac": {"$regex": f"^{pen_mac}$", "$options": "i"},
        "book_type": book_type,
        "page_number": page_number,
    }
    if copy_id:
        query["copy_id"] = copy_id
    cursor = tenant_db["strokes"].find(query, {"strokes": 1})
    async for doc in cursor:
        count += len(doc.get("strokes", []))
    return count


async def _fetch_page_strokes(
    tenant_db, user_id: str, pen_mac: str, book_type: str, page_number: int,
    *, copy_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    # Prefer canvas_pages (copy-scoped) over legacy strokes
    cp_query: Dict[str, Any] = {
        "user_id": _build_user_id_match(user_id),
        "page_number": page_number,
        "book_type": book_type,
    }
    if copy_id:
        cp_query["copy_id"] = copy_id
    cp_docs = await tenant_db["canvas_pages"].find(cp_query).sort("last_modified", 1).to_list(length=100)
    if cp_docs:
        all_strokes: List[Dict[str, Any]] = []
        for doc in cp_docs:
            for s in (doc.get("strokes") or []):
                s_copy = dict(s)
                if "session_id" not in s_copy and doc.get("session_id"):
                    s_copy["session_id"] = doc["session_id"]
                all_strokes.append(s_copy)
        if all_strokes:
            return all_strokes
    # Fallback to legacy strokes collection
    query: Dict[str, Any] = {
        "user_id": _build_user_id_match(user_id),
        "pen_mac": {"$regex": f"^{pen_mac}$", "$options": "i"},
        "book_type": book_type,
        "page_number": page_number,
    }
    if copy_id:
        query["copy_id"] = copy_id
    cursor = tenant_db["strokes"].find(query).sort("timestamp", 1)
    return await cursor.to_list(length=1000)


def _should_reclassify(existing: Dict[str, Any], current_stroke_count: int) -> bool:
    old = existing.get("stroke_count_at_classification", 0)
    if old == 0:
        return True
    return (current_stroke_count - old) / old >= RECLASSIFY_THRESHOLD


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

async def _ocr_page(png_bytes: bytes, openai_client: Optional[AsyncOpenAI] = None) -> str:
    """OCR handwritten page. Mistral primary, OpenAI Vision fallback."""
    b64 = base64.b64encode(png_bytes).decode()
    data_url = f"data:image/png;base64,{b64}"

    # Try Mistral OCR first (cheaper)
    if MISTRAL_API_KEY:
        try:
            return await _mistral_ocr_image(png_bytes)
        except Exception as e:
            logger.warning(f"Mistral OCR failed, falling back to OpenAI: {e}")

    # Fallback: OpenAI Vision
    if openai_client or OPENAI_API_KEY:
        try:
            client = openai_client or AsyncOpenAI(api_key=OPENAI_API_KEY)
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text from this handwritten page. Return only the extracted text, nothing else."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                max_tokens=2000,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"OpenAI Vision OCR failed: {e}")

    return ""


async def _mistral_ocr_image(png_bytes: bytes) -> str:
    """Use Mistral OCR on a PNG image (run sync SDK in thread pool)."""
    from mistralai.client import Mistral

    client = Mistral(api_key=MISTRAL_API_KEY)
    b64 = base64.b64encode(png_bytes).decode()

    def _run_sync():
        return client.ocr.process(
            model=MISTRAL_OCR_MODEL,
            document={
                "type": "image_url",
                "image_url": f"data:image/png;base64,{b64}",
            },
        )

    result = await asyncio.get_running_loop().run_in_executor(None, _run_sync)

    # Extract text from all pages
    text_parts = []
    if hasattr(result, "pages"):
        for page in result.pages:
            if hasattr(page, "markdown"):
                text_parts.append(page.markdown)
    return "\n".join(text_parts)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _build_text_classify_prompt(ocr_text: str, existing_topics: List[Dict[str, str]], all_subjects: set | None = None) -> str:
    subjects_list = ", ".join(sorted(all_subjects)) if all_subjects else ", ".join(DEFAULT_SUBJECTS)
    topics_ctx = ""
    if existing_topics:
        by_subj: Dict[str, List[str]] = defaultdict(list)
        for t in existing_topics:
            by_subj[t["subject"]].append(t["topic"])
        topics_ctx = "\nExisting topics (REUSE if similar):\n"
        for s, ts in by_subj.items():
            topics_ctx += f"- {s}: {', '.join(ts[:5])}\n"

    return f"""Classify this handwritten note content.
Subjects: {subjects_list}
{topics_ctx}
Content: {ocr_text[:1000]}

Respond JSON only: {{"subject": "...", "topic": "2-4 word topic name", "confidence": 0.0-1.0}}
Use "Unorganised" if unclear."""


async def _text_classify(
    ocr_text: str,
    existing_topics: List[Dict[str, str]],
    all_subjects: set | None = None,
) -> Dict[str, Any]:
    """Classify text-heavy page via Groq (primary) or OpenAI (fallback)."""
    prompt = _build_text_classify_prompt(ocr_text, existing_topics, all_subjects)

    # Try Groq first
    if GROQ_API_KEY:
        try:
            client = AsyncOpenAI(
                api_key=GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
            )
            resp = await client.chat.completions.create(
                model=GROQ_CLASSIFY_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
            )
            return _parse_classification_json(resp.choices[0].message.content or "")
        except Exception as e:
            logger.warning(f"Groq classification failed: {e}")

    # Fallback: OpenAI
    if OPENAI_API_KEY:
        try:
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
            )
            return _parse_classification_json(resp.choices[0].message.content or "")
        except Exception as e:
            logger.warning(f"OpenAI text classification failed: {e}")

    return {"subject": "Unorganised", "topic": "General", "confidence": 0.3}


async def _vision_classify(
    png_bytes: bytes,
    existing_topics: List[Dict[str, str]],
    openai_client: Optional[AsyncOpenAI] = None,
    all_subjects: set | None = None,
) -> Dict[str, Any]:
    """Classify diagram/equation-heavy page via GPT-4o-mini Vision."""
    if not OPENAI_API_KEY and not openai_client:
        return {"subject": "Unorganised", "topic": "General", "confidence": 0.3}

    subjects_list = ", ".join(sorted(all_subjects)) if all_subjects else ", ".join(DEFAULT_SUBJECTS)
    b64 = base64.b64encode(png_bytes).decode()
    data_url = f"data:image/png;base64,{b64}"

    topics_ctx = ""
    if existing_topics:
        by_subj: Dict[str, List[str]] = defaultdict(list)
        for t in existing_topics:
            by_subj[t["subject"]].append(t["topic"])
        topics_ctx = "\nExisting topics (REUSE if similar):\n"
        for s, ts in by_subj.items():
            topics_ctx += f"- {s}: {', '.join(ts[:5])}\n"

    prompt = f"""Classify this handwritten note page (may contain diagrams, equations, or drawings).
Subjects: {subjects_list}
{topics_ctx}
Respond JSON only: {{"subject": "...", "topic": "2-4 word topic name", "confidence": 0.0-1.0}}
Use "Unorganised" if unclear."""

    try:
        client = openai_client or AsyncOpenAI(api_key=OPENAI_API_KEY)
        resp = await client.chat.completions.create(
            model=VISION_CLASSIFY_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=200,
            temperature=0.1,
        )
        return _parse_classification_json(resp.choices[0].message.content or "")
    except Exception as e:
        logger.warning(f"Vision classification failed: {e}")
        return {"subject": "Unorganised", "topic": "General", "confidence": 0.3}


def _parse_classification_json(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        subject = data.get("subject", "Unorganised")
        topic = data.get("topic", "General")
        confidence = float(data.get("confidence", 0.5))
        return {"subject": subject, "topic": topic[:50], "confidence": min(max(confidence, 0), 1)}
    except (json.JSONDecodeError, ValueError):
        logger.warning(f"Failed to parse classification JSON: {text[:200]}")
        return {"subject": "Unorganised", "topic": "General", "confidence": 0.3}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_existing_topics(
    tenant_db, user_id: str, copy_id: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Get existing subject+topic pairs for deduplication (copy-scoped)."""
    match_stage: Dict[str, Any] = {"user_id": user_id}
    if copy_id:
        match_stage["copy_id"] = copy_id
    pipeline = [
        {"$match": match_stage},
        {"$group": {"_id": {"subject": "$subject", "topic": "$topic"}}},
        {"$limit": 50},
    ]
    results = []
    async for doc in tenant_db["note_classifications"].aggregate(pipeline):
        results.append({
            "subject": doc["_id"]["subject"],
            "topic": doc["_id"]["topic"],
        })
    return results


async def _upload_thumbnail(
    png_bytes: bytes,
    user_id: str,
    pen_mac: str,
    book_type: str,
    page_number: int,
    copy_id: Optional[str] = None,
) -> Optional[str]:
    """Upload PNG thumbnail to S3/local storage."""
    try:
        from utils.s3_storage import upload_file

        safe_mac = pen_mac.replace(":", "")
        cid_segment = f"_{copy_id}" if copy_id else ""
        path = f"note_thumbnails/{user_id}/{safe_mac}_{book_type}_p{page_number}{cid_segment}.png"
        success, storage_path = await upload_file(
            png_bytes, path, content_type="image/png"
        )
        if success:
            return storage_path
        logger.warning(f"Thumbnail upload failed for page {page_number}")
    except Exception as e:
        logger.warning(f"Thumbnail upload error: {e}")
    return None


async def _save_classification(
    tenant_db,
    page_key: Dict[str, Any],
    subject: str,
    topic: str,
    confidence: float,
    ocr_text: str,
    thumbnail_url: Optional[str],
    stroke_count: int,
    existing: Optional[Dict[str, Any]] = None,
    *,
    session_id: Optional[str] = None,
    first_activity: Optional[Any] = None,
    last_activity: Optional[Any] = None,
):
    """Upsert into note_classifications."""
    now = datetime.now(timezone.utc)
    update_doc: Dict[str, Any] = {
        "$set": {
            "subject": subject,
            "topic": topic,
            "confidence": confidence,
            "classification_source": "auto",
            "stroke_count_at_classification": stroke_count,
            "ocr_text": ocr_text,
            "updated_at": now,
        },
        "$setOnInsert": {
            "created_at": now,
            "original_subject": None,
            "original_topic": None,
        },
        "$unset": {
            "pending_ai_previous_source": "",
        },
    }

    if thumbnail_url:
        update_doc["$set"]["thumbnail_url"] = thumbnail_url
    if session_id is not None:
        update_doc["$set"]["session_id"] = session_id
    if first_activity is not None:
        update_doc["$set"]["first_activity"] = first_activity
    if last_activity is not None:
        update_doc["$set"]["last_activity"] = last_activity

    try:
        await tenant_db["note_classifications"].update_one(
            page_key, update_doc, upsert=True
        )
    except DuplicateKeyError as exc:
        legacy_key = {k: v for k, v in page_key.items() if k != "copy_id"}
        legacy_doc = await tenant_db["note_classifications"].find_one(
            legacy_key,
            {"_id": 1, "copy_id": 1},
            sort=[("updated_at", -1), ("created_at", -1), ("_id", -1)],
        )
        if not legacy_doc:
            raise
        fallback_update = dict(update_doc)
        fallback_set = dict(fallback_update.get("$set", {}))
        if page_key.get("copy_id") and not legacy_doc.get("copy_id"):
            fallback_set["copy_id"] = page_key["copy_id"]
        fallback_update["$set"] = fallback_set
        await tenant_db["note_classifications"].update_one(
            {"_id": legacy_doc["_id"]},
            fallback_update,
            upsert=False,
        )
        logger.warning(
            "Recovered note classification write via legacy identity fallback for user=%s pen=%s book=%s page=%s copy=%s: %s",
            page_key.get("user_id"),
            page_key.get("pen_mac"),
            page_key.get("book_type"),
            page_key.get("page_number"),
            page_key.get("copy_id"),
            exc,
        )


async def _find_existing_note_classification(
    tenant_db,
    user_id: str,
    pen_mac: str,
    book_type: str,
    page_number: int,
    *,
    copy_id: Optional[str] = None,
):
    exact_key: Dict[str, Any] = {
        "user_id": user_id,
        "book_type": book_type or "A5",
        "page_number": page_number,
    }
    if copy_id:
        exact_key["copy_id"] = copy_id
    doc = await tenant_db["note_classifications"].find_one(
        exact_key,
        sort=[("updated_at", -1), ("created_at", -1), ("_id", -1)],
    )
    if doc is not None or not copy_id:
        return doc
    legacy_key = {
        "user_id": user_id,
        "book_type": book_type or "A5",
        "page_number": page_number,
    }
    return await tenant_db["note_classifications"].find_one(
        legacy_key,
        sort=[("updated_at", -1), ("created_at", -1), ("_id", -1)],
    )


async def clear_pending_ai_state(
    tenant_db,
    user_id: str,
    pen_mac: str,
    book_type: str,
    page_number: int,
    *,
    copy_id: Optional[str] = None,
) -> bool:
    """Clear a pending AI placeholder or restore the prior classification source."""
    page_key: Dict[str, Any] = {
        "user_id": user_id,
        "book_type": book_type or "A5",
        "page_number": page_number,
    }
    if copy_id:
        page_key["copy_id"] = copy_id

    existing = await tenant_db["note_classifications"].find_one(
        page_key,
        sort=[("updated_at", -1), ("created_at", -1), ("_id", -1)],
    )
    if not existing or existing.get("classification_source") != "pending_ai":
        return False

    now = datetime.now(timezone.utc)
    previous_source = existing.get("pending_ai_previous_source")
    if previous_source:
        await tenant_db["note_classifications"].update_one(
            {"_id": existing["_id"]},
            {
                "$set": {
                    "classification_source": previous_source,
                    "updated_at": now,
                },
                "$unset": {
                    "pending_ai_previous_source": "",
                },
            },
        )
        return True

    is_placeholder = (
        (existing.get("subject") in (None, "", "Unorganised"))
        and (existing.get("topic") in (None, ""))
        and not bool(existing.get("is_favorite"))
        and not bool(existing.get("is_archived"))
        and not existing.get("thumbnail_url")
        and not (existing.get("ocr_text") or "").strip()
        and (existing.get("stroke_count_at_classification") in (None, 0))
        and existing.get("confidence") in (None, 0)
    )

    if is_placeholder:
        await tenant_db["note_classifications"].delete_one({"_id": existing["_id"]})
        return True

    await tenant_db["note_classifications"].update_one(
        {"_id": existing["_id"]},
        {
            "$set": {
                "classification_source": "system",
                "updated_at": now,
            },
            "$unset": {
                "pending_ai_previous_source": "",
            },
        },
    )
    return True
