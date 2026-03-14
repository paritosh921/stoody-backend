"""
Note Organisation API for Stoody

Provides endpoints for browsing AI-classified notes by subject/topic,
viewing page thumbnails, reclassifying pages, and generating flashcards
and practice questions from handwritten notes.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.v1.auth_async import get_current_user, get_database
from core.database import DatabaseManager
from core.user_identity import canvas_user_id_variants
from utils.s3_storage import get_public_url

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Note Organisation"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ReclassifyRequest(BaseModel):
    subject: str
    topic: str


class BatchReclassifyRequest(BaseModel):
    classification_ids: List[str]
    subject: str
    topic: str


class ArchiveRequest(BaseModel):
    archived: bool = True


class BatchArchiveRequest(BaseModel):
    classification_ids: List[str]
    archived: bool = True


class GenerateRequest(BaseModel):
    force: bool = False  # Force regeneration even if cached


class RegenerateRequest(BaseModel):
    content_type: str = Field(..., pattern="^(flashcards|practice)$")


# ---------------------------------------------------------------------------
# GET /subjects
# ---------------------------------------------------------------------------

@router.get("/subjects")
async def get_subjects(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Get all subjects with topic and page counts."""
    tenant_db = await db.get_tenant_db(current_user.get("db_name"))
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(current_user)
    uid_match = _user_id_match(current_user)

    pipeline = [
        {"$match": {"user_id": uid_match, "is_archived": {"$ne": True}}},
        {"$group": {
            "_id": "$subject",
            "topic_count": {"$addToSet": "$topic"},
            "page_count": {"$sum": 1},
            "latest_update": {"$max": "$updated_at"},
        }},
        {"$project": {
            "subject": "$_id",
            "topic_count": {"$size": "$topic_count"},
            "page_count": 1,
            "latest_update": 1,
            "_id": 0,
        }},
        {"$sort": {"subject": 1}},
    ]

    results = await tenant_db["note_classifications"].aggregate(pipeline).to_list(20)
    return {"success": True, "subjects": results}


# ---------------------------------------------------------------------------
# GET /subjects/{subject}/topics
# ---------------------------------------------------------------------------

@router.get("/subjects/{subject}/topics")
async def get_topics(
    subject: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Get topics within a subject, with page counts and staleness info."""
    tenant_db = await db.get_tenant_db(current_user.get("db_name"))
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(current_user)
    uid_match = _user_id_match(current_user)

    pipeline = [
        {"$match": {"user_id": uid_match, "subject": subject, "is_archived": {"$ne": True}}},
        {"$group": {
            "_id": "$topic",
            "page_count": {"$sum": 1},
            "latest_update": {"$max": "$updated_at"},
            "pages": {"$push": {
                "pen_mac": "$pen_mac",
                "book_type": "$book_type",
                "page_number": "$page_number",
                "thumbnail_url": "$thumbnail_url",
            }},
        }},
        {"$sort": {"latest_update": -1}},
    ]

    results = await tenant_db["note_classifications"].aggregate(pipeline).to_list(100)

    topics = []
    for r in results:
        topic_name = r["_id"]
        page_keys = [
            {"pen_mac": p["pen_mac"], "book_type": p["book_type"], "page_number": p["page_number"]}
            for p in r.get("pages", [])
        ]

        # Check staleness against generated content
        new_pages_since_gen = 0
        for collection in ["note_flashcards", "note_practice_sets"]:
            existing = await tenant_db[collection].find_one(
                {"user_id": uid_match, "subject": subject, "topic": topic_name},
                {"source_page_count": 1},
            )
            if existing:
                new_pages_since_gen = max(
                    new_pages_since_gen,
                    r["page_count"] - existing.get("source_page_count", 0),
                )

        thumbnails = [
            get_public_url(p["thumbnail_url"]) for p in r.get("pages", [])[:5]
            if p.get("thumbnail_url")
        ]

        topics.append({
            "topic": topic_name,
            "page_count": r["page_count"],
            "latest_update": r["latest_update"],
            "new_pages_since_gen": new_pages_since_gen,
            "thumbnails": thumbnails,
        })

    return {"success": True, "subject": subject, "topics": topics}


# ---------------------------------------------------------------------------
# GET /subjects/{subject}/pages
# ---------------------------------------------------------------------------

@router.get("/subjects/{subject}/pages")
async def get_subject_pages(
    subject: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Get all pages for a subject (regardless of topic), sorted by updated_at DESC."""
    tenant_db = await db.get_tenant_db(current_user.get("db_name"))
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(current_user)
    uid_match = _user_id_match(current_user)

    cursor = tenant_db["note_classifications"].find(
        {"user_id": uid_match, "subject": subject, "is_archived": {"$ne": True}},
    ).sort("updated_at", -1)

    pages = []
    async for doc in cursor:
        p = _doc_to_page(doc)
        p["topic"] = doc.get("topic")
        pages.append(p)

    # Batch-enrich pages missing session info from strokes
    await _enrich_pages_with_session_info(tenant_db, user_id, pages, canvas_user_id_variants(current_user))

    return {"success": True, "subject": subject, "pages": pages}


# ---------------------------------------------------------------------------
# GET /subjects/{subject}/topics/{topic}/pages
# ---------------------------------------------------------------------------

@router.get("/subjects/{subject}/topics/{topic}/pages")
async def get_topic_pages(
    subject: str,
    topic: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Get all pages in a topic with thumbnail URLs."""
    tenant_db = await db.get_tenant_db(current_user.get("db_name"))
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(current_user)
    uid_match = _user_id_match(current_user)

    cursor = tenant_db["note_classifications"].find(
        {"user_id": uid_match, "subject": subject, "topic": topic, "is_archived": {"$ne": True}},
    ).sort("page_number", 1)

    pages = []
    async for doc in cursor:
        pages.append(_doc_to_page(doc))

    # Batch-enrich pages missing session info from strokes
    await _enrich_pages_with_session_info(tenant_db, user_id, pages, canvas_user_id_variants(current_user))

    return {"success": True, "subject": subject, "topic": topic, "pages": pages}


# ---------------------------------------------------------------------------
# GET /pages/{pen_mac}/{page_number}/thumbnail
# ---------------------------------------------------------------------------

@router.get("/pages/{pen_mac}/{page_number}/thumbnail")
async def get_page_thumbnail(
    pen_mac: str,
    page_number: int,
    book_type: str = Query("A5"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Get thumbnail URL for a specific page."""
    tenant_db = await db.get_tenant_db(current_user.get("db_name"))
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(current_user)
    uid_match = _user_id_match(current_user)

    doc = await tenant_db["note_classifications"].find_one(
        {
            "user_id": uid_match,
            "pen_mac": pen_mac.upper(),
            "book_type": book_type,
            "page_number": page_number,
        },
        {"thumbnail_url": 1},
    )

    if not doc or not doc.get("thumbnail_url"):
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    url = get_public_url(doc["thumbnail_url"])
    return {"success": True, "thumbnail_url": url}


# ---------------------------------------------------------------------------
# PATCH /pages/{classification_id}/reclassify
# ---------------------------------------------------------------------------

@router.patch("/pages/{classification_id}/reclassify")
async def reclassify_page(
    classification_id: str,
    body: ReclassifyRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Manually move a page to a different subject/topic."""
    tenant_db = await db.get_tenant_db(current_user.get("db_name"))
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(current_user)
    variants = canvas_user_id_variants(current_user)

    if not body.subject.strip() or not body.topic.strip():
        raise HTTPException(status_code=400, detail="Subject and topic must not be empty")

    doc = await tenant_db["note_classifications"].find_one({"_id": ObjectId(classification_id)})
    if not doc or str(doc.get("user_id", "")) not in variants:
        raise HTTPException(status_code=404, detail="Classification not found")

    now = datetime.utcnow()
    update: Dict[str, Any] = {
        "$set": {
            "subject": body.subject,
            "topic": body.topic,
            "classification_source": "manual",
            "updated_at": now,
        },
    }

    # Preserve original values on first manual override
    if doc.get("classification_source") != "manual":
        update["$set"]["original_subject"] = doc.get("subject")
        update["$set"]["original_topic"] = doc.get("topic")

    await tenant_db["note_classifications"].update_one(
        {"_id": ObjectId(classification_id)}, update
    )
    return {"success": True, "message": "Page reclassified"}


# ---------------------------------------------------------------------------
# PATCH /pages/batch-reclassify
# ---------------------------------------------------------------------------

@router.patch("/pages/batch-reclassify")
async def batch_reclassify(
    body: BatchReclassifyRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Move multiple pages to a different subject/topic at once."""
    tenant_db = await db.get_tenant_db(current_user.get("db_name"))
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(current_user)
    uid_match = _user_id_match(current_user)

    if not body.subject.strip() or not body.topic.strip():
        raise HTTPException(status_code=400, detail="Subject and topic must not be empty")

    ids = [ObjectId(cid) for cid in body.classification_ids]
    now = datetime.utcnow()

    result = await tenant_db["note_classifications"].update_many(
        {"_id": {"$in": ids}, "user_id": uid_match},
        {"$set": {
            "subject": body.subject,
            "topic": body.topic,
            "classification_source": "manual",
            "updated_at": now,
        }},
    )

    return {
        "success": True,
        "modified_count": result.modified_count,
    }


# ---------------------------------------------------------------------------
# POST /topics/{subject}/{topic}/generate-flashcards
# ---------------------------------------------------------------------------

@router.post("/topics/{subject}/{topic}/generate-flashcards")
async def generate_flashcards(
    subject: str,
    topic: str,
    body: GenerateRequest = Body(default=GenerateRequest()),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Generate flashcards for a topic from OCR text of classified pages."""
    tenant_db = await db.get_tenant_db(current_user.get("db_name"))
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(current_user)
    uid_match = _user_id_match(current_user)

    # Check for cached flashcards
    if not body.force:
        existing = await tenant_db["note_flashcards"].find_one(
            {"user_id": uid_match, "subject": subject, "topic": topic}
        )
        if existing:
            existing["_id"] = str(existing["_id"])
            return {"success": True, "cached": True, "flashcards": existing}

    # Get OCR text from classified pages
    pages = await tenant_db["note_classifications"].find(
        {"user_id": uid_match, "subject": subject, "topic": topic},
        {"ocr_text": 1, "pen_mac": 1, "book_type": 1, "page_number": 1},
    ).to_list(50)

    if not pages:
        raise HTTPException(status_code=404, detail="No pages found for this topic")

    texts = [p.get("ocr_text", "") for p in pages if p.get("ocr_text")]
    if not texts:
        raise HTTPException(status_code=400, detail="No OCR text available for these pages")

    from services.note_content_generator import generate_flashcards as gen_fc
    result = await gen_fc(user_id, subject, topic, texts, pages, tenant_db)

    return {"success": True, "cached": False, "flashcards": result}


# ---------------------------------------------------------------------------
# POST /topics/{subject}/{topic}/generate-practice
# ---------------------------------------------------------------------------

@router.post("/topics/{subject}/{topic}/generate-practice")
async def generate_practice(
    subject: str,
    topic: str,
    body: GenerateRequest = Body(default=GenerateRequest()),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Generate practice questions for a topic from OCR text of classified pages."""
    tenant_db = await db.get_tenant_db(current_user.get("db_name"))
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(current_user)
    uid_match = _user_id_match(current_user)

    # Check for cached practice set
    if not body.force:
        existing = await tenant_db["note_practice_sets"].find_one(
            {"user_id": uid_match, "subject": subject, "topic": topic}
        )
        if existing:
            existing["_id"] = str(existing["_id"])
            return {"success": True, "cached": True, "practice": existing}

    # Get OCR text from classified pages
    pages = await tenant_db["note_classifications"].find(
        {"user_id": uid_match, "subject": subject, "topic": topic},
        {"ocr_text": 1, "pen_mac": 1, "book_type": 1, "page_number": 1},
    ).to_list(50)

    if not pages:
        raise HTTPException(status_code=404, detail="No pages found for this topic")

    texts = [p.get("ocr_text", "") for p in pages if p.get("ocr_text")]
    if not texts:
        raise HTTPException(status_code=400, detail="No OCR text available for these pages")

    from services.note_content_generator import generate_practice as gen_pr
    result = await gen_pr(user_id, subject, topic, texts, pages, tenant_db)

    return {"success": True, "cached": False, "practice": result}


# ---------------------------------------------------------------------------
# GET /topics/{subject}/{topic}/flashcards
# ---------------------------------------------------------------------------

@router.get("/topics/{subject}/{topic}/flashcards")
async def get_flashcards(
    subject: str,
    topic: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Get cached flashcards for a topic."""
    tenant_db = await db.get_tenant_db(current_user.get("db_name"))
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(current_user)
    uid_match = _user_id_match(current_user)

    doc = await tenant_db["note_flashcards"].find_one(
        {"user_id": uid_match, "subject": subject, "topic": topic}
    )
    if not doc:
        return {"success": True, "flashcards": None}

    doc["_id"] = str(doc["_id"])
    return {"success": True, "flashcards": doc}


# ---------------------------------------------------------------------------
# GET /topics/{subject}/{topic}/practice
# ---------------------------------------------------------------------------

@router.get("/topics/{subject}/{topic}/practice")
async def get_practice(
    subject: str,
    topic: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Get cached practice questions for a topic."""
    tenant_db = await db.get_tenant_db(current_user.get("db_name"))
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(current_user)
    uid_match = _user_id_match(current_user)

    doc = await tenant_db["note_practice_sets"].find_one(
        {"user_id": uid_match, "subject": subject, "topic": topic}
    )
    if not doc:
        return {"success": True, "practice": None}

    doc["_id"] = str(doc["_id"])
    return {"success": True, "practice": doc}


# ---------------------------------------------------------------------------
# POST /topics/{subject}/{topic}/regenerate
# ---------------------------------------------------------------------------

@router.post("/topics/{subject}/{topic}/regenerate")
async def regenerate_content(
    subject: str,
    topic: str,
    body: RegenerateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Delta regeneration — requires 5+ new pages since last generation."""
    tenant_db = await db.get_tenant_db(current_user.get("db_name"))
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(current_user)

    from services.note_content_generator import regenerate_content as regen
    result = await regen(user_id, subject, topic, body.content_type, tenant_db)
    return {"success": True, "result": result}


# ---------------------------------------------------------------------------
# GET /stats
# ---------------------------------------------------------------------------

@router.get("/stats")
async def get_stats(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Get classification statistics for the current user."""
    tenant_db = await db.get_tenant_db(current_user.get("db_name"))
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(current_user)
    uid_match = _user_id_match(current_user)

    total_pages = await tenant_db["note_classifications"].count_documents({"user_id": uid_match})

    pipeline = [
        {"$match": {"user_id": uid_match}},
        {"$group": {
            "_id": "$subject",
            "count": {"$sum": 1},
        }},
    ]
    by_subject = {}
    async for doc in tenant_db["note_classifications"].aggregate(pipeline):
        by_subject[doc["_id"]] = doc["count"]

    pending_queue = await tenant_db["classification_queue"].count_documents(
        {"user_id": uid_match, "status": "pending"}
    )
    failed_queue = await tenant_db["classification_queue"].count_documents(
        {"user_id": uid_match, "status": "failed"}
    )

    flashcard_sets = await tenant_db["note_flashcards"].count_documents({"user_id": uid_match})
    practice_sets = await tenant_db["note_practice_sets"].count_documents({"user_id": uid_match})

    return {
        "success": True,
        "stats": {
            "total_classified_pages": total_pages,
            "by_subject": by_subject,
            "pending_classifications": pending_queue,
            "failed_classifications": failed_queue,
            "flashcard_sets": flashcard_sets,
            "practice_sets": practice_sets,
        },
    }


@router.post("/backfill")
async def backfill_classifications(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Queue all existing stroke pages for classification (one-time backfill)."""
    db_name = current_user.get("db_name")
    is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"

    # Resolve the correct database (same logic as copies_async.py)
    if is_b2c:
        strokes_db = db.b2c_db
        if not db_name:
            try:
                from config_async import MONGODB_DB_STOODY
                db_name = MONGODB_DB_STOODY
            except ImportError:
                db_name = "STOODY-b2c"
    else:
        strokes_db = await db.get_tenant_db(db_name)

    if strokes_db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(current_user)
    uid_match = _user_id_match(current_user)

    # Build user_id match covering ObjectId variant for strokes collection
    user_identifiers = canvas_user_id_variants(current_user)
    try:
        if ObjectId.is_valid(user_id):
            user_identifiers.append(ObjectId(user_id))
    except Exception:
        pass

    # Find all distinct pages this user has strokes for
    pipeline = [
        {"$match": {"user_id": {"$in": user_identifiers}}},
        {"$group": {
            "_id": {
                "pen_mac": "$pen_mac",
                "book_type": "$book_type",
                "page_number": "$page_number",
            },
        }},
    ]
    distinct_pages = await strokes_db["strokes"].aggregate(pipeline).to_list(5000)

    # Check which are already classified (note_classifications uses canonical str user_id)
    classify_db = strokes_db if is_b2c else await db.get_tenant_db(db_name)
    already = set()
    async for doc in classify_db["note_classifications"].find(
        {"user_id": uid_match},
        {"pen_mac": 1, "book_type": 1, "page_number": 1},
    ):
        already.add((doc.get("pen_mac", ""), doc.get("book_type", ""), doc.get("page_number", 0)))

    from services.note_classification_service import queue_classification

    queued = 0
    for page in distinct_pages:
        key = page["_id"]
        pen_mac = key.get("pen_mac", "")
        book_type = key.get("book_type", "A5")
        page_number = key.get("page_number", 0)

        if (pen_mac.upper(), book_type, page_number) in already:
            continue

        await queue_classification(db, db_name, user_id, pen_mac, book_type, page_number)
        queued += 1

    return {
        "success": True,
        "total_stroke_pages": len(distinct_pages),
        "already_classified": len(already),
        "queued_for_classification": queued,
    }


# ---------------------------------------------------------------------------
# GET /pages/all
# ---------------------------------------------------------------------------

@router.get("/pages/all")
async def get_all_pages(
    favorites_only: bool = Query(False),
    archived_only: bool = Query(False),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Get all classified pages across all subjects, sorted by updated_at DESC."""
    tenant_db = await db.get_tenant_db(current_user.get("db_name"))
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(current_user)
    uid_match = _user_id_match(current_user)

    query: Dict[str, Any] = {"user_id": uid_match}
    query["is_archived"] = True if archived_only else {"$ne": True}
    if favorites_only:
        query["is_favorite"] = True

    cursor = tenant_db["note_classifications"].find(
        query,
        {
            "pen_mac": 1, "book_type": 1, "page_number": 1,
            "thumbnail_url": 1, "confidence": 1,
            "classification_source": 1, "ocr_text": 1,
            "subject": 1, "topic": 1, "is_favorite": 1,
            "is_archived": 1,
            "created_at": 1, "updated_at": 1,
            "session_id": 1, "first_activity": 1, "last_activity": 1,
        },
    ).sort("updated_at", -1)

    pages = []
    async for doc in cursor:
        pages.append(_doc_to_page(doc, include_subject_topic=True))

    await _enrich_pages_with_session_info(tenant_db, user_id, pages, canvas_user_id_variants(current_user))

    return {"success": True, "pages": pages}


# ---------------------------------------------------------------------------
# GET /search
# ---------------------------------------------------------------------------

@router.get("/search")
async def search_notes(
    q: str = Query(..., min_length=1, max_length=200),
    scope: str = Query("all", pattern="^(subject|topic|text|all)$"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Search notes by subject, topic, OCR text, or all fields."""
    tenant_db = await db.get_tenant_db(current_user.get("db_name"))
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(current_user)
    uid_match = _user_id_match(current_user)

    import re as _re
    safe_q = _re.escape(q)
    regex = {"$regex": safe_q, "$options": "i"}

    base_match: Dict[str, Any] = {"user_id": uid_match}
    base_match["is_archived"] = {"$ne": True}

    if scope == "subject":
        base_match["subject"] = regex
    elif scope == "topic":
        base_match["topic"] = regex
    elif scope == "text":
        base_match["ocr_text"] = regex
    else:  # all
        base_match["$or"] = [
            {"subject": regex},
            {"topic": regex},
            {"ocr_text": regex},
        ]

    cursor = tenant_db["note_classifications"].find(
        base_match,
        {
            "pen_mac": 1, "book_type": 1, "page_number": 1,
            "thumbnail_url": 1, "confidence": 1,
            "classification_source": 1, "ocr_text": 1,
            "subject": 1, "topic": 1,
            "is_archived": 1,
            "created_at": 1, "updated_at": 1,
            "session_id": 1, "first_activity": 1, "last_activity": 1,
        },
    ).sort("updated_at", -1).limit(200)

    pages = []
    async for doc in cursor:
        pages.append(_doc_to_page(doc, include_subject_topic=True))

    await _enrich_pages_with_session_info(tenant_db, user_id, pages, canvas_user_id_variants(current_user))

    return {"success": True, "query": q, "scope": scope, "pages": pages}


# ---------------------------------------------------------------------------
# DELETE /pages/{classification_id}
# ---------------------------------------------------------------------------

@router.patch("/pages/{classification_id}/favorite")
async def toggle_favorite(
    classification_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Toggle the favorite status of a note page."""
    tenant_db = await db.get_tenant_db(current_user.get("db_name"))
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(current_user)
    variants = canvas_user_id_variants(current_user)

    doc = await tenant_db["note_classifications"].find_one({"_id": ObjectId(classification_id)})
    if not doc or str(doc.get("user_id", "")) not in variants:
        raise HTTPException(status_code=404, detail="Classification not found")

    new_value = not doc.get("is_favorite", False)
    await tenant_db["note_classifications"].update_one(
        {"_id": ObjectId(classification_id)},
        {"$set": {"is_favorite": new_value, "updated_at": datetime.utcnow()}},
    )

    return {"success": True, "is_favorite": new_value}


@router.patch("/pages/{classification_id}/archive")
async def set_archive_status(
    classification_id: str,
    body: ArchiveRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Archive or unarchive a note page."""
    tenant_db = await db.get_tenant_db(current_user.get("db_name"))
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(current_user)
    variants = canvas_user_id_variants(current_user)

    doc = await tenant_db["note_classifications"].find_one({"_id": ObjectId(classification_id)})
    if not doc or str(doc.get("user_id", "")) not in variants:
        raise HTTPException(status_code=404, detail="Classification not found")

    await tenant_db["note_classifications"].update_one(
        {"_id": ObjectId(classification_id)},
        {"$set": {"is_archived": body.archived, "updated_at": datetime.utcnow()}},
    )

    return {"success": True, "is_archived": body.archived}


@router.patch("/pages/batch-archive")
async def batch_archive(
    body: BatchArchiveRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Archive or unarchive multiple pages at once."""
    tenant_db = await db.get_tenant_db(current_user.get("db_name"))
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(current_user)
    uid_match = _user_id_match(current_user)
    ids = [ObjectId(cid) for cid in body.classification_ids]

    result = await tenant_db["note_classifications"].update_many(
        {"_id": {"$in": ids}, "user_id": uid_match},
        {"$set": {"is_archived": body.archived, "updated_at": datetime.utcnow()}},
    )

    return {"success": True, "modified_count": result.modified_count, "is_archived": body.archived}


@router.delete("/pages/{classification_id}")
async def delete_page(
    classification_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Delete a note page: classification doc, S3 thumbnail, and strokes."""
    tenant_db = await db.get_tenant_db(current_user.get("db_name"))
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(current_user)
    variants = canvas_user_id_variants(current_user)

    doc = await tenant_db["note_classifications"].find_one({"_id": ObjectId(classification_id)})
    if not doc or str(doc.get("user_id", "")) not in variants:
        raise HTTPException(status_code=404, detail="Classification not found")

    pen_mac = doc.get("pen_mac", "")
    book_type = doc.get("book_type", "A5")
    page_number = doc.get("page_number")

    from utils.page_delete import delete_page_by_identity
    result = await delete_page_by_identity(
        tenant_db,
        pen_mac=pen_mac,
        book_type=book_type,
        page_number=page_number,
        user_id=user_id,
        user_id_variants=variants,
    )

    return {
        "success": True,
        "deleted_strokes": result["deleted_strokes"],
        "deleted_canvas_pages": result["deleted_canvas_pages"],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc_to_page(doc: Dict[str, Any], include_subject_topic: bool = False) -> Dict[str, Any]:
    """Convert a MongoDB note_classifications document to a page dict."""
    thumbnail_url = doc.get("thumbnail_url")
    page: Dict[str, Any] = {
        "id": str(doc["_id"]),
        "pen_mac": doc.get("pen_mac"),
        "book_type": doc.get("book_type"),
        "page_number": doc.get("page_number"),
        "thumbnail_url": get_public_url(thumbnail_url) if thumbnail_url else None,
        "confidence": doc.get("confidence"),
        "classification_source": doc.get("classification_source"),
        "ocr_text_preview": (doc.get("ocr_text") or "")[:200],
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "session_id": doc.get("session_id"),
        "first_activity": doc.get("first_activity"),
        "last_activity": doc.get("last_activity"),
        "is_favorite": doc.get("is_favorite", False),
        "is_archived": doc.get("is_archived", False),
    }
    if include_subject_topic:
        page["subject"] = doc.get("subject")
        page["topic"] = doc.get("topic")
    return page


def _get_user_id(current_user: Dict[str, Any]) -> str:
    uid = current_user.get("user_id") or current_user.get("_id")
    if isinstance(uid, ObjectId):
        return str(uid)
    return str(uid) if uid else ""


def _user_id_match(current_user: Dict[str, Any]) -> Dict[str, Any]:
    """Build a ``{"$in": [...]}`` clause covering all known user_id variants."""
    return {"$in": canvas_user_id_variants(current_user)}


async def _enrich_pages_with_session_info(
    tenant_db, user_id: str, pages: List[Dict[str, Any]],
    user_id_variants: Optional[List[str]] = None,
) -> None:
    """Batch-enrich pages missing session_id by looking up strokes."""
    pages_needing_session = [p for p in pages if not p.get("session_id")]
    if not pages_needing_session:
        return

    # Use full variant set if provided, otherwise fall back to ObjectId pair
    uid_list: list = list(user_id_variants) if user_id_variants else [user_id, str(user_id)]

    page_conditions = [
        {"pen_mac": p["pen_mac"], "book_type": p["book_type"], "page_number": p["page_number"]}
        for p in pages_needing_session
    ]
    pipeline = [
        {"$match": {"user_id": {"$in": uid_list}, "$or": page_conditions}},
        {"$sort": {"timestamp": -1}},
        {"$group": {
            "_id": {"pen_mac": "$pen_mac", "book_type": "$book_type", "page_number": "$page_number"},
            "session_id": {"$first": "$session_id"},
            "first_activity": {"$min": "$timestamp"},
            "last_activity": {"$max": "$timestamp"},
        }},
    ]
    session_map: Dict[tuple, Dict[str, Any]] = {}
    async for doc in tenant_db["strokes"].aggregate(pipeline):
        key = (doc["_id"]["pen_mac"], doc["_id"]["book_type"], doc["_id"]["page_number"])
        session_map[key] = doc

    for p in pages_needing_session:
        info = session_map.get((p["pen_mac"], p["book_type"], p["page_number"]))
        if info:
            p["session_id"] = info.get("session_id")
            p["first_activity"] = info.get("first_activity")
            p["last_activity"] = info.get("last_activity")
