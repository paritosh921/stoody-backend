import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

LOCKS_COLLECTION = "online_class_locks"
SUBMISSIONS_COLLECTION = "online_class_submissions"


async def create_lock(
    db,
    meeting_id: str,
    tutor_id: str,
    question_text: Optional[str],
    question_image_id: Optional[str],
    question_bbox: Optional[Dict[str, Any]],
    duration_seconds: int,
) -> Dict[str, Any]:
    existing = await db.mongo_find_one(
        LOCKS_COLLECTION,
        {"meeting_id": meeting_id, "status": "active"},
    )
    if existing:
        raise ValueError("An active lock already exists for this meeting")

    lock_id = f"lck-{uuid.uuid4().hex[:12]}"
    now = datetime.utcnow()
    doc = {
        "lock_id": lock_id,
        "meeting_id": meeting_id,
        "tutor_id": tutor_id,
        "question_text": question_text,
        "question_image_id": question_image_id,
        "question_bbox": question_bbox,
        "duration_seconds": duration_seconds,
        "start_ts": now,
        "end_ts": None,
        "status": "active",
        "created_at": now,
        "ended_at": None,
    }
    await db.mongo_insert_one(LOCKS_COLLECTION, doc)
    logger.info("Created lock %s for meeting %s", lock_id, meeting_id)
    return doc


async def get_current_lock(db, meeting_id: str) -> Optional[Dict[str, Any]]:
    return await db.mongo_find_one(
        LOCKS_COLLECTION,
        {"meeting_id": meeting_id, "status": "active"},
    )


async def get_lock_by_id(db, meeting_id: str, lock_id: str) -> Optional[Dict[str, Any]]:
    return await db.mongo_find_one(
        LOCKS_COLLECTION,
        {"meeting_id": meeting_id, "lock_id": lock_id},
    )


async def end_lock(db, meeting_id: str, lock_id: str) -> Dict[str, Any]:
    lock = await get_lock_by_id(db, meeting_id, lock_id)
    if not lock:
        raise ValueError("Lock not found")
    if lock.get("status") == "ended":
        return lock
    now = datetime.utcnow()
    await db.mongo_update_one(
        LOCKS_COLLECTION,
        {"lock_id": lock_id, "meeting_id": meeting_id},
        {"$set": {"status": "ended", "end_ts": now, "ended_at": now}},
    )
    lock["status"] = "ended"
    lock["end_ts"] = now
    lock["ended_at"] = now
    logger.info("Ended lock %s for meeting %s", lock_id, meeting_id)
    return lock
