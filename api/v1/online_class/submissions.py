import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

LOCKS_COLLECTION = "online_class_locks"
SUBMISSIONS_COLLECTION = "online_class_submissions"


async def create_or_update_submission(
    db,
    meeting_id: str,
    lock_id: str,
    student_id: str,
    canvas_pages: List[str],
    question_page_refs: Optional[Dict[str, Any]],
    answer_text: Optional[str],
    time_spent: Optional[float],
    client_submitted_at: Optional[datetime],
) -> Dict[str, Any]:
    existing = await db.mongo_find_one(
        SUBMISSIONS_COLLECTION,
        {"meeting_id": meeting_id, "lock_id": lock_id, "student_id": student_id},
    )

    now = datetime.utcnow()

    if existing:
        update_fields: Dict[str, Any] = {
            "canvas_pages": canvas_pages,
            "updated_at": now,
        }
        if question_page_refs is not None:
            update_fields["question_page_refs"] = question_page_refs
        if answer_text is not None:
            update_fields["answer_text"] = answer_text
        if time_spent is not None:
            update_fields["time_spent"] = time_spent
        if client_submitted_at is not None:
            update_fields["client_submitted_at"] = client_submitted_at

        await db.mongo_update_one(
            SUBMISSIONS_COLLECTION,
            {"submission_id": existing["submission_id"]},
            {"$set": update_fields},
        )
        existing.update(update_fields)
        logger.info("Updated submission %s", existing["submission_id"])
        return existing

    submission_id = f"sub-{uuid.uuid4().hex[:12]}"
    doc = {
        "submission_id": submission_id,
        "meeting_id": meeting_id,
        "lock_id": lock_id,
        "student_id": student_id,
        "canvas_pages": canvas_pages,
        "question_page_refs": question_page_refs,
        "answer_text": answer_text,
        "time_spent": time_spent,
        "client_submitted_at": client_submitted_at,
        "analysis_status": "pending",
        "score": None,
        "is_correct": None,
        "what_went_wrong": None,
        "correct_solution": None,
        "created_at": now,
        "updated_at": now,
    }
    await db.mongo_insert_one(SUBMISSIONS_COLLECTION, doc)
    logger.info("Created submission %s for lock %s student %s", submission_id, lock_id, student_id)
    return doc


async def get_submissions_for_lock(
    db,
    meeting_id: str,
    lock_id: str,
) -> List[Dict[str, Any]]:
    cursor = await db.mongo_find(
        SUBMISSIONS_COLLECTION,
        {"meeting_id": meeting_id, "lock_id": lock_id},
    )
    return list(cursor)
