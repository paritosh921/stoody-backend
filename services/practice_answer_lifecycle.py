"""Durable Practice attempts and immutable, user-scoped submission evidence.

Canvas strokes are the draft ledger. Snapshots freeze verified evidence so
evaluation retries never re-read a page that the student is still writing on.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from bson import ObjectId
from fastapi import HTTPException
from pymongo import ReturnDocument

from core.user_identity import canonical_canvas_user_id
from services.practice_stroke_evidence import _canvas_collection, resolve_practice_stroke_evidence


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


async def collections(current_user, db):
    pages = await _canvas_collection(current_user, db)
    return pages.database


async def resolve_attempt(current_user, db, document_id, preferred_session_id, has_pending_writing=False):
    database = await collections(current_user, db)
    owner = canonical_canvas_user_id(current_user)
    key = digest([owner, document_id])
    # A device carrying unacknowledged ink must resume its own attempt rather
    # than silently switching that ink to another device's active draft.
    if has_pending_writing and preferred_session_id:
        await database["practice_drafts"].update_one(
            {"_id": key}, {"$set": {"session_id": preferred_session_id},
                           "$setOnInsert": {"user_id": owner, "document_id": document_id,
                                            "created_at": datetime.now(timezone.utc)}}, upsert=True,
        )
    record = await database["practice_drafts"].find_one_and_update(
        {"_id": key}, {"$setOnInsert": {
            "user_id": owner, "document_id": document_id,
            "session_id": preferred_session_id or f"practice_{uuid4()}",
            "created_at": datetime.now(timezone.utc),
        }}, upsert=True, return_document=ReturnDocument.AFTER,
    )
    return record["session_id"]


async def prepare_answer(current_user, db, refs):
    resolved = await resolve_practice_stroke_evidence(
        current_user=current_user, db=db, refs=refs,
        payload_question_id=refs.questionId, discover_owned=True, allow_empty=refs.allowEmptyAnswer,
    )
    owner = canonical_canvas_user_id(current_user)
    snapshot_id = digest([owner, resolved.receipt])
    record = {
        "user_id": owner, "question_id": refs.questionId,
        "session_id": refs.practiceSessionId, "copy_id": refs.copyId,
        "images": resolved.data_urls, "receipt": resolved.receipt,
        "page_refs": resolved.page_refs, "created_at": datetime.now(timezone.utc),
    }
    # MongoDB's document limit is 16 MB; fail explicitly before an oversized
    # snapshot can become an apparently successful preparation.
    if len(json.dumps(record, default=str).encode()) > 12_000_000:
        raise HTTPException(413, "Answer snapshot is too large. Split this answer before submitting.")
    database = await collections(current_user, db)
    await database["practice_answer_snapshots"].update_one(
        {"_id": snapshot_id}, {"$setOnInsert": record}, upsert=True,
    )
    return {"snapshotId": snapshot_id, "questionPageRefs": resolved.page_refs}


async def load_snapshot(current_user, db, snapshot_id, question_id, session_id):
    database = await collections(current_user, db)
    snapshot = await database["practice_answer_snapshots"].find_one({
        "_id": snapshot_id, "user_id": canonical_canvas_user_id(current_user),
        "question_id": question_id, "session_id": session_id,
    })
    if not snapshot:
        raise HTTPException(409, "Prepared answer is unavailable or belongs to another question. Prepare again.")
    return snapshot


async def begin_evaluation(current_user, db, payload):
    database = await collections(current_user, db)
    # Time spent is UI telemetry, not a different answer. Attachments/text are
    # included so adding a file cannot accidentally reuse an older evaluation.
    answer = payload.model_dump(exclude={"timeSpent", "hintsUsed", "questionPageRefs", "canvasPages"})
    key = digest([canonical_canvas_user_id(current_user), answer])
    coll = database["practice_answer_evaluations"]
    await coll.update_one({"_id": key}, {"$setOnInsert": {"state": "ready"}}, upsert=True)
    previous = await coll.find_one({"_id": key})
    if previous.get("state") == "complete":
        await persist_evaluation_history(coll, key, previous)
        return (coll, key, None, previous["result"])
    token = uuid4().hex
    now = datetime.now(timezone.utc)
    claimed = await coll.find_one_and_update(
        {"_id": key, "$or": [{"state": {"$in": ["ready", "failed"]}}, {"lease_until": {"$lt": now}}]},
        {"$set": {"state": "running", "lease": token, "lease_until": now + timedelta(minutes=15)}},
        return_document=ReturnDocument.AFTER,
    )
    if not claimed:
        raise HTTPException(409, "This answer is already being analyzed. Retry shortly to retrieve its result.")
    return (coll, key, token, None)


async def persist_evaluation_history(coll, key, record):
    if record.get("attempt"):
        await coll.database["practice_attempts"].update_one(
            {"_id": ObjectId(key[:24]), "submission_key": key}, {"$setOnInsert": record["attempt"]}, upsert=True,
        )


async def finish_evaluation(claim, result=None, attempt=None):
    if not claim or not claim[2]:
        return
    coll, key, token, _ = claim
    values = {"state": "complete", "result": result, "attempt": attempt} if result else {"state": "failed"}
    updated = await coll.update_one({"_id": key, "lease": token, "state": "running"}, {"$set": values, "$unset": {"lease_until": ""}})
    if result and not updated.matched_count:
        raise HTTPException(409, "Analysis was resumed by another request. Retry to retrieve the committed result.")
    if result and updated.matched_count:
        # The result and its history payload are committed together. If this
        # projection fails, the next retry repairs it without another LLM call.
        await persist_evaluation_history(coll, key, values)
