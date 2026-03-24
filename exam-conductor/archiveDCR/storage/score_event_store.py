"""MongoDB event-sourced score store.

Collections:
    exampen_score_events  — append-only event log (NO update/delete)
    exampen_score_current — materialised current scores (upserted on each event)

Every query includes tenant_id filter (replacing PostgreSQL RLS).
Uses MongoDB transactions for atomicity of the dual-write.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument

logger = logging.getLogger(__name__)

EVENTS_COLLECTION = "exampen_score_events"
CURRENT_COLLECTION = "exampen_score_current"


class ScoreEventStore:
    """Async MongoDB event-sourced store for exam scores."""

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db
        self._events = db[EVENTS_COLLECTION]
        self._current = db[CURRENT_COLLECTION]

    async def append_event(
        self, tenant_id: str, event: dict[str, Any]
    ) -> dict:
        """Append a score event and update the current projection.

        Uses a MongoDB transaction to guarantee both writes succeed
        or neither does.

        Returns the inserted event document.
        """
        now = datetime.now(timezone.utc)
        event_doc = {
            "_id": uuid4().hex,
            "tenant_id": tenant_id,
            "created_at": now,
            **event,
        }

        # Build the current-score projection update
        current_update: dict[str, Any] = {
            "$set": {
                "tenant_id": tenant_id,
                "updated_at": now,
            },
            "$setOnInsert": {
                "_id": uuid4().hex,
                "created_at": now,
            },
        }
        # Merge score-relevant fields into $set
        for key in (
            "exam_id", "student_id", "question_id", "score",
            "max_marks", "state", "evaluator_id",
        ):
            if key in event:
                current_update["$set"][key] = event[key]

        current_filter = {
            "exam_id": event.get("exam_id"),
            "student_id": event.get("student_id"),
            "question_id": event.get("question_id"),
            "tenant_id": tenant_id,
        }

        # Attempt transactional dual-write
        async with await self._db.client.start_session() as session:
            async with session.start_transaction():
                await self._events.insert_one(
                    event_doc, session=session
                )
                await self._current.find_one_and_update(
                    current_filter,
                    current_update,
                    upsert=True,
                    return_document=ReturnDocument.AFTER,
                    session=session,
                )

        return event_doc

    async def get_current_scores(
        self,
        exam_id: str,
        student_id: str,
        tenant_id: str,
    ) -> list[dict]:
        """Get the current scores for a student in an exam.

        Returns one document per question.
        """
        cursor = self._current.find(
            {
                "exam_id": exam_id,
                "student_id": student_id,
                "tenant_id": tenant_id,
            }
        )
        return await cursor.to_list(length=500)

    async def get_exam_overview(
        self, exam_id: str, tenant_id: str
    ) -> list[dict[str, Any]]:
        """Aggregated overview: total score per student.

        Returns a list of ``{"student_id": str, "total_score": float,
        "question_count": int}`` dicts.
        """
        pipeline = [
            {
                "$match": {
                    "exam_id": exam_id,
                    "tenant_id": tenant_id,
                }
            },
            {
                "$group": {
                    "_id": "$student_id",
                    "total_score": {"$sum": "$score"},
                    "question_count": {"$sum": 1},
                }
            },
            {"$sort": {"total_score": -1}},
        ]
        results = await self._current.aggregate(pipeline).to_list(
            length=1000
        )
        return [
            {
                "student_id": row["_id"],
                "total_score": row["total_score"],
                "question_count": row["question_count"],
            }
            for row in results
        ]

    async def get_event_history(
        self,
        exam_id: str,
        student_id: str,
        tenant_id: str,
    ) -> list[dict]:
        """Get the full event history for a student in an exam.

        Returns events sorted by created_at ascending.
        """
        cursor = self._events.find(
            {
                "exam_id": exam_id,
                "student_id": student_id,
                "tenant_id": tenant_id,
            }
        ).sort("created_at", 1)
        return await cursor.to_list(length=5000)
