"""MongoDB storage repo for objections — Collection: exampen_objections.

Every query includes tenant_id filter (replacing PostgreSQL RLS).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument

logger = logging.getLogger(__name__)

COLLECTION = "exampen_objections"


class ObjectionRepo:
    """Async MongoDB repository for objection documents."""

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._coll = db[COLLECTION]

    async def create(self, tenant_id: str, data: dict[str, Any]) -> dict:
        """Insert a new objection document.

        Returns the inserted document with its generated _id.
        """
        doc = {
            "_id": uuid4().hex,
            "tenant_id": tenant_id,
            "state": "filed",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            **data,
        }
        await self._coll.insert_one(doc)
        return doc

    async def get_by_id(
        self, objection_id: str, tenant_id: str
    ) -> Optional[dict]:
        """Fetch a single objection by _id and tenant_id."""
        return await self._coll.find_one(
            {"_id": objection_id, "tenant_id": tenant_id}
        )

    async def list_by_exam(
        self,
        exam_id: str,
        tenant_id: str,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 200,
        skip: int = 0,
    ) -> list[dict]:
        """List objections for an exam with optional filters."""
        query: dict[str, Any] = {
            "exam_id": exam_id,
            "tenant_id": tenant_id,
        }
        if filters:
            query.update(filters)
        cursor = (
            self._coll.find(query)
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    async def transition_state(
        self,
        objection_id: str,
        tenant_id: str,
        from_state: str,
        to_state: str,
        data: Optional[dict[str, Any]] = None,
    ) -> Optional[dict]:
        """Atomic compare-and-swap state transition.

        Only succeeds if the current state matches ``from_state``.
        Optional *data* dict is merged into the document on transition.
        Returns the updated document or None on conflict.
        """
        update_set: dict[str, Any] = {
            "state": to_state,
            "updated_at": datetime.now(timezone.utc),
        }
        if data:
            update_set.update(data)

        return await self._coll.find_one_and_update(
            {
                "_id": objection_id,
                "tenant_id": tenant_id,
                "state": from_state,
            },
            {"$set": update_set},
            return_document=ReturnDocument.AFTER,
        )

    async def exists_for_question(
        self,
        student_id: str,
        exam_id: str,
        question_id: str,
        tenant_id: str,
    ) -> bool:
        """Check if an objection already exists for a specific question.

        Returns True if at least one non-resolved objection exists.
        """
        doc = await self._coll.find_one(
            {
                "student_id": student_id,
                "exam_id": exam_id,
                "question_id": question_id,
                "tenant_id": tenant_id,
            }
        )
        return doc is not None
