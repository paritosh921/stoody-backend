"""MongoDB storage repo for exams — Collection: exampen_exams.

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

COLLECTION = "exampen_exams"


class ExamRepo:
    """Async MongoDB repository for exam documents."""

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._coll = db[COLLECTION]

    async def create(self, tenant_id: str, data: dict[str, Any]) -> dict:
        """Insert a new exam document.

        Returns the inserted document with its generated _id.
        """
        doc = {
            "_id": uuid4().hex,
            "tenant_id": tenant_id,
            "state": "created",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            **data,
        }
        await self._coll.insert_one(doc)
        return doc

    async def get_by_id(
        self, exam_id: str, tenant_id: str
    ) -> Optional[dict]:
        """Fetch a single exam by _id and tenant_id."""
        return await self._coll.find_one(
            {"_id": exam_id, "tenant_id": tenant_id}
        )

    async def list_exams(
        self,
        tenant_id: str,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> list[dict]:
        """List exams for a tenant with optional filters."""
        query: dict[str, Any] = {"tenant_id": tenant_id}
        if filters:
            query.update(filters)
        cursor = (
            self._coll.find(query)
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    async def update(
        self, exam_id: str, tenant_id: str, updates: dict[str, Any]
    ) -> Optional[dict]:
        """Update arbitrary fields on an exam (non-state fields).

        Returns the updated document or None if not found.
        """
        updates["updated_at"] = datetime.now(timezone.utc)
        return await self._coll.find_one_and_update(
            {"_id": exam_id, "tenant_id": tenant_id},
            {"$set": updates},
            return_document=ReturnDocument.AFTER,
        )

    async def transition_state(
        self,
        exam_id: str,
        tenant_id: str,
        from_state: str,
        to_state: str,
    ) -> Optional[dict]:
        """Atomic compare-and-swap state transition.

        Only succeeds if the current state matches ``from_state``.
        Returns the updated document or None on conflict.
        """
        return await self._coll.find_one_and_update(
            {
                "_id": exam_id,
                "tenant_id": tenant_id,
                "state": from_state,
            },
            {
                "$set": {
                    "state": to_state,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
            return_document=ReturnDocument.AFTER,
        )

    async def update_rubric(
        self, exam_id: str, tenant_id: str, rubric: dict[str, Any]
    ) -> None:
        """Replace the rubric definition on an exam."""
        await self._coll.update_one(
            {"_id": exam_id, "tenant_id": tenant_id},
            {
                "$set": {
                    "rubric": rubric,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

    async def update_regions(
        self,
        exam_id: str,
        tenant_id: str,
        regions: list[dict[str, Any]],
    ) -> None:
        """Replace the question regions on an exam."""
        await self._coll.update_one(
            {"_id": exam_id, "tenant_id": tenant_id},
            {
                "$set": {
                    "regions": regions,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )
