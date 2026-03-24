"""MongoDB storage repo for pen-student bindings — Collection: exampen_bindings.

Every query includes tenant_id filter (replacing PostgreSQL RLS).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

logger = logging.getLogger(__name__)

COLLECTION = "exampen_bindings"


class BindingConflictError(Exception):
    """Raised when a binding insert conflicts with an existing record."""


class BindingRepo:
    """Async MongoDB repository for pen-student binding documents."""

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._coll = db[COLLECTION]

    async def create(
        self,
        exam_id: str,
        pen_mac: str,
        tenant_id: str,
        data: dict[str, Any],
    ) -> dict:
        """Insert a new binding.

        Raises BindingConflictError if a duplicate pen_mac+exam_id
        binding already exists.
        """
        doc = {
            "_id": uuid4().hex,
            "exam_id": exam_id,
            "pen_mac": pen_mac,
            "tenant_id": tenant_id,
            "status": "provisional",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            **data,
        }
        try:
            await self._coll.insert_one(doc)
        except DuplicateKeyError:
            raise BindingConflictError(
                f"Binding already exists for pen {pen_mac} in exam {exam_id}"
            )
        return doc

    async def list_by_exam(
        self, exam_id: str, tenant_id: str
    ) -> list[dict]:
        """List all bindings for an exam."""
        cursor = self._coll.find(
            {"exam_id": exam_id, "tenant_id": tenant_id}
        )
        return await cursor.to_list(length=500)

    async def update_status(
        self,
        exam_id: str,
        pen_mac: str,
        tenant_id: str,
        status: str,
    ) -> Optional[dict]:
        """Update the status of a specific binding.

        Returns the updated document or None if not found.
        """
        return await self._coll.find_one_and_update(
            {
                "exam_id": exam_id,
                "pen_mac": pen_mac,
                "tenant_id": tenant_id,
            },
            {
                "$set": {
                    "status": status,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
            return_document=ReturnDocument.AFTER,
        )
