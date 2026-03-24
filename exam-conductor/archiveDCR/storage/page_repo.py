"""MongoDB storage repo for exam pages — Collection: exampen_pages.

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

COLLECTION = "exampen_pages"


class PageRepo:
    """Async MongoDB repository for exam page documents."""

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._coll = db[COLLECTION]

    async def upsert(
        self,
        exam_id: str,
        student_id: str,
        page_number: int,
        tenant_id: str,
        data: dict[str, Any],
    ) -> dict:
        """Upsert a page document for a student's exam page.

        Creates the page if it does not exist; updates it otherwise.
        Returns the resulting document.
        """
        now = datetime.now(timezone.utc)
        return await self._coll.find_one_and_update(
            {
                "exam_id": exam_id,
                "student_id": student_id,
                "page_number": page_number,
                "tenant_id": tenant_id,
            },
            {
                "$set": {
                    **data,
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "_id": uuid4().hex,
                    "created_at": now,
                },
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    async def get_page(
        self,
        exam_id: str,
        student_id: str,
        page_number: int,
        tenant_id: str,
    ) -> Optional[dict]:
        """Fetch a single page document."""
        return await self._coll.find_one(
            {
                "exam_id": exam_id,
                "student_id": student_id,
                "page_number": page_number,
                "tenant_id": tenant_id,
            }
        )
