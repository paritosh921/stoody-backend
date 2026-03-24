"""MongoDB storage repo for AI results — Collection: exampen_ai_results.

AI results are immutable once stored.  The model_version is preserved
and results are never updated — a new version produces a new document.

Every query includes tenant_id filter (replacing PostgreSQL RLS).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

COLLECTION = "exampen_ai_results"


class AIResultRepo:
    """Async MongoDB repository for AI recognition/scoring result documents."""

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._coll = db[COLLECTION]

    async def store_result(
        self, tenant_id: str, data: dict[str, Any]
    ) -> dict:
        """Store a new AI result (immutable, never updated).

        The ``model_version`` field in *data* is preserved as-is.
        Returns the inserted document.
        """
        doc = {
            "_id": uuid4().hex,
            "tenant_id": tenant_id,
            "created_at": datetime.now(timezone.utc),
            **data,
        }
        await self._coll.insert_one(doc)
        return doc

    async def get_results(
        self,
        exam_id: str,
        student_id: str,
        tenant_id: str,
    ) -> list[dict]:
        """Fetch all AI results for a student in an exam.

        Returns documents sorted by created_at ascending so the
        latest result is last.
        """
        cursor = self._coll.find(
            {
                "exam_id": exam_id,
                "student_id": student_id,
                "tenant_id": tenant_id,
            }
        ).sort("created_at", 1)
        return await cursor.to_list(length=1000)
