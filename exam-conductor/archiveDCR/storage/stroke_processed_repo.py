"""MongoDB storage repo for processed strokes — Collection: exampen_strokes_processed.

Every query includes tenant_id filter (replacing PostgreSQL RLS).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

logger = logging.getLogger(__name__)

COLLECTION = "exampen_strokes_processed"


class StrokeProcessedRepo:
    """Async MongoDB repository for processed (normalized) stroke documents."""

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._coll = db[COLLECTION]

    async def commit_strokes(
        self,
        idempotency_key: str,
        strokes: list[dict[str, Any]],
        tenant_id: str,
    ) -> bool:
        """Insert processed strokes with idempotency.

        Each stroke document is tagged with the batch's idempotency_key.
        If the key was already committed (DuplicateKeyError on the first
        document), returns False (idempotent no-op).  Otherwise True.
        """
        if not strokes:
            return True

        now = datetime.now(timezone.utc)
        docs = []
        for i, stroke in enumerate(strokes):
            docs.append(
                {
                    "_id": f"{idempotency_key}:{i}",
                    "idempotency_key": idempotency_key,
                    "tenant_id": tenant_id,
                    "committed_at": now,
                    **stroke,
                }
            )

        try:
            await self._coll.insert_many(docs, ordered=True)
            return True
        except DuplicateKeyError:
            logger.debug(
                "Duplicate stroke batch ignored: %s", idempotency_key
            )
            return False

    async def get_strokes(
        self,
        exam_id: str,
        pen_mac: str,
        page_number: int,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        """Fetch processed strokes for a specific pen + page.

        Returns a list of stroke documents sorted by committed_at.
        """
        cursor = self._coll.find(
            {
                "exam_id": exam_id,
                "pen_mac": pen_mac,
                "page_number": page_number,
                "tenant_id": tenant_id,
            }
        ).sort("committed_at", 1)
        return await cursor.to_list(length=10_000)
