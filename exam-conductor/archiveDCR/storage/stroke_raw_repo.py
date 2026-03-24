"""MongoDB storage repo for raw stroke chunks — Collection: exampen_strokes_raw.

Every query includes tenant_id filter (replacing PostgreSQL RLS).
Idempotent writes: DuplicateKeyError on the idempotency key is silently
swallowed (chunk already recorded).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError

logger = logging.getLogger(__name__)

COLLECTION = "exampen_strokes_raw"


class StrokeRawRepo:
    """Async MongoDB repository for raw stroke chunk documents."""

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._coll = db[COLLECTION]

    async def record_chunk(
        self,
        exam_id: str,
        pen_mac: str,
        chunk_index: int,
        tenant_id: str,
        data: dict[str, Any],
    ) -> bool:
        """Record a raw stroke chunk.

        Uses the idempotency key ``{exam_id}:{pen_mac}:{chunk_index}``
        as the document _id.  If the chunk was already recorded
        (DuplicateKeyError), returns False.  Otherwise returns True.
        """
        idempotency_key = f"{exam_id}:{pen_mac}:{chunk_index}"
        doc = {
            "_id": idempotency_key,
            "exam_id": exam_id,
            "pen_mac": pen_mac,
            "chunk_index": chunk_index,
            "tenant_id": tenant_id,
            "received_at": datetime.now(timezone.utc),
            **data,
        }
        try:
            await self._coll.insert_one(doc)
            return True
        except DuplicateKeyError:
            logger.debug(
                "Duplicate chunk ignored: %s", idempotency_key
            )
            return False

    async def get_pen_progress(
        self, exam_id: str, pen_mac: str, tenant_id: str
    ) -> dict[str, Any]:
        """Get upload progress for a specific pen.

        Returns a dict with ``received_chunks`` (list of chunk indices)
        and ``count``.
        """
        pipeline = [
            {
                "$match": {
                    "exam_id": exam_id,
                    "pen_mac": pen_mac,
                    "tenant_id": tenant_id,
                }
            },
            {
                "$group": {
                    "_id": None,
                    "received_chunks": {"$push": "$chunk_index"},
                    "count": {"$sum": 1},
                }
            },
        ]
        results = await self._coll.aggregate(pipeline).to_list(length=1)
        if not results:
            return {"received_chunks": [], "count": 0}
        row = results[0]
        return {
            "received_chunks": sorted(row["received_chunks"]),
            "count": row["count"],
        }

    async def get_exam_upload_status(
        self, exam_id: str, tenant_id: str
    ) -> list[dict[str, Any]]:
        """Get upload status for all pens in an exam, grouped by pen_mac.

        Returns a list of dicts:
        ``[{"pen_mac": str, "received_chunks": [int], "count": int}, ...]``
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
                    "_id": "$pen_mac",
                    "received_chunks": {"$push": "$chunk_index"},
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"_id": 1}},
        ]
        results = await self._coll.aggregate(pipeline).to_list(length=500)
        return [
            {
                "pen_mac": row["_id"],
                "received_chunks": sorted(row["received_chunks"]),
                "count": row["count"],
            }
            for row in results
        ]
