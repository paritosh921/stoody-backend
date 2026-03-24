"""MongoDB storage repo for analytics cache — Collection: exampen_analytics_cache.

Stores pre-computed percentiles, leaderboard entries, and class stats
for fast retrieval.

Every query includes tenant_id filter (replacing PostgreSQL RLS).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument, UpdateOne

logger = logging.getLogger(__name__)

COLLECTION = "exampen_analytics_cache"


class AnalyticsCacheRepo:
    """Async MongoDB repository for pre-computed analytics cache."""

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._coll = db[COLLECTION]

    async def upsert_percentiles(
        self,
        exam_id: str,
        tenant_id: str,
        percentiles: list[dict[str, Any]],
    ) -> None:
        """Bulk upsert percentile entries for an exam.

        Each entry should contain ``student_id`` and ``percentile``.
        """
        if not percentiles:
            return

        now = datetime.now(timezone.utc)
        ops: list[UpdateOne] = []
        for entry in percentiles:
            ops.append(
                UpdateOne(
                    {
                        "exam_id": exam_id,
                        "tenant_id": tenant_id,
                        "type": "percentile",
                        "student_id": entry["student_id"],
                    },
                    {
                        "$set": {
                            "percentile": entry["percentile"],
                            "updated_at": now,
                        },
                        "$setOnInsert": {
                            "_id": uuid4().hex,
                            "created_at": now,
                        },
                    },
                    upsert=True,
                )
            )
        await self._coll.bulk_write(ops, ordered=False)

    async def upsert_leaderboard(
        self,
        exam_id: str,
        tenant_id: str,
        entries: list[dict[str, Any]],
    ) -> None:
        """Bulk upsert leaderboard entries for an exam.

        Each entry should contain ``student_id``, ``rank``, ``score``, etc.
        """
        if not entries:
            return

        now = datetime.now(timezone.utc)
        ops: list[UpdateOne] = []
        for entry in entries:
            ops.append(
                UpdateOne(
                    {
                        "exam_id": exam_id,
                        "tenant_id": tenant_id,
                        "type": "leaderboard",
                        "student_id": entry["student_id"],
                    },
                    {
                        "$set": {
                            **entry,
                            "updated_at": now,
                        },
                        "$setOnInsert": {
                            "_id": uuid4().hex,
                            "created_at": now,
                        },
                    },
                    upsert=True,
                )
            )
        await self._coll.bulk_write(ops, ordered=False)

    async def get_leaderboard(
        self, exam_id: str, tenant_id: str
    ) -> list[dict]:
        """Fetch the cached leaderboard for an exam, sorted by rank."""
        cursor = self._coll.find(
            {
                "exam_id": exam_id,
                "tenant_id": tenant_id,
                "type": "leaderboard",
            }
        ).sort("rank", 1)
        return await cursor.to_list(length=1000)

    async def get_class_stats(
        self, exam_id: str, tenant_id: str
    ) -> Optional[dict]:
        """Fetch the cached class statistics for an exam."""
        return await self._coll.find_one(
            {
                "exam_id": exam_id,
                "tenant_id": tenant_id,
                "type": "class_stats",
            }
        )
