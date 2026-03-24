"""MongoDB storage repo for plagiarism flags — Collection: exampen_plagiarism_flags.

Every query includes tenant_id filter (replacing PostgreSQL RLS).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

COLLECTION = "exampen_plagiarism_flags"


class PlagiarismRepo:
    """Async MongoDB repository for plagiarism flag documents."""

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._coll = db[COLLECTION]

    async def bulk_insert(
        self, flags: list[dict[str, Any]], tenant_id: str
    ) -> list[str]:
        """Insert multiple plagiarism flag documents.

        Each flag document gets a generated _id and tenant_id.
        Returns the list of inserted document IDs.
        """
        if not flags:
            return []

        now = datetime.now(timezone.utc)
        docs = []
        for flag in flags:
            doc_id = uuid4().hex
            docs.append(
                {
                    "_id": doc_id,
                    "tenant_id": tenant_id,
                    "verdict": None,
                    "verdict_reason": None,
                    "created_at": now,
                    "updated_at": now,
                    **flag,
                }
            )

        result = await self._coll.insert_many(docs)
        return [str(oid) for oid in result.inserted_ids]

    async def list_by_exam(
        self, exam_id: str, tenant_id: str
    ) -> list[dict]:
        """List all plagiarism flags for an exam."""
        cursor = self._coll.find(
            {"exam_id": exam_id, "tenant_id": tenant_id}
        ).sort("created_at", -1)
        return await cursor.to_list(length=5000)

    async def get_by_id(
        self, flag_id: str, tenant_id: str
    ) -> Optional[dict]:
        """Fetch a single plagiarism flag by _id and tenant_id."""
        return await self._coll.find_one(
            {"_id": flag_id, "tenant_id": tenant_id}
        )

    async def update_verdict(
        self,
        flag_id: str,
        tenant_id: str,
        verdict: str,
        reason: str,
    ) -> bool:
        """Update the teacher's verdict on a plagiarism flag.

        Returns True if a document was modified, False otherwise.
        """
        result = await self._coll.update_one(
            {"_id": flag_id, "tenant_id": tenant_id},
            {
                "$set": {
                    "verdict": verdict,
                    "verdict_reason": reason,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )
        return result.modified_count > 0
