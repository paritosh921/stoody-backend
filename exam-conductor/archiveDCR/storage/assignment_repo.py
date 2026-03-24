"""MongoDB storage repo for exam role assignments — Collection: exampen_assignments.

Every query includes tenant_id filter (replacing PostgreSQL RLS).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument, UpdateOne

logger = logging.getLogger(__name__)

COLLECTION = "exampen_assignments"


class AssignmentRepo:
    """Async MongoDB repository for exam role assignment documents."""

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._coll = db[COLLECTION]

    async def upsert(
        self,
        exam_id: str,
        tenant_id: str,
        invigilator_ids: list[str],
        evaluator_ids: list[str],
    ) -> None:
        """Bulk upsert role assignments for an exam.

        Each user_id gets a document with their assigned roles.
        """
        now = datetime.now(timezone.utc)
        ops: list[UpdateOne] = []

        # Build a user_id -> roles mapping
        user_roles: dict[str, list[str]] = {}
        for uid in invigilator_ids:
            user_roles.setdefault(uid, []).append("invigilator")
        for uid in evaluator_ids:
            user_roles.setdefault(uid, []).append("evaluator")

        for user_id, roles in user_roles.items():
            ops.append(
                UpdateOne(
                    {
                        "exam_id": exam_id,
                        "user_id": user_id,
                        "tenant_id": tenant_id,
                    },
                    {
                        "$set": {
                            "roles": roles,
                            "updated_at": now,
                        },
                        "$setOnInsert": {
                            "created_at": now,
                        },
                    },
                    upsert=True,
                )
            )

        if ops:
            await self._coll.bulk_write(ops, ordered=False)

    async def list_by_exam(
        self, exam_id: str, tenant_id: str
    ) -> list[dict]:
        """List all role assignments for an exam."""
        cursor = self._coll.find(
            {"exam_id": exam_id, "tenant_id": tenant_id}
        )
        return await cursor.to_list(length=500)

    async def get_user_roles(
        self, exam_id: str, user_id: str, tenant_id: str
    ) -> Optional[dict]:
        """Get a specific user's roles for an exam."""
        return await self._coll.find_one(
            {
                "exam_id": exam_id,
                "user_id": user_id,
                "tenant_id": tenant_id,
            }
        )
