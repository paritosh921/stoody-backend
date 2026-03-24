"""MongoDB storage repo for exam chat messages — Collection: exampen_chat_messages.

APPEND-ONLY: No update or delete methods are provided.

Every query includes tenant_id filter (replacing PostgreSQL RLS).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

COLLECTION = "exampen_chat_messages"


class ChatRepo:
    """Async MongoDB repository for exam chat message documents.

    This is an append-only store.  Messages are never updated or deleted.
    """

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._coll = db[COLLECTION]

    async def append_message(
        self, tenant_id: str, data: dict[str, Any]
    ) -> dict:
        """Insert a new chat message (append-only).

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

    async def get_thread(
        self,
        exam_id: str,
        teacher_id: str,
        student_id: str,
        tenant_id: str,
    ) -> list[dict]:
        """Fetch all messages in a teacher-student thread for an exam.

        Returns documents sorted by created_at ascending (chronological).
        """
        cursor = self._coll.find(
            {
                "exam_id": exam_id,
                "teacher_id": teacher_id,
                "student_id": student_id,
                "tenant_id": tenant_id,
            }
        ).sort("created_at", 1)
        return await cursor.to_list(length=5000)
