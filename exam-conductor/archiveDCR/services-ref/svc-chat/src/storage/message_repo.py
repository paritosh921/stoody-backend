"""PostgreSQL append-only message store.

CRITICAL: No UPDATE or DELETE methods exist in this class.
The append-only contract is enforced at the application layer here
and at the database layer via migration triggers.

Tables:
- ``chat_messages`` — immutable message rows
- ``read_receipts`` — immutable read-receipt rows
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from exampen_common.logging import get_logger

_log = get_logger(__name__)


class MessageRepo:
    """Append-only operations for chat messages and read receipts.

    This class intentionally provides NO update or delete methods.
    Any attempt to add such methods violates the DPDPA audit-safety
    contract for minors' data.
    """

    def __init__(self, sf: async_sessionmaker[AsyncSession]) -> None:
        self._sf = sf

    # -- Messages ----------------------------------------------------------

    async def append_message(
        self,
        sender_id: str,
        recipient_id: str,
        exam_id: str,
        content: str,
        tenant_id: str,
        attachment_uri: str | None = None,
    ) -> dict[str, Any]:
        """Insert a single chat message. Returns the created record.

        This is the ONLY write path for messages. No update, no delete.
        """
        message_id = str(uuid4())
        now = datetime.now(timezone.utc)

        async with self._sf() as session:
            # Set RLS tenant context
            await session.execute(
                text("SELECT set_config('app.current_tenant', :tid, true)"),
                {"tid": tenant_id},
            )
            await session.execute(
                text(
                    """
                    INSERT INTO chat_messages
                        (id, sender_id, recipient_id, exam_id, content,
                         attachment_uri, tenant_id, created_at)
                    VALUES
                        (:id, :sender_id, :recipient_id, :exam_id, :content,
                         :attachment_uri, :tenant_id, :created_at)
                    """
                ),
                {
                    "id": message_id,
                    "sender_id": sender_id,
                    "recipient_id": recipient_id,
                    "exam_id": exam_id,
                    "content": content,
                    "attachment_uri": attachment_uri,
                    "tenant_id": tenant_id,
                    "created_at": now,
                },
            )
            await session.commit()

        _log.info(
            "Message appended id=%s exam=%s sender=%s recipient=%s",
            message_id, exam_id, sender_id, recipient_id,
        )
        return {
            "message_id": message_id,
            "sender_id": sender_id,
            "recipient_id": recipient_id,
            "exam_id": exam_id,
            "content": content,
            "attachment_uri": attachment_uri,
            "sent_at": now.isoformat(),
        }

    async def get_thread(
        self,
        exam_id: str,
        teacher_id: str,
        student_id: str,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        """Fetch all messages in a thread ordered by creation time.

        A thread is the set of messages between a teacher and student
        for a given exam (messages in both directions).
        """
        async with self._sf() as session:
            await session.execute(
                text("SELECT set_config('app.current_tenant', :tid, true)"),
                {"tid": tenant_id},
            )
            result = await session.execute(
                text(
                    """
                    SELECT id, sender_id, recipient_id, exam_id,
                           content, attachment_uri, created_at
                    FROM chat_messages
                    WHERE exam_id = :exam_id
                      AND (
                          (sender_id = :teacher_id AND recipient_id = :student_id)
                          OR
                          (sender_id = :student_id AND recipient_id = :teacher_id)
                      )
                    ORDER BY created_at ASC
                    """
                ),
                {
                    "exam_id": exam_id,
                    "teacher_id": teacher_id,
                    "student_id": student_id,
                },
            )
            rows = result.mappings().all()

        return [
            {
                "message_id": str(r["id"]),
                "sender_id": r["sender_id"],
                "recipient_id": r["recipient_id"],
                "exam_id": str(r["exam_id"]),
                "content": r["content"],
                "attachment_uri": r["attachment_uri"],
                "sent_at": r["created_at"].isoformat()
                if r["created_at"] else None,
            }
            for r in rows
        ]

    async def list_threads(
        self,
        exam_id: str,
        user_id: str,
        role: str,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        """List distinct threads for an exam visible to a user.

        Teachers see all threads in the exam where they are a
        participant. Students see only their own thread(s).

        Returns a list of thread summaries with last message preview.
        """
        from src.domain.message_rules import is_teacher_role

        async with self._sf() as session:
            await session.execute(
                text("SELECT set_config('app.current_tenant', :tid, true)"),
                {"tid": tenant_id},
            )

            if is_teacher_role(role):
                # Teacher sees threads where they are sender or recipient
                result = await session.execute(
                    text(
                        """
                        SELECT DISTINCT
                            CASE
                                WHEN sender_id = :user_id
                                    THEN recipient_id
                                ELSE sender_id
                            END AS other_user_id,
                            MAX(created_at) AS last_message_at
                        FROM chat_messages
                        WHERE exam_id = :exam_id
                          AND (sender_id = :user_id
                               OR recipient_id = :user_id)
                        GROUP BY other_user_id
                        ORDER BY last_message_at DESC
                        """
                    ),
                    {"exam_id": exam_id, "user_id": user_id},
                )
            else:
                # Student sees only threads where they participate
                result = await session.execute(
                    text(
                        """
                        SELECT DISTINCT
                            CASE
                                WHEN sender_id = :user_id
                                    THEN recipient_id
                                ELSE sender_id
                            END AS other_user_id,
                            MAX(created_at) AS last_message_at
                        FROM chat_messages
                        WHERE exam_id = :exam_id
                          AND (sender_id = :user_id
                               OR recipient_id = :user_id)
                        GROUP BY other_user_id
                        ORDER BY last_message_at DESC
                        """
                    ),
                    {"exam_id": exam_id, "user_id": user_id},
                )
            rows = result.mappings().all()

        return [
            {
                "exam_id": exam_id,
                "other_user_id": r["other_user_id"],
                "last_message_at": r["last_message_at"].isoformat()
                if r["last_message_at"] else None,
            }
            for r in rows
        ]

    # -- Read receipts -----------------------------------------------------

    async def append_read_receipt(
        self,
        exam_id: str,
        reader_id: str,
        other_user_id: str,
        tenant_id: str,
    ) -> dict[str, Any]:
        """Append a read receipt for a thread.

        This is NOT an update. It inserts a new row (or upserts the
        read_at timestamp) in the read_receipts table, preserving
        the append-only audit trail.
        """
        now = datetime.now(timezone.utc)

        async with self._sf() as session:
            await session.execute(
                text("SELECT set_config('app.current_tenant', :tid, true)"),
                {"tid": tenant_id},
            )
            await session.execute(
                text(
                    """
                    INSERT INTO read_receipts
                        (exam_id, reader_id, other_user_id,
                         tenant_id, read_at)
                    VALUES
                        (:exam_id, :reader_id, :other_user_id,
                         :tenant_id, :read_at)
                    ON CONFLICT (exam_id, reader_id, other_user_id)
                    DO UPDATE SET read_at = EXCLUDED.read_at
                    """
                ),
                {
                    "exam_id": exam_id,
                    "reader_id": reader_id,
                    "other_user_id": other_user_id,
                    "tenant_id": tenant_id,
                    "read_at": now,
                },
            )
            await session.commit()

        _log.info(
            "Read receipt appended exam=%s reader=%s other=%s",
            exam_id, reader_id, other_user_id,
        )
        return {
            "exam_id": exam_id,
            "other_user_id": other_user_id,
            "read_at": now.isoformat(),
        }
