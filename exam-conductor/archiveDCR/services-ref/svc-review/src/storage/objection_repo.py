"""PostgreSQL CRUD for objections — single-writer locking.

All state transitions use ``SELECT ... FOR UPDATE`` to prevent
concurrent mutation of the same objection row.

Owner: svc-review (per STATE_OWNERSHIP_MAP.md)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from exampen_common.logging import get_logger

_log = get_logger(__name__)


class ObjectionNotFoundError(Exception):
    """Raised when an objection ID does not exist."""


class ObjectionRepo:
    """CRUD operations for the ``objections`` table."""

    def __init__(self, sf: async_sessionmaker[AsyncSession]) -> None:
        self._sf = sf

    # -- Create ----------------------------------------------------------------

    async def create(
        self,
        *,
        exam_id: str,
        student_id: str,
        question_id: str,
        objection_text: str,
        tenant_id: str,
    ) -> dict[str, Any]:
        """Insert a new objection in ``filed`` state."""
        objection_id = str(uuid4())
        now = datetime.now(timezone.utc)
        async with self._sf() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO objections
                        (objection_id, tenant_id, exam_id, student_id,
                         question_id, objection_text, status, filed_at)
                    VALUES
                        (:objection_id, :tenant_id, :exam_id, :student_id,
                         :question_id, :objection_text, 'filed', :filed_at)
                    """
                ),
                {
                    "objection_id": objection_id,
                    "tenant_id": tenant_id,
                    "exam_id": exam_id,
                    "student_id": student_id,
                    "question_id": question_id,
                    "objection_text": objection_text,
                    "filed_at": now,
                },
            )
            await session.commit()

        _log.info(
            "Objection created id=%s exam=%s student=%s question=%s",
            objection_id, exam_id, student_id, question_id,
        )
        return {
            "objection_id": objection_id,
            "exam_id": exam_id,
            "student_id": student_id,
            "question_id": question_id,
            "objection_text": objection_text,
            "status": "filed",
            "filed_at": now.isoformat(),
        }

    # -- Read ------------------------------------------------------------------

    async def get_by_id(self, objection_id: str) -> dict[str, Any]:
        """Fetch a single objection by ID. Raises if not found."""
        async with self._sf() as session:
            result = await session.execute(
                text(
                    """
                    SELECT objection_id, exam_id, student_id, question_id,
                           objection_text, status, filed_at, assigned_to,
                           resolution, resolution_reason, score_delta
                    FROM objections
                    WHERE objection_id = :objection_id
                    """
                ),
                {"objection_id": objection_id},
            )
            row = result.mappings().first()

        if row is None:
            raise ObjectionNotFoundError(objection_id)
        return _row_to_dict(row)

    async def list_objections(
        self,
        *,
        exam_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List objections with optional filters."""
        clauses: list[str] = []
        params: dict[str, Any] = {}

        if exam_id is not None:
            clauses.append("exam_id = :exam_id")
            params["exam_id"] = exam_id
        if status is not None:
            clauses.append("status = :status")
            params["status"] = status

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        query = f"""
            SELECT objection_id, exam_id, student_id, question_id,
                   status, filed_at
            FROM objections
            {where}
            ORDER BY filed_at DESC
        """
        async with self._sf() as session:
            result = await session.execute(text(query), params)
            rows = result.mappings().all()

        return [_summary_to_dict(r) for r in rows]

    async def exists_for_question(
        self, *, student_id: str, exam_id: str, question_id: str
    ) -> bool:
        """Return True if the student already filed for this question."""
        async with self._sf() as session:
            result = await session.execute(
                text(
                    """
                    SELECT 1 FROM objections
                    WHERE student_id = :student_id
                      AND exam_id = :exam_id
                      AND question_id = :question_id
                    LIMIT 1
                    """
                ),
                {
                    "student_id": student_id,
                    "exam_id": exam_id,
                    "question_id": question_id,
                },
            )
            return result.first() is not None

    # -- Transitions (single-writer locking) -----------------------------------

    async def transition_state(
        self,
        objection_id: str,
        *,
        expected_state: str,
        new_state: str,
        assigned_to: str | None = None,
        resolution: str | None = None,
        resolution_reason: str | None = None,
        score_delta: float | None = None,
    ) -> dict[str, Any]:
        """Atomically transition an objection's state with row-level lock.

        Uses ``SELECT ... FOR UPDATE`` to prevent concurrent mutation.
        Raises ``ObjectionNotFoundError`` if the row is missing and
        ``ValueError`` if the current state does not match *expected_state*.
        """
        async with self._sf() as session:
            result = await session.execute(
                text(
                    """
                    SELECT objection_id, status
                    FROM objections
                    WHERE objection_id = :objection_id
                    FOR UPDATE
                    """
                ),
                {"objection_id": objection_id},
            )
            row = result.mappings().first()

            if row is None:
                raise ObjectionNotFoundError(objection_id)

            if row["status"] != expected_state:
                raise ValueError(
                    f"Expected state '{expected_state}', "
                    f"found '{row['status']}'"
                )

            set_parts = ["status = :new_state"]
            params: dict[str, Any] = {
                "objection_id": objection_id,
                "new_state": new_state,
            }

            if assigned_to is not None:
                set_parts.append("assigned_to = :assigned_to")
                params["assigned_to"] = assigned_to
            if resolution is not None:
                set_parts.append("resolution = :resolution")
                params["resolution"] = resolution
            if resolution_reason is not None:
                set_parts.append("resolution_reason = :resolution_reason")
                params["resolution_reason"] = resolution_reason
            if score_delta is not None:
                set_parts.append("score_delta = :score_delta")
                params["score_delta"] = score_delta

            update_sql = f"""
                UPDATE objections
                SET {', '.join(set_parts)}
                WHERE objection_id = :objection_id
            """
            await session.execute(text(update_sql), params)
            await session.commit()

        _log.info(
            "Objection %s transitioned %s -> %s",
            objection_id, expected_state, new_state,
        )
        return await self.get_by_id(objection_id)


# ---------------------------------------------------------------------------
# Row mapping helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert a full objection row to a detail dict."""
    filed_at = row["filed_at"]
    return {
        "objection_id": row["objection_id"],
        "exam_id": row["exam_id"],
        "student_id": row["student_id"],
        "question_id": row["question_id"],
        "objection_text": row["objection_text"],
        "status": row["status"],
        "filed_at": filed_at.isoformat() if hasattr(filed_at, "isoformat") else str(filed_at),
        "assigned_to": row["assigned_to"],
        "resolution": row["resolution"],
        "resolution_reason": row["resolution_reason"],
        "score_delta": float(row["score_delta"]) if row["score_delta"] is not None else None,
    }


def _summary_to_dict(row: Any) -> dict[str, Any]:
    """Convert a summary row to a list-item dict."""
    filed_at = row["filed_at"]
    return {
        "objection_id": row["objection_id"],
        "exam_id": row["exam_id"],
        "student_id": row["student_id"],
        "question_id": row["question_id"],
        "status": row["status"],
        "filed_at": filed_at.isoformat() if hasattr(filed_at, "isoformat") else str(filed_at),
    }
