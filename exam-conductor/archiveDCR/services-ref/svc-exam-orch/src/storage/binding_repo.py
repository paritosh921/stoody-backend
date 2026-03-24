"""PostgreSQL CRUD for pen-student bindings.

RLS tenant context is assumed to be set by the caller.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from exampen_common.logging import get_logger

_log = get_logger(__name__)


def _row_to_binding(row: Any) -> dict[str, Any]:
    """Convert a database row mapping to a dict."""
    return {
        "exam_id": str(row["exam_id"]),
        "pen_mac": row["pen_mac"],
        "student_id": row["student_id"],
        "student_name": row.get("student_name"),
        "student_roll": row.get("student_roll"),
        "status": row["status"],
        "source": row["source"],
        "bound_at": row["bound_at"].isoformat() if row["bound_at"] else None,
        "server_confirmed_at": (
            row["server_confirmed_at"].isoformat()
            if row.get("server_confirmed_at")
            else None
        ),
        "rejection_reason": row.get("rejection_reason"),
    }


class BindingRepo:
    """Pen binding table CRUD."""

    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def create(self, data: dict[str, Any]) -> dict[str, Any]:
        """Insert a new provisional binding."""
        now = datetime.now(timezone.utc)
        result = await self._s.execute(
            text("""
                INSERT INTO pen_bindings (
                    exam_id, tenant_id, pen_mac, student_id,
                    student_name, student_roll,
                    status, source, bound_at
                ) VALUES (
                    :exam_id, :tenant_id, :pen_mac, :student_id,
                    :student_name, :student_roll,
                    'provisional', :source, :now
                )
                RETURNING *
            """),
            {
                "exam_id": data["exam_id"],
                "tenant_id": data["tenant_id"],
                "pen_mac": data["pen_mac"],
                "student_id": data["student_id"],
                "student_name": data.get("student_name"),
                "student_roll": data.get("student_roll"),
                "source": data["source"],
                "now": now,
            },
        )
        row = result.mappings().one()
        _log.info(
            "Binding created: exam=%s pen=%s student=%s",
            data["exam_id"], data["pen_mac"], data["student_id"],
        )
        return _row_to_binding(row)

    async def list_by_exam(self, exam_id: str) -> list[dict[str, Any]]:
        """Return all bindings for an exam."""
        result = await self._s.execute(
            text("""
                SELECT * FROM pen_bindings
                WHERE exam_id = :eid
                ORDER BY bound_at
            """),
            {"eid": exam_id},
        )
        return [_row_to_binding(r) for r in result.mappings().all()]

    async def get_by_pen(
        self, exam_id: str, pen_mac: str,
    ) -> dict[str, Any] | None:
        """Fetch binding by exam + pen MAC."""
        result = await self._s.execute(
            text("""
                SELECT * FROM pen_bindings
                WHERE exam_id = :eid AND pen_mac = :mac
            """),
            {"eid": exam_id, "mac": pen_mac},
        )
        row = result.mappings().first()
        return _row_to_binding(row) if row else None

    async def confirm_or_reject(
        self,
        exam_id: str,
        pen_mac: str,
        new_status: str,
        rejection_reason: str | None = None,
    ) -> dict[str, Any] | None:
        """Update a provisional binding to confirmed or rejected.

        Uses ``SELECT ... FOR UPDATE`` to prevent races.
        """
        lock_result = await self._s.execute(
            text("""
                SELECT status FROM pen_bindings
                WHERE exam_id = :eid AND pen_mac = :mac
                FOR UPDATE
            """),
            {"eid": exam_id, "mac": pen_mac},
        )
        locked = lock_result.mappings().first()
        if locked is None:
            return None

        now = datetime.now(timezone.utc)
        params: dict[str, Any] = {
            "eid": exam_id,
            "mac": pen_mac,
            "status": new_status,
            "reason": rejection_reason,
            "now": now,
        }

        if new_status == "confirmed":
            result = await self._s.execute(
                text("""
                    UPDATE pen_bindings
                    SET status = :status, server_confirmed_at = :now,
                        rejection_reason = NULL
                    WHERE exam_id = :eid AND pen_mac = :mac
                    RETURNING *
                """),
                params,
            )
        else:
            result = await self._s.execute(
                text("""
                    UPDATE pen_bindings
                    SET status = :status, rejection_reason = :reason
                    WHERE exam_id = :eid AND pen_mac = :mac
                    RETURNING *
                """),
                params,
            )

        row = result.mappings().first()
        _log.info(
            "Binding %s/%s -> %s", exam_id, pen_mac, new_status,
        )
        return _row_to_binding(row) if row else None
