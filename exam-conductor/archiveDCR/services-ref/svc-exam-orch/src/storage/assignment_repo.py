"""PostgreSQL CRUD for invigilator and evaluator assignments.

RLS tenant context is assumed to be set by the caller.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from exampen_common.logging import get_logger

_log = get_logger(__name__)


def _row_to_assignment(row: Any) -> dict[str, Any]:
    return {
        "exam_id": str(row["exam_id"]),
        "user_id": row["user_id"],
        "role": row["role"],
        "assigned_at": row["assigned_at"].isoformat()
        if row.get("assigned_at")
        else None,
    }


class AssignmentRepo:
    """Invigilator / evaluator assignment CRUD."""

    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def upsert_assignments(
        self,
        exam_id: str,
        tenant_id: str,
        invigilator_ids: list[str],
        evaluator_ids: list[str],
        double_blind: bool = False,
    ) -> dict[str, Any]:
        """Replace all assignments for an exam and return the new set.

        Deletes existing assignments, then bulk-inserts the new ones.
        """
        await self._s.execute(
            text("DELETE FROM assignments WHERE exam_id = :eid"),
            {"eid": exam_id},
        )

        now = datetime.now(timezone.utc)
        rows: list[dict[str, Any]] = []

        for uid in invigilator_ids:
            rows.append({
                "exam_id": exam_id,
                "tenant_id": tenant_id,
                "user_id": uid,
                "role": "invigilator",
                "now": now,
            })
        for uid in evaluator_ids:
            rows.append({
                "exam_id": exam_id,
                "tenant_id": tenant_id,
                "user_id": uid,
                "role": "evaluator",
                "now": now,
            })

        for r in rows:
            await self._s.execute(
                text("""
                    INSERT INTO assignments (
                        exam_id, tenant_id, user_id, role, assigned_at
                    ) VALUES (
                        :exam_id, :tenant_id, :user_id, :role, :now
                    )
                """),
                r,
            )

        _log.info(
            "Assignments updated for exam %s: %d invigilators, %d evaluators",
            exam_id, len(invigilator_ids), len(evaluator_ids),
        )

        return {
            "invigilator_ids": invigilator_ids,
            "evaluator_ids": evaluator_ids,
            "double_blind": double_blind,
        }

    async def list_by_exam(self, exam_id: str) -> list[dict[str, Any]]:
        """Return all assignments for an exam."""
        result = await self._s.execute(
            text("""
                SELECT * FROM assignments
                WHERE exam_id = :eid
                ORDER BY role, assigned_at
            """),
            {"eid": exam_id},
        )
        return [_row_to_assignment(r) for r in result.mappings().all()]
