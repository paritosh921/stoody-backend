"""PostgreSQL CRUD for exams — with row-level locking on FSM transitions.

Uses SQLAlchemy async sessions. RLS tenant context is assumed to be set
by the caller (via ``exampen_common.db.rls_session``).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from exampen_common.logging import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Row mappers
# ---------------------------------------------------------------------------


def _row_to_exam(row: Any) -> dict[str, Any]:
    """Convert a database row mapping to a dict."""
    return {
        "exam_id": str(row["exam_id"]),
        "title": row["title"],
        "subject_id": row["subject_id"],
        "class_id": row["class_id"],
        "section_id": row["section_id"],
        "scheduled_at": row["scheduled_at"].isoformat()
        if row["scheduled_at"]
        else None,
        "duration_min": row["duration_min"],
        "question_count": row["question_count"],
        "total_marks": float(row["total_marks"]),
        "negative_marking": row["negative_marking"],
        "variants": row["variants"] or [],
        "state": row["state"],
        "created_by": row["created_by"],
        "late_entry_cutoff_min": row.get("late_entry_cutoff_min"),
        "objection_window_days": row.get("objection_window_days"),
        "created_at": row["created_at"].isoformat()
        if row.get("created_at")
        else None,
        "updated_at": row["updated_at"].isoformat()
        if row.get("updated_at")
        else None,
    }


# ---------------------------------------------------------------------------
# ExamRepo
# ---------------------------------------------------------------------------


class ExamRepo:
    """Exam table CRUD operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def create(self, data: dict[str, Any]) -> dict[str, Any]:
        """Insert a new exam row and return it."""
        exam_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        result = await self._s.execute(
            text("""
                INSERT INTO exams (
                    exam_id, tenant_id, title, subject_id, class_id,
                    section_id, scheduled_at, duration_min,
                    question_count, total_marks, negative_marking,
                    variants, state, created_by, created_at, updated_at
                ) VALUES (
                    :exam_id, :tenant_id, :title, :subject_id, :class_id,
                    :section_id, :scheduled_at, :duration_min,
                    :question_count, :total_marks, :negative_marking,
                    :variants, 'created', :created_by, :now, :now
                )
                RETURNING *
            """),
            {
                "exam_id": exam_id,
                "tenant_id": data["tenant_id"],
                "title": data["title"],
                "subject_id": data["subject_id"],
                "class_id": data["class_id"],
                "section_id": data["section_id"],
                "scheduled_at": data["scheduled_at"],
                "duration_min": data["duration_min"],
                "question_count": data["question_count"],
                "total_marks": data["total_marks"],
                "negative_marking": data.get("negative_marking", False),
                "variants": data.get("variants", []),
                "created_by": data["created_by"],
                "now": now,
            },
        )
        row = result.mappings().one()
        _log.info("Exam created: %s", exam_id)
        return _row_to_exam(row)

    async def get_by_id(self, exam_id: str) -> dict[str, Any] | None:
        """Fetch a single exam by ID, or ``None``."""
        result = await self._s.execute(
            text("SELECT * FROM exams WHERE exam_id = :eid"),
            {"eid": exam_id},
        )
        row = result.mappings().first()
        return _row_to_exam(row) if row else None

    async def list_exams(
        self,
        *,
        state: str | None = None,
        subject_id: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """List exams with optional filters."""
        clauses: list[str] = ["1=1"]
        params: dict[str, Any] = {}

        if state:
            clauses.append("state = :state")
            params["state"] = state
        if subject_id:
            clauses.append("subject_id = :subject_id")
            params["subject_id"] = subject_id
        if from_date:
            clauses.append("scheduled_at >= :from_date")
            params["from_date"] = from_date
        if to_date:
            clauses.append("scheduled_at <= :to_date")
            params["to_date"] = to_date

        where = " AND ".join(clauses)
        result = await self._s.execute(
            text(f"SELECT * FROM exams WHERE {where} ORDER BY scheduled_at DESC"),  # noqa: S608
            params,
        )
        return [_row_to_exam(r) for r in result.mappings().all()]

    async def update(
        self, exam_id: str, data: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Update mutable fields (only in 'created' state)."""
        sets: list[str] = []
        params: dict[str, Any] = {"eid": exam_id, "now": datetime.now(timezone.utc)}

        for key in ("scheduled_at", "duration_min",
                     "objection_window_days", "late_entry_cutoff_min"):
            if key in data:
                sets.append(f"{key} = :{key}")
                params[key] = data[key]

        if not sets:
            return await self.get_by_id(exam_id)

        sets.append("updated_at = :now")
        set_clause = ", ".join(sets)

        result = await self._s.execute(
            text(f"""
                UPDATE exams SET {set_clause}
                WHERE exam_id = :eid AND state = 'created'
                RETURNING *
            """),
            params,
        )
        row = result.mappings().first()
        return _row_to_exam(row) if row else None

    async def transition_state(
        self,
        exam_id: str,
        expected_from: str,
        new_state: str,
    ) -> dict[str, Any] | None:
        """Atomically transition exam state with row-level locking.

        Uses ``SELECT ... FOR UPDATE`` to prevent concurrent transitions.
        Returns the updated row, or ``None`` if the exam was not found or
        the current state did not match ``expected_from``.
        """
        # Lock the row
        lock_result = await self._s.execute(
            text("""
                SELECT state FROM exams
                WHERE exam_id = :eid
                FOR UPDATE
            """),
            {"eid": exam_id},
        )
        locked = lock_result.mappings().first()
        if locked is None:
            return None
        if locked["state"] != expected_from:
            return None

        now = datetime.now(timezone.utc)
        result = await self._s.execute(
            text("""
                UPDATE exams
                SET state = :new_state, updated_at = :now
                WHERE exam_id = :eid
                RETURNING *
            """),
            {"eid": exam_id, "new_state": new_state, "now": now},
        )
        row = result.mappings().one()
        _log.info(
            "Exam %s transitioned %s -> %s", exam_id, expected_from, new_state,
        )
        return _row_to_exam(row)

    async def update_rubric(self, exam_id: str, rubric: dict[str, Any]) -> None:
        """Store rubric JSON on the exam row."""
        import json
        await self._session.execute(
            text("UPDATE exams SET rubric = :rubric WHERE exam_id = :eid"),
            {"eid": exam_id, "rubric": json.dumps(rubric)},
        )
        await self._session.commit()

    async def update_regions(self, exam_id: str, regions: list[dict[str, Any]]) -> None:
        """Store question region JSON on the exam row."""
        import json
        await self._session.execute(
            text("UPDATE exams SET question_regions = :regions WHERE exam_id = :eid"),
            {"eid": exam_id, "regions": json.dumps(regions)},
        )
        await self._session.commit()
