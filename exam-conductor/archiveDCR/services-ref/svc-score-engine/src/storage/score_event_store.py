"""Event-sourced score storage.

Score events are APPENDED -- never updated in place.
A materialised view (``score_materialized``) is updated atomically
within the same DB transaction for fast reads.

Only ``svc-score-engine`` writes to these tables.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def append_event(
    session: AsyncSession,
    *,
    exam_id: str,
    student_id: str,
    question_id: str | None,
    event_type: str,
    old_value: float | None,
    new_value: float,
    actor_id: str,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Append a score event and refresh the materialised view.

    Both writes happen within the caller's transaction boundary.
    Returns the generated ``event_id``.
    """
    event_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    await session.execute(
        text(
            """
            INSERT INTO score_events
                (event_id, exam_id, student_id, question_id,
                 event_type, old_value, new_value, actor_id,
                 reason, metadata, created_at)
            VALUES
                (:eid, :exam, :stu, :qid,
                 :etype, :old, :new, :actor,
                 :reason, :meta, :ts)
            """
        ),
        {
            "eid": event_id,
            "exam": exam_id,
            "stu": student_id,
            "qid": question_id,
            "etype": event_type,
            "old": old_value,
            "new": new_value,
            "actor": actor_id,
            "reason": reason,
            "meta": str(metadata) if metadata else None,
            "ts": now,
        },
    )

    # Upsert materialised view row atomically in same transaction.
    await session.execute(
        text(
            """
            INSERT INTO score_materialized
                (exam_id, student_id, question_id,
                 current_score, lifecycle_state, rubric_version,
                 updated_at)
            VALUES
                (:exam, :stu, :qid,
                 :score, :state, :rv, :ts)
            ON CONFLICT (exam_id, student_id, question_id)
            DO UPDATE SET
                current_score   = EXCLUDED.current_score,
                lifecycle_state = EXCLUDED.lifecycle_state,
                updated_at      = EXCLUDED.updated_at
            """
        ),
        {
            "exam": exam_id,
            "stu": student_id,
            "qid": question_id or "__exam__",
            "score": new_value,
            "state": event_type,
            "rv": (metadata or {}).get("rubric_version"),
            "ts": now,
        },
    )

    return event_id


async def get_current_scores(
    session: AsyncSession,
    exam_id: str,
    student_id: str,
) -> Sequence[dict[str, Any]]:
    """Return latest materialised scores for one student in an exam."""
    result = await session.execute(
        text(
            """
            SELECT question_id, current_score, lifecycle_state,
                   rubric_version, updated_at
            FROM score_materialized
            WHERE exam_id = :exam AND student_id = :stu
            ORDER BY question_id
            """
        ),
        {"exam": exam_id, "stu": student_id},
    )
    return [dict(row._mapping) for row in result.fetchall()]


async def get_event_history(
    session: AsyncSession,
    exam_id: str,
    student_id: str,
) -> Sequence[dict[str, Any]]:
    """Return full event log for one student in an exam."""
    result = await session.execute(
        text(
            """
            SELECT event_id, event_type, question_id,
                   old_value, new_value, actor_id,
                   reason, created_at
            FROM score_events
            WHERE exam_id = :exam AND student_id = :stu
            ORDER BY created_at ASC
            """
        ),
        {"exam": exam_id, "stu": student_id},
    )
    return [dict(row._mapping) for row in result.fetchall()]


async def get_exam_overview(
    session: AsyncSession,
    exam_id: str,
) -> Sequence[dict[str, Any]]:
    """Return per-student score totals for all students in an exam.

    Aggregates the materialised view grouped by student, excluding
    the ``__exam__`` sentinel rows.
    """
    result = await session.execute(
        text(
            """
            SELECT student_id,
                   SUM(current_score) AS total_score,
                   COUNT(*)           AS question_count,
                   MAX(lifecycle_state) AS lifecycle_state
            FROM score_materialized
            WHERE exam_id = :exam
              AND question_id != '__exam__'
              AND student_id  != '__all__'
            GROUP BY student_id
            ORDER BY student_id
            """
        ),
        {"exam": exam_id},
    )
    return [dict(row._mapping) for row in result.fetchall()]


async def get_exam_lifecycle_state(
    session: AsyncSession,
    exam_id: str,
) -> str | None:
    """Return the current lifecycle state for the exam (from latest event)."""
    result = await session.execute(
        text(
            """
            SELECT lifecycle_state
            FROM score_materialized
            WHERE exam_id = :exam AND question_id = '__exam__'
            LIMIT 1
            """
        ),
        {"exam": exam_id},
    )
    row = result.fetchone()
    return row._mapping["lifecycle_state"] if row else None
