"""Rubric CRUD with versioning.

Every rubric edit creates a new version row.  Scores record the
``rubric_version`` at the time of scoring so that rubric changes
after partial scoring are explicit (see FAILURE_MITIGATION_REGISTER A5.5).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def get_rubric(
    session: AsyncSession,
    question_id: str,
    *,
    version: int | None = None,
) -> dict[str, Any] | None:
    """Fetch a rubric.  Latest version if *version* is ``None``."""
    if version is not None:
        result = await session.execute(
            text(
                """
                SELECT question_id, version, body, created_at
                FROM rubrics
                WHERE question_id = :qid AND version = :v
                """
            ),
            {"qid": question_id, "v": version},
        )
    else:
        result = await session.execute(
            text(
                """
                SELECT question_id, version, body, created_at
                FROM rubrics
                WHERE question_id = :qid
                ORDER BY version DESC
                LIMIT 1
                """
            ),
            {"qid": question_id},
        )
    row = result.fetchone()
    return dict(row._mapping) if row else None


async def upsert_rubric(
    session: AsyncSession,
    question_id: str,
    body: str,
) -> dict[str, Any]:
    """Insert a new rubric version.  Returns the created row."""
    now = datetime.now(timezone.utc)
    result = await session.execute(
        text(
            """
            INSERT INTO rubrics (question_id, body, created_at)
            VALUES (:qid, :body, :ts)
            RETURNING question_id, version, body, created_at
            """
        ),
        {"qid": question_id, "body": body, "ts": now},
    )
    row = result.fetchone()
    assert row is not None
    return dict(row._mapping)


async def list_rubric_versions(
    session: AsyncSession,
    question_id: str,
) -> Sequence[dict[str, Any]]:
    """Return all versions for a given question rubric."""
    result = await session.execute(
        text(
            """
            SELECT question_id, version, created_at
            FROM rubrics
            WHERE question_id = :qid
            ORDER BY version ASC
            """
        ),
        {"qid": question_id},
    )
    return [dict(row._mapping) for row in result.fetchall()]
