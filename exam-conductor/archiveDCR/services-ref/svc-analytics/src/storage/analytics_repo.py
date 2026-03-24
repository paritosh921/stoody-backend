"""PostgreSQL write operations for analytics materialized views.

svc-analytics is the ONLY writer of percentile data (per STATE_OWNERSHIP_MAP.md).
Scores are read from svc-score-engine's published events -- never written here.

Write operations live here; read-only queries are in analytics_queries.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from exampen_common.logging import get_logger

from src.storage.analytics_queries import AnalyticsQueries, _set_tenant

_log = get_logger(__name__)


class AnalyticsRepo(AnalyticsQueries):
    """Read/write operations for analytics materialized views.

    Inherits all read operations from AnalyticsQueries.
    Adds write methods for percentiles, leaderboard, and score cache.
    """

    def __init__(self, sf: async_sessionmaker[AsyncSession]) -> None:
        super().__init__(sf)

    # -- Write: percentiles ------------------------------------------------

    async def upsert_percentiles(
        self,
        exam_id: str,
        percentiles: dict[str, float],
        tenant_id: str,
    ) -> None:
        """Replace all percentile rows for an exam (idempotent).

        Uses DELETE + INSERT in a single transaction so recomputation
        always produces a clean, consistent state.
        """
        now = datetime.now(timezone.utc)
        async with self._sf() as session:
            await _set_tenant(session, tenant_id)
            await session.execute(
                text(
                    "DELETE FROM exam_percentiles "
                    "WHERE exam_id = :exam_id"
                ),
                {"exam_id": exam_id},
            )
            for student_id, percentile in percentiles.items():
                await session.execute(
                    text(
                        "INSERT INTO exam_percentiles "
                        "(id, exam_id, student_id, percentile, "
                        "tenant_id, computed_at) "
                        "VALUES (:id, :exam_id, :student_id, "
                        ":percentile, :tenant_id, :computed_at)"
                    ),
                    {
                        "id": str(uuid4()),
                        "exam_id": exam_id,
                        "student_id": student_id,
                        "percentile": percentile,
                        "tenant_id": tenant_id,
                        "computed_at": now,
                    },
                )
            await session.commit()
        _log.info(
            "Percentiles upserted exam=%s count=%d",
            exam_id, len(percentiles),
        )

    # -- Write: leaderboard cache ------------------------------------------

    async def upsert_leaderboard(
        self,
        exam_id: str,
        rows: list[dict[str, Any]],
        tenant_id: str,
    ) -> None:
        """Replace all leaderboard rows for an exam (idempotent)."""
        now = datetime.now(timezone.utc)
        async with self._sf() as session:
            await _set_tenant(session, tenant_id)
            await session.execute(
                text(
                    "DELETE FROM leaderboard_cache "
                    "WHERE exam_id = :exam_id"
                ),
                {"exam_id": exam_id},
            )
            for row in rows:
                await session.execute(
                    text(
                        "INSERT INTO leaderboard_cache "
                        "(id, exam_id, student_id, student_name, "
                        "rank, score, percentile, "
                        "tenant_id, computed_at) "
                        "VALUES (:id, :exam_id, :student_id, "
                        ":student_name, :rank, :score, "
                        ":percentile, :tenant_id, :computed_at)"
                    ),
                    {
                        "id": str(uuid4()),
                        "exam_id": exam_id,
                        "student_id": row["student_id"],
                        "student_name": row.get("student_name", ""),
                        "rank": row["rank"],
                        "score": row["score"],
                        "percentile": row["percentile"],
                        "tenant_id": tenant_id,
                        "computed_at": now,
                    },
                )
            await session.commit()
        _log.info(
            "Leaderboard upserted exam=%s rows=%d",
            exam_id, len(rows),
        )

    # -- Write: score cache (from events) ----------------------------------

    async def upsert_score_cache(
        self,
        exam_id: str,
        student_id: str,
        total_score: float,
        tenant_id: str,
    ) -> None:
        """Upsert a student's score into the local event-driven cache."""
        now = datetime.now(timezone.utc)
        async with self._sf() as session:
            await _set_tenant(session, tenant_id)
            await session.execute(
                text(
                    "INSERT INTO exam_score_cache "
                    "(exam_id, student_id, total_score, "
                    "tenant_id, updated_at) "
                    "VALUES (:exam_id, :student_id, "
                    ":total_score, :tenant_id, :updated_at) "
                    "ON CONFLICT (exam_id, student_id) "
                    "DO UPDATE SET total_score = :total_score, "
                    "updated_at = :updated_at"
                ),
                {
                    "exam_id": exam_id,
                    "student_id": student_id,
                    "total_score": total_score,
                    "tenant_id": tenant_id,
                    "updated_at": now,
                },
            )
            await session.commit()
