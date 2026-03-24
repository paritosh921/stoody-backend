"""Read-only query operations for analytics.

Separated from analytics_repo.py to stay within 300-line file limits.
All methods are read-only — no mutations to durable state.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class AnalyticsQueries:
    """Read-only queries for analytics data."""

    def __init__(self, sf: async_sessionmaker[AsyncSession]) -> None:
        self._sf = sf

    async def get_leaderboard(
        self, exam_id: str, tenant_id: str,
    ) -> list[dict[str, Any]]:
        """Fetch cached leaderboard rows for an exam."""
        async with self._sf() as session:
            await _set_tenant(session, tenant_id)
            result = await session.execute(
                text(
                    "SELECT rank, student_id, student_name, "
                    "score, percentile "
                    "FROM leaderboard_cache "
                    "WHERE exam_id = :exam_id "
                    "ORDER BY rank ASC, student_name ASC"
                ),
                {"exam_id": exam_id},
            )
            rows = result.mappings().all()
        return [
            {
                "rank": r["rank"],
                "student_id": r["student_id"],
                "student_name": r["student_name"],
                "score": float(r["score"]),
                "percentile": float(r["percentile"]),
            }
            for r in rows
        ]

    async def get_exam_scores(
        self, exam_id: str, tenant_id: str,
    ) -> list[dict[str, Any]]:
        """Fetch all score records for an exam from the local cache."""
        async with self._sf() as session:
            await _set_tenant(session, tenant_id)
            result = await session.execute(
                text(
                    "SELECT student_id, student_name, total_score "
                    "FROM exam_score_cache "
                    "WHERE exam_id = :exam_id"
                ),
                {"exam_id": exam_id},
            )
            rows = result.mappings().all()
        return [
            {
                "student_id": r["student_id"],
                "student_name": r.get("student_name", ""),
                "total_score": float(r["total_score"]),
            }
            for r in rows
        ]

    async def get_student_history(
        self, student_id: str, tenant_id: str,
    ) -> list[dict[str, Any]]:
        """Fetch cross-exam percentile history for a student."""
        async with self._sf() as session:
            await _set_tenant(session, tenant_id)
            result = await session.execute(
                text(
                    "SELECT ep.exam_id, esc.total_score AS score, "
                    "ep.percentile, ep.computed_at "
                    "FROM exam_percentiles ep "
                    "JOIN exam_score_cache esc "
                    "ON ep.exam_id = esc.exam_id "
                    "AND ep.student_id = esc.student_id "
                    "WHERE ep.student_id = :student_id "
                    "ORDER BY ep.computed_at ASC"
                ),
                {"student_id": student_id},
            )
            rows = result.mappings().all()
        return [
            {
                "exam_id": str(r["exam_id"]),
                "score": float(r["score"]),
                "percentile": float(r["percentile"]),
            }
            for r in rows
        ]

    async def get_student_exam_score(
        self, exam_id: str, student_id: str, tenant_id: str,
    ) -> dict[str, Any] | None:
        """Fetch a single student's score and percentile for an exam."""
        async with self._sf() as session:
            await _set_tenant(session, tenant_id)
            result = await session.execute(
                text(
                    "SELECT esc.total_score, ep.percentile "
                    "FROM exam_score_cache esc "
                    "LEFT JOIN exam_percentiles ep "
                    "ON esc.exam_id = ep.exam_id "
                    "AND esc.student_id = ep.student_id "
                    "WHERE esc.exam_id = :exam_id "
                    "AND esc.student_id = :student_id"
                ),
                {"exam_id": exam_id, "student_id": student_id},
            )
            row = result.mappings().first()
        if row is None:
            return None
        return {
            "score": float(row["total_score"]),
            "percentile": float(row["percentile"])
            if row["percentile"] is not None
            else None,
        }

    async def get_question_responses(
        self, exam_id: str, tenant_id: str,
    ) -> list[dict[str, Any]]:
        """Fetch question-level responses for difficulty analysis."""
        async with self._sf() as session:
            await _set_tenant(session, tenant_id)
            result = await session.execute(
                text(
                    "SELECT question_id, score, max_score, attempted "
                    "FROM question_response_cache "
                    "WHERE exam_id = :exam_id"
                ),
                {"exam_id": exam_id},
            )
            rows = result.mappings().all()
        return [
            {
                "question_id": r["question_id"],
                "score": float(r["score"]),
                "max_score": float(r["max_score"]),
                "attempted": bool(r["attempted"]),
            }
            for r in rows
        ]


async def _set_tenant(
    session: AsyncSession, tenant_id: str,
) -> None:
    """Set the RLS tenant context for the current transaction."""
    await session.execute(
        text(
            "SELECT set_config("
            "'app.current_tenant', :tid, true)"
        ),
        {"tid": tenant_id},
    )
