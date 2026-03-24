"""PostgreSQL storage for AI recognition results.

Key rule from STATE_OWNERSHIP_MAP: model version stored with every result.
Re-running AI with a new model creates a new version row — old versions
are NEVER overwritten.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import TYPE_CHECKING

import asyncpg

if TYPE_CHECKING:
    from src.domain.result_builder import AIResult

logger = logging.getLogger(__name__)


class ResultRepo:
    """Manages ai_results rows in PostgreSQL."""

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    async def _connect(self) -> asyncpg.Connection:
        return await asyncpg.connect(self._database_url)

    async def store_result(self, result: AIResult) -> None:
        """INSERT a new AI result row. Never UPDATE existing rows.

        Each invocation creates a new row keyed by
        (event_id, model_version). Re-runs with a different model
        version produce a new row; old rows are preserved.
        """
        conn = await self._connect()
        try:
            await conn.execute(
                """
                INSERT INTO ai_results (
                    event_id,
                    exam_id,
                    student_id,
                    model_version,
                    source_type,
                    question_results,
                    occurred_at
                ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
                """,
                result.event_id,
                result.exam_id,
                result.student_id,
                result.model_version,
                result.source_type,
                json.dumps(
                    [asdict(qr) for qr in result.question_results]
                ),
                result.occurred_at,
            )
            logger.info(
                "Stored ai_result event_id=%s model=%s",
                result.event_id,
                result.model_version,
            )
        finally:
            await conn.close()

    async def get_results(
        self,
        exam_id: str,
        student_id: str,
    ) -> list[dict]:
        """Fetch all AI result rows for a student in an exam.

        Returns all model versions, ordered newest first.
        """
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT
                    event_id,
                    exam_id,
                    student_id,
                    model_version,
                    source_type,
                    question_results,
                    occurred_at
                FROM ai_results
                WHERE exam_id = $1 AND student_id = $2
                ORDER BY occurred_at DESC
                """,
                exam_id,
                student_id,
            )
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    async def get_latest_result(
        self,
        exam_id: str,
        student_id: str,
    ) -> dict | None:
        """Fetch the most recent AI result for a student in an exam."""
        conn = await self._connect()
        try:
            row = await conn.fetchrow(
                """
                SELECT
                    event_id,
                    exam_id,
                    student_id,
                    model_version,
                    source_type,
                    question_results,
                    occurred_at
                FROM ai_results
                WHERE exam_id = $1 AND student_id = $2
                ORDER BY occurred_at DESC
                LIMIT 1
                """,
                exam_id,
                student_id,
            )
            return dict(row) if row else None
        finally:
            await conn.close()
