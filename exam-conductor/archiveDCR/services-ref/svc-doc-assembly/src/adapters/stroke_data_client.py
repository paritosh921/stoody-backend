"""Client to fetch processed stroke data from svc-stroke-proc's TimescaleDB.

The stroke.processed event contains only page_assignments (page_number,
question_id, point_count) and a normalized_stroke_uri pointing to
TimescaleDB.  This adapter resolves the URI by querying the stroke
database directly for the actual normalized point data.

Design note: In production this would be a REST call to svc-stroke-proc's
read API.  For V1 we use a shared-DB read (TimescaleDB) since both
services are co-deployed and the read path is documented in
STATE_OWNERSHIP_MAP as allowed for svc-doc-assembly.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class StrokeDataSource(Protocol):
    """Protocol for fetching stroke data — allows test injection."""

    async def get_strokes_for_page(
        self,
        exam_id: str,
        pen_mac: str,
        page_number: int,
    ) -> list[dict[str, Any]]:
        """Return stroke dicts with normalized_points for a specific page."""
        ...


class TimescaleStrokeClient:
    """Fetches stroke data from TimescaleDB (svc-stroke-proc's database)."""

    def __init__(self, pool: Any) -> None:
        """Accept an asyncpg connection pool."""
        self._pool = pool

    async def get_strokes_for_page(
        self,
        exam_id: str,
        pen_mac: str,
        page_number: int,
    ) -> list[dict[str, Any]]:
        """Query processed_strokes for a specific exam/pen/page."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT stroke_id, page_number, question_id,
                       normalized_points, book_type
                FROM processed_strokes
                WHERE exam_id = $1 AND pen_mac = $2 AND page_number = $3
                ORDER BY created_at
                """,
                exam_id,
                pen_mac,
                page_number,
            )
        return [
            {
                "stroke_id": row["stroke_id"],
                "page_number": row["page_number"],
                "question_id": row["question_id"],
                "normalized_points": (
                    json.loads(row["normalized_points"])
                    if isinstance(row["normalized_points"], str)
                    else row["normalized_points"]
                ),
                "book_type": row["book_type"],
            }
            for row in rows
        ]


class NullStrokeClient:
    """Fallback that returns empty data — used when DB is unavailable."""

    async def get_strokes_for_page(
        self,
        exam_id: str,
        pen_mac: str,
        page_number: int,
    ) -> list[dict[str, Any]]:
        logger.warning(
            "NullStrokeClient: no stroke data for exam=%s pen=%s page=%d",
            exam_id, pen_mac, page_number,
        )
        return []
