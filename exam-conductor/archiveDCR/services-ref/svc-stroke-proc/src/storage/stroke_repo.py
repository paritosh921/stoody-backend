"""TimescaleDB repository for processed strokes.

Provides atomic batch commit per pen per exam and idempotency-key
based dedup via ``INSERT ... ON CONFLICT (idempotency_key) DO NOTHING``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from exampen_common.logging import get_logger

_log = get_logger(__name__)


class StrokeRepo:
    """TimescaleDB write/read for processed strokes.

    Uses raw ``asyncpg`` connections for precise transaction control
    and ``ON CONFLICT DO NOTHING`` for race-safe idempotency.
    """

    def __init__(self, database_url: str) -> None:
        self._dsn = _to_asyncpg_dsn(database_url)
        self._pool: Any = None  # asyncpg.Pool

    async def connect(self) -> None:
        """Create the asyncpg connection pool."""
        import asyncpg  # noqa: local import — storage layer

        self._pool = await asyncpg.create_pool(dsn=self._dsn, min_size=2, max_size=10)
        _log.info("TimescaleDB pool connected")

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def chunk_exists(self, idempotency_key: str) -> bool:
        """Check if a chunk with this idempotency key is already committed."""
        row = await self._pool.fetchrow(
            "SELECT 1 FROM processed_strokes WHERE idempotency_key = $1 LIMIT 1",
            idempotency_key,
        )
        return row is not None

    async def get_strokes(
        self,
        exam_id: str,
        pen_mac: str,
    ) -> list[dict[str, Any]]:
        """Read all processed strokes for a given pen in an exam."""
        rows = await self._pool.fetch(
            """
            SELECT stroke_id, page_number, question_id, book_type,
                   normalized_points, created_at
            FROM processed_strokes
            WHERE exam_id = $1 AND pen_mac = $2
            ORDER BY created_at ASC
            """,
            exam_id,
            pen_mac,
        )
        return [
            {
                "stroke_id": r["stroke_id"],
                "page_number": r["page_number"],
                "question_id": r["question_id"],
                "book_type": r["book_type"],
                "normalized_points": json.loads(r["normalized_points"]),
                "created_at": r["created_at"].isoformat(),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def commit_processed_strokes(
        self,
        exam_id: str,
        pen_mac: str,
        chunk_index: int,
        idempotency_key: str,
        strokes: list[dict[str, Any]],
    ) -> bool:
        """Atomic batch commit: all strokes in one transaction.

        Uses ``INSERT ... ON CONFLICT (idempotency_key, stroke_id) DO
        NOTHING`` so that a re-delivered chunk (same key + same
        stroke_ids) is safely rejected, while multiple strokes within a
        single chunk (same key, different stroke_ids) all insert.

        Returns ``True`` if any strokes were inserted (new chunk),
        ``False`` if every stroke was a duplicate (chunk re-delivery).
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                now = datetime.now(timezone.utc)
                total_inserted = 0
                for stroke in strokes:
                    result = await conn.execute(
                        """
                        INSERT INTO processed_strokes (
                            exam_id, pen_mac, chunk_index,
                            idempotency_key, stroke_id,
                            page_number, question_id,
                            normalized_points, book_type,
                            created_at
                        ) VALUES (
                            $1, $2, $3, $4, $5,
                            $6, $7, $8, $9, $10
                        )
                        ON CONFLICT (idempotency_key, stroke_id) DO NOTHING
                        """,
                        exam_id,
                        pen_mac,
                        chunk_index,
                        idempotency_key,
                        stroke.get("stroke_id", ""),
                        stroke.get("page_number", 0),
                        stroke.get("question_id"),
                        json.dumps(stroke.get("normalized_points", [])),
                        stroke.get("book_type", "LS"),
                        now,
                    )
                    # asyncpg returns 'INSERT 0 1' or 'INSERT 0 0'
                    if result and result.endswith("0 1"):
                        total_inserted += 1

                if total_inserted == 0:
                    _log.debug(
                        "dedup — already committed: %s",
                        idempotency_key,
                    )
                    return False

                _log.info(
                    "committed %d strokes for %s",
                    total_inserted,
                    idempotency_key,
                )
                return True


def _to_asyncpg_dsn(url: str) -> str:
    """Convert SQLAlchemy-style URL to plain asyncpg DSN.

    Strips the ``+asyncpg`` driver suffix if present so the URL can
    be passed directly to ``asyncpg.create_pool``.
    """
    return url.replace("postgresql+asyncpg://", "postgresql://")
