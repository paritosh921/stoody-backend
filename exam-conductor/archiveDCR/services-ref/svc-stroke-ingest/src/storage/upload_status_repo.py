"""PostgreSQL-backed upload progress tracking.

Table ``upload_progress`` records one row per acknowledged chunk.
Queries aggregate per-pen status for the reconciliation endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from exampen_common.logging import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PenProgress:
    """Persisted upload progress snapshot for a single pen."""

    pen_mac: str
    total_chunks: int
    received_indices: frozenset[int]

    @property
    def total_received(self) -> int:
        return len(self.received_indices)

    @property
    def complete(self) -> bool:
        if self.total_chunks <= 0:
            return False
        return self.total_received >= self.total_chunks

    @property
    def missing_chunks(self) -> list[int]:
        expected = set(range(self.total_chunks))
        return sorted(expected - set(self.received_indices))

    @property
    def next_expected_chunk(self) -> int:
        missing = self.missing_chunks
        return missing[0] if missing else self.total_chunks


class UploadStatusRepo:
    """Track per-pen per-exam upload progress in PostgreSQL."""

    def __init__(self, database_url: str) -> None:
        self._url = database_url
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    async def connect(self) -> None:
        """Create the async engine and session factory."""
        self._engine = create_async_engine(
            self._url,
            pool_size=5,
            max_overflow=10,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
        )
        _log.info("Upload status DB connected: %s", self._url)

    async def close(self) -> None:
        """Dispose the engine."""
        if self._engine is not None:
            await self._engine.dispose()

    async def record_chunk(
        self,
        exam_id: str,
        pen_mac: str,
        chunk_index: int,
        total_chunks: int,
    ) -> None:
        """Upsert a received chunk record.

        Uses ``ON CONFLICT DO NOTHING`` so duplicate inserts are safe.
        """
        assert self._session_factory is not None
        async with self._session_factory() as session:
            await session.execute(
                text("""
                    INSERT INTO upload_progress
                        (exam_id, pen_mac, chunk_index, total_chunks, received_at)
                    VALUES
                        (:exam_id, :pen_mac, :chunk_index, :total_chunks, :received_at)
                    ON CONFLICT (exam_id, pen_mac, chunk_index) DO NOTHING
                """),
                {
                    "exam_id": exam_id,
                    "pen_mac": pen_mac,
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "received_at": datetime.now(timezone.utc),
                },
            )
            await session.commit()

    async def get_pen_progress(
        self,
        exam_id: str,
        pen_mac: str,
    ) -> PenProgress | None:
        """Return persisted upload progress for a single pen.

        Returns ``None`` if no chunks have been recorded for this pen.
        """
        assert self._session_factory is not None
        async with self._session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT
                        MAX(total_chunks)                AS total_chunks,
                        ARRAY_AGG(chunk_index ORDER BY chunk_index) AS indices
                    FROM upload_progress
                    WHERE exam_id = :exam_id AND pen_mac = :pen_mac
                """),
                {"exam_id": exam_id, "pen_mac": pen_mac},
            )
            row = result.fetchone()

        if row is None or row.total_chunks is None:
            return None

        return PenProgress(
            pen_mac=pen_mac,
            total_chunks=row.total_chunks,
            received_indices=frozenset(row.indices or []),
        )

    async def get_exam_status(
        self,
        exam_id: str,
    ) -> list[dict[str, Any]]:
        """Return per-pen upload status for an exam.

        Each dict matches the ``PenUploadStatus`` schema:
        ``{pen_mac, acked_chunks, total_chunks, complete}``.
        """
        assert self._session_factory is not None
        async with self._session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT
                        pen_mac,
                        MAX(total_chunks)                AS total_chunks,
                        ARRAY_AGG(chunk_index ORDER BY chunk_index) AS acked_chunks
                    FROM upload_progress
                    WHERE exam_id = :exam_id
                    GROUP BY pen_mac
                    ORDER BY pen_mac
                """),
                {"exam_id": exam_id},
            )
            rows = result.fetchall()

        pens = []
        for row in rows:
            total = row.total_chunks
            acked = row.acked_chunks or []
            expected = set(range(total))
            missing = sorted(expected - set(acked))
            pens.append({
                "pen_mac": row.pen_mac,
                "acked_chunks": acked,
                "missing_chunks": missing,
                "total_chunks": total,
                "complete": len(acked) >= total,
            })
        return pens
