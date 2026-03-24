"""Pure-logic upload progress tracking and reconciliation.

ZERO I/O -- this module must never import asyncio, aiohttp, sqlalchemy,
nats, redis, or any I/O library.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PenProgress:
    """Upload progress for a single pen within an exam."""

    pen_mac: str
    total_chunks: int
    acked_chunks: frozenset[int] = field(default_factory=frozenset)

    @property
    def complete(self) -> bool:
        """True when every chunk index has been acknowledged."""
        if self.total_chunks <= 0:
            return False
        return len(self.acked_chunks) >= self.total_chunks

    @property
    def missing_chunks(self) -> list[int]:
        """Return sorted list of chunk indices not yet acknowledged."""
        expected = set(range(self.total_chunks))
        return sorted(expected - set(self.acked_chunks))

    @property
    def next_expected_chunk(self) -> int:
        """Lowest chunk index not yet acknowledged (or total if done)."""
        missing = self.missing_chunks
        return missing[0] if missing else self.total_chunks


@dataclass(slots=True)
class ExamUploadTracker:
    """In-memory tracker for per-pen upload progress within an exam.

    Intended to be built from DB/cache state and used for reconciliation
    logic.  Does NOT perform any I/O itself.
    """

    exam_id: str
    _pens: dict[str, PenProgress] = field(default_factory=dict)

    def record_ack(
        self,
        pen_mac: str,
        chunk_index: int,
        total_chunks: int,
    ) -> PenProgress:
        """Record that a chunk was acknowledged for a pen.

        Returns the updated :class:`PenProgress`.
        """
        existing = self._pens.get(pen_mac)
        if existing is None:
            acked = frozenset({chunk_index})
        else:
            acked = existing.acked_chunks | {chunk_index}
            # Use the max total_chunks seen (handles late corrections)
            total_chunks = max(total_chunks, existing.total_chunks)

        progress = PenProgress(
            pen_mac=pen_mac,
            total_chunks=total_chunks,
            acked_chunks=acked,
        )
        self._pens[pen_mac] = progress
        return progress

    def get_pen_progress(self, pen_mac: str) -> PenProgress | None:
        """Return progress for a specific pen, or ``None``."""
        return self._pens.get(pen_mac)

    def all_pens(self) -> list[PenProgress]:
        """Return progress records for all pens."""
        return list(self._pens.values())

    def all_complete(self) -> bool:
        """True when every tracked pen has all chunks acknowledged."""
        if not self._pens:
            return False
        return all(p.complete for p in self._pens.values())

    def reconciliation_summary(self) -> list[dict]:
        """Build a per-pen reconciliation list suitable for API response.

        Returns a list of dicts matching the ``PenUploadStatus`` schema.
        """
        result = []
        for pen in self._pens.values():
            result.append({
                "pen_mac": pen.pen_mac,
                "acked_chunks": sorted(pen.acked_chunks),
                "total_chunks": pen.total_chunks,
                "complete": pen.complete,
            })
        return result
