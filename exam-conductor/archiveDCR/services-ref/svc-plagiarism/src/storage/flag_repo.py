"""PostgreSQL CRUD for plagiarism flags and teacher verdicts."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from dataclasses import dataclass

import asyncpg

from ..domain.composite_scorer import CompositeScore, Severity


# ---- row dataclass -------------------------------------------------------- #


@dataclass(slots=True)
class FlagRow:
    """Flat representation of a plagiarism_flags row."""

    flag_id: str
    exam_id: str
    student_a_id: str
    student_b_id: str
    question_id: str
    text_sim: float
    structural_sim: float
    temporal_corr: float
    proximity_score: float
    composite_score: float
    severity: str
    student_a_text: str
    student_b_text: str
    teacher_verdict: str
    verdict_reason: str | None
    verdict_by: str | None
    verdict_at: datetime | None
    created_at: datetime


def _row_to_flag(record: asyncpg.Record) -> FlagRow:
    return FlagRow(
        flag_id=str(record["flag_id"]),
        exam_id=str(record["exam_id"]),
        student_a_id=record["student_a_id"],
        student_b_id=record["student_b_id"],
        question_id=record["question_id"],
        text_sim=float(record["text_sim"]),
        structural_sim=float(record["structural_sim"]),
        temporal_corr=float(record["temporal_corr"]),
        proximity_score=float(record["proximity_score"]),
        composite_score=float(record["composite_score"]),
        severity=record["severity"],
        student_a_text=record["student_a_text"],
        student_b_text=record["student_b_text"],
        teacher_verdict=record["teacher_verdict"],
        verdict_reason=record.get("verdict_reason"),
        verdict_by=record.get("verdict_by"),
        verdict_at=record.get("verdict_at"),
        created_at=record["created_at"],
    )


# ---- repository ----------------------------------------------------------- #


class FlagRepo:
    """Async PostgreSQL operations for plagiarism flags."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def list_by_exam(self, exam_id: str) -> list[FlagRow]:
        """Return all flags for a given exam, newest first."""
        sql = """
            SELECT * FROM plagiarism_flags
            WHERE exam_id = $1
            ORDER BY composite_score DESC, created_at DESC
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, uuid.UUID(exam_id))
        return [_row_to_flag(r) for r in rows]

    async def get_by_id(self, flag_id: str) -> FlagRow | None:
        """Return a single flag by ID, or None."""
        sql = "SELECT * FROM plagiarism_flags WHERE flag_id = $1"
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, uuid.UUID(flag_id))
        return _row_to_flag(row) if row else None

    async def bulk_insert(
        self,
        flags: list[dict],
    ) -> list[str]:
        """Insert multiple flags in a single transaction.

        Each dict in *flags* must contain: exam_id, student_a_id,
        student_b_id, question_id, score (CompositeScore),
        student_a_text, student_b_text.

        Returns the list of generated flag_id strings.
        """
        sql = """
            INSERT INTO plagiarism_flags (
                flag_id, exam_id, student_a_id, student_b_id,
                question_id, text_sim, structural_sim, temporal_corr,
                proximity_score, composite_score, severity,
                student_a_text, student_b_text,
                teacher_verdict, created_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15
            )
        """
        now = datetime.now(timezone.utc)
        ids: list[str] = []
        rows_to_insert: list[tuple] = []

        for f in flags:
            flag_id = uuid.uuid4()
            score: CompositeScore = f["score"]
            severity: Severity = score.severity  # type: ignore[assignment]
            rows_to_insert.append((
                flag_id,
                uuid.UUID(f["exam_id"]),
                f["student_a_id"],
                f["student_b_id"],
                f["question_id"],
                score.text_sim,
                score.structural_sim,
                score.temporal_corr,
                score.proximity_score,
                score.composite,
                severity.value,
                f["student_a_text"],
                f["student_b_text"],
                "pending",
                now,
            ))
            ids.append(str(flag_id))

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(sql, rows_to_insert)

        return ids

    async def update_verdict(
        self,
        flag_id: str,
        teacher_id: str,
        verdict: str,
        reason: str,
    ) -> FlagRow | None:
        """Persist a teacher verdict on a flag. Returns updated row."""
        sql = """
            UPDATE plagiarism_flags
            SET teacher_verdict = $1,
                verdict_reason  = $2,
                verdict_by      = $3,
                verdict_at      = $4
            WHERE flag_id = $5
            RETURNING *
        """
        now = datetime.now(timezone.utc)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                sql, verdict, reason, teacher_id, now,
                uuid.UUID(flag_id),
            )
        return _row_to_flag(row) if row else None
