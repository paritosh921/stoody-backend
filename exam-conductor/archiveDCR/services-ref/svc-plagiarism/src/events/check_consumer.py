"""NATS consumer for plagiarism.check events.

Subscribes to ``plagiarism.check``, fetches AI-recognized texts for
every student-question pair in the exam, scores all student pairs per
question, bulk-inserts flags, and publishes ``plagiarism.result``.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from itertools import combinations

import nats
from nats.aio.client import Client as NatsClient
from nats.js import JetStreamContext

from ..domain.composite_scorer import (
    CompositeScore,
    QuestionType,
    score_pair,
)
from ..storage.flag_repo import FlagRepo

logger = logging.getLogger(__name__)


# ---- types for fetched data ----------------------------------------------- #


class _AnswerRecord:
    """Lightweight in-memory holder for a student's recognized answer."""

    __slots__ = ("student_id", "question_id", "text", "question_type")

    def __init__(
        self,
        student_id: str,
        question_id: str,
        text: str,
        question_type: str,
    ) -> None:
        self.student_id = student_id
        self.question_id = question_id
        self.text = text
        self.question_type = question_type


# ---- helpers -------------------------------------------------------------- #


def _parse_question_type(raw: str) -> QuestionType:
    """Safely map a raw question type string to the enum."""
    mapping = {
        "mcq": QuestionType.MCQ,
        "objective": QuestionType.OBJECTIVE,
    }
    return mapping.get(raw.lower(), QuestionType.SUBJECTIVE)


async def _fetch_answers(
    pool,  # asyncpg.Pool -- type not annotated to avoid import in domain
    exam_id: str,
) -> list[_AnswerRecord]:
    """Fetch AI-recognized answers for all students in an exam.

    Reads from ``ai_results`` table which is owned by svc-ai-pipeline.
    This is a read-only cross-service query (acceptable per
    STATE_OWNERSHIP_MAP -- svc-plagiarism reads AI results).
    """
    sql = """
        SELECT student_id, question_id, recognized_text, question_type
        FROM ai_results
        WHERE exam_id = $1
        ORDER BY question_id, student_id
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, exam_id)

    return [
        _AnswerRecord(
            student_id=r["student_id"],
            question_id=r["question_id"],
            text=r["recognized_text"] or "",
            question_type=r["question_type"] or "subjective",
        )
        for r in rows
    ]


def _group_by_question(
    answers: list[_AnswerRecord],
) -> dict[str, list[_AnswerRecord]]:
    """Group answers by question_id."""
    groups: dict[str, list[_AnswerRecord]] = {}
    for ans in answers:
        groups.setdefault(ans.question_id, []).append(ans)
    return groups


# ---- core check logic ----------------------------------------------------- #


def _score_all_pairs(
    question_answers: list[_AnswerRecord],
) -> list[dict]:
    """Compare every student pair for one question and return flags
    for pairs exceeding the review threshold."""
    flaggable: list[dict] = []
    question_type = _parse_question_type(question_answers[0].question_type)

    for ans_a, ans_b in combinations(question_answers, 2):
        result: CompositeScore = score_pair(
            text_a=ans_a.text,
            text_b=ans_b.text,
            question_type=question_type,
            # Temporal and proximity data would be injected from
            # exam session metadata in production.  For now pass
            # defaults (0.0) -- the composite scorer handles this
            # gracefully.
            temporal_corr=0.0,
            proximity_score=0.0,
        )

        if result.severity is not None:
            flaggable.append({
                "student_a_id": ans_a.student_id,
                "student_b_id": ans_b.student_id,
                "question_id": ans_a.question_id,
                "score": result,
                "student_a_text": ans_a.text,
                "student_b_text": ans_b.text,
            })

    return flaggable


# ---- NATS message handler ------------------------------------------------- #


async def _handle_check(
    msg,
    repo: FlagRepo,
    pool,
    js: JetStreamContext,
) -> None:
    """Process a single plagiarism.check message."""
    payload = json.loads(msg.data.decode())
    exam_id: str = payload["exam_id"]
    logger.info("plagiarism.check received for exam %s", exam_id)

    answers = await _fetch_answers(pool, exam_id)
    if not answers:
        logger.warning("No AI results found for exam %s", exam_id)
        await msg.ack()
        return

    by_question = _group_by_question(answers)
    all_flags: list[dict] = []

    for _qid, q_answers in by_question.items():
        if len(q_answers) < 2:
            continue
        flags = _score_all_pairs(q_answers)
        for f in flags:
            f["exam_id"] = exam_id
        all_flags.extend(flags)

    # Bulk insert into PostgreSQL
    flag_ids: list[str] = []
    if all_flags:
        flag_ids = await repo.bulk_insert(all_flags)
        logger.info(
            "Inserted %d plagiarism flags for exam %s",
            len(flag_ids), exam_id,
        )

    # Publish plagiarism.result event
    result_event = {
        "event_id": str(uuid.uuid4()),
        "event_type": "plagiarism.result",
        "event_version": "1.0.0",
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "exam_id": exam_id,
        "flags": [
            {
                "flag_id": fid,
                "student_a_id": f["student_a_id"],
                "student_b_id": f["student_b_id"],
                "question_id": f["question_id"],
                "composite_score": f["score"].composite,
                "severity": f["score"].severity.value,
            }
            for fid, f in zip(flag_ids, all_flags)
        ],
    }

    await js.publish(
        "EXAMPEN.plagiarism.result",
        json.dumps(result_event).encode(),
    )
    logger.info(
        "Published plagiarism.result for exam %s (%d flags)",
        exam_id, len(flag_ids),
    )

    await msg.ack()


# ---- subscription setup --------------------------------------------------- #


async def start_consumer(
    nats_url: str,
    repo: FlagRepo,
    pool,
) -> NatsClient:
    """Connect to NATS and subscribe to plagiarism.check.

    Returns the NATS client so the caller can close it on shutdown.
    """
    nc = await nats.connect(nats_url)
    js = nc.jetstream()

    # Ensure the stream exists (idempotent)
    try:
        await js.add_stream(
            name="EXAMPEN",
            subjects=["EXAMPEN.>"],
        )
    except Exception:
        # Stream may already exist with compatible config
        pass

    sub = await js.subscribe(
        "EXAMPEN.plagiarism.check",
        durable="svc-plagiarism-check",
    )

    async def _listener() -> None:
        async for msg in sub.messages:
            try:
                await _handle_check(msg, repo, pool, js)
            except Exception:
                logger.exception(
                    "Error processing plagiarism.check message"
                )
                await msg.nak()

    # Run listener in background -- caller is responsible for lifecycle
    import asyncio
    asyncio.create_task(_listener())

    return nc
