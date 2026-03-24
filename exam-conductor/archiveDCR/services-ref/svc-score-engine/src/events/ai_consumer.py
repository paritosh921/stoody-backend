"""Subscribe to ``ai.result`` NATS events and create ``ai_draft`` scores.

On receipt of an ``ai.result`` event the consumer:
1. Loads the rubric for each question.
2. Runs ``rubric_eval.evaluate()`` (pure domain, no I/O).
3. Appends an ``ai_draft_created`` score event to the event store.
4. Publishes ``score.updated`` AFTER the DB commit.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import nats
from nats.aio.client import Client as NatsClient

from src.config import settings
from src.domain.rubric_eval import Rubric, RubricStep, evaluate
from src.events.score_publisher import publish_score_updated
from src.storage.db import async_session
from src.storage.rubric_repo import get_rubric
from src.storage.score_event_store import append_event

logger = logging.getLogger(__name__)

_nc: NatsClient | None = None
_sub: Any = None


def _rubric_from_row(row: dict[str, Any]) -> Rubric:
    """Deserialise a rubric DB row into the domain model."""
    body = json.loads(row["body"]) if isinstance(row["body"], str) else row["body"]
    steps = [
        RubricStep(
            label=s["label"],
            max_marks=s["max_marks"],
            keywords=s.get("keywords", []),
        )
        for s in body.get("steps", [])
    ]
    return Rubric(
        question_id=row["question_id"],
        version=row["version"],
        steps=steps,
        negative_marking=body.get("negative_marking", False),
        negative_factor=body.get("negative_factor", 0.0),
    )


async def _handle_ai_result(msg: Any) -> None:
    """Process a single ``ai.result`` message."""
    try:
        data = json.loads(msg.data.decode())
        exam_id: str = data["exam_id"]
        student_id: str = data["student_id"]

        async with async_session() as session:
            for qr in data.get("question_results", []):
                question_id = qr["question_id"]

                rubric_row = await get_rubric(session, question_id)
                if rubric_row is None:
                    logger.warning("No rubric for question %s -- skipping", question_id)
                    continue

                rubric = _rubric_from_row(rubric_row)
                score = evaluate(qr, rubric)

                event_id = await append_event(
                    session,
                    exam_id=exam_id,
                    student_id=student_id,
                    question_id=question_id,
                    event_type="ai_draft_created",
                    old_value=None,
                    new_value=score.total_marks,
                    actor_id="ai_pipeline",
                    reason="ai_draft_created",
                    metadata={
                        "rubric_version": rubric.version,
                        "confidence": score.confidence,
                        "step_scores": [
                            {"label": s.label, "awarded": s.awarded, "max": s.max}
                            for s in score.step_scores
                        ],
                    },
                )

            # Commit all question scores for this student atomically.
            await session.commit()

        # Publish NATS events AFTER DB commit.
        for qr in data.get("question_results", []):
            await publish_score_updated(
                exam_id=exam_id,
                student_id=student_id,
                question_id=qr["question_id"],
                lifecycle_state="ai_draft",
                total_score=0.0,  # per-question, not aggregate
                previous_total_score=None,
                reason="ai_draft_created",
            )

        await msg.ack()
    except Exception:
        logger.exception("Failed to process ai.result event")
        await msg.nak()


async def start_ai_consumer() -> None:
    """Connect to NATS and subscribe to ``ai.result``."""
    global _nc, _sub
    _nc = await nats.connect(settings.nats_url)
    js = _nc.jetstream()
    _sub = await js.subscribe(
        "EXAMPEN.ai.result",
        durable=settings.nats_consumer,
        cb=_handle_ai_result,
    )
    logger.info("Subscribed to EXAMPEN.ai.result")


async def stop_ai_consumer() -> None:
    """Unsubscribe and drain."""
    global _nc, _sub
    if _sub:
        await _sub.unsubscribe()
        _sub = None
    if _nc and not _nc.is_closed:
        await _nc.drain()
        _nc = None
