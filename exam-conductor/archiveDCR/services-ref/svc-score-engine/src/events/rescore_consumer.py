"""Subscribe to ``EXAMPEN.score.rescore_command`` and apply objection re-scores.

On receipt of a re-score command (published by svc-review when an
objection is approved):

1. Load the current materialised score for the question.
2. Append an ``objection_rescored`` score event with old/new values.
3. Publish ``score.updated`` AFTER the DB commit.

The command payload carries the ``new_score`` already approved by the
evaluator/HOD — svc-score-engine trusts it and records the change.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import nats
from nats.aio.client import Client as NatsClient

from src.config import settings
from src.events.score_publisher import publish_score_updated
from src.storage.db import async_session
from src.storage.score_event_store import append_event, get_current_scores

logger = logging.getLogger(__name__)

RESCORE_SUBJECT = "EXAMPEN.score.rescore_command"
_DURABLE_NAME = "score-engine-rescore"

_nc: NatsClient | None = None
_sub: Any = None


async def _handle_rescore_command(msg: Any) -> None:
    """Process a single ``rescore_command`` message."""
    try:
        data = json.loads(msg.data.decode())
        exam_id: str = data["exam_id"]
        student_id: str = data["student_id"]
        question_id: str = data["question_id"]
        new_score: float = float(data["new_score"])
        actor_id: str = data["actor_id"]
        objection_id: str = data["objection_id"]

        old_value: float | None = None

        async with async_session() as session:
            # Fetch current score so we can record the delta.
            rows = await get_current_scores(session, exam_id, student_id)
            current_row = next(
                (r for r in rows if r["question_id"] == question_id),
                None,
            )
            if current_row is not None:
                old_value = current_row["current_score"]

            await append_event(
                session,
                exam_id=exam_id,
                student_id=student_id,
                question_id=question_id,
                event_type="objection_rescored",
                old_value=old_value,
                new_value=new_score,
                actor_id=actor_id,
                reason="objection_approved",
                metadata={
                    "objection_id": objection_id,
                    "source": "svc-review",
                },
            )
            await session.commit()

        # Publish NATS event AFTER DB commit.
        await publish_score_updated(
            exam_id=exam_id,
            student_id=student_id,
            question_id=question_id,
            lifecycle_state="objection_rescored",
            total_score=new_score,
            previous_total_score=old_value,
            reason="objection_approved",
        )

        logger.info(
            "Rescored question=%s student=%s exam=%s old=%s new=%s objection=%s",
            question_id, student_id, exam_id, old_value, new_score, objection_id,
        )
        await msg.ack()
    except Exception:
        logger.exception("Failed to process rescore_command event")
        await msg.nak()


async def start_rescore_consumer() -> None:
    """Connect to NATS and subscribe to rescore commands."""
    global _nc, _sub
    _nc = await nats.connect(settings.nats_url)
    js = _nc.jetstream()
    _sub = await js.subscribe(
        RESCORE_SUBJECT,
        durable=_DURABLE_NAME,
        cb=_handle_rescore_command,
    )
    logger.info("Subscribed to %s", RESCORE_SUBJECT)


async def stop_rescore_consumer() -> None:
    """Unsubscribe and drain."""
    global _nc, _sub
    if _sub:
        await _sub.unsubscribe()
        _sub = None
    if _nc and not _nc.is_closed:
        await _nc.drain()
        _nc = None
