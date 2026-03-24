"""Publish ``score.updated`` events to NATS JetStream.

Events are published AFTER the DB commit succeeds.  If NATS publish
fails, the event is already durable in PostgreSQL -- an async retry
worker can pick it up later.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import nats
from nats.aio.client import Client as NatsClient

from src.config import settings

_nc: NatsClient | None = None
_js: Any = None  # JetStreamContext


async def _ensure_connected() -> Any:
    """Return a JetStream context, connecting lazily."""
    global _nc, _js
    if _nc is None or _nc.is_closed:
        _nc = await nats.connect(settings.nats_url)
        _js = _nc.jetstream()
    return _js


async def publish_score_updated(
    *,
    exam_id: str,
    student_id: str,
    question_id: str | None,
    lifecycle_state: str,
    total_score: float,
    previous_total_score: float | None,
    reason: str,
) -> None:
    """Publish a ``score.updated`` event matching the contract schema."""
    js = await _ensure_connected()

    payload = {
        "event_id": str(uuid.uuid4()),
        "event_type": "score.updated",
        "event_version": "1.0.0",
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "exam_id": exam_id,
        "student_id": student_id,
        "lifecycle_state": lifecycle_state,
        "total_score": total_score,
        "previous_total_score": previous_total_score,
        "reason": reason,
    }
    if question_id:
        payload["question_id"] = question_id

    await js.publish(
        "EXAMPEN.score.updated",
        json.dumps(payload).encode(),
    )


async def close() -> None:
    """Drain and close the NATS connection."""
    global _nc
    if _nc and not _nc.is_closed:
        await _nc.drain()
        _nc = None
