"""Publish ``exam.lifecycle`` NATS events after FSM transitions.

Events are published AFTER PostgreSQL commit — never inside the
transaction — to prevent phantom events on rollback.

Payload matches ``contracts/events/exam.lifecycle.schema.json``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from exampen_common.logging import get_logger
from exampen_common.nats_client import NatsClient

_log = get_logger(__name__)

SUBJECT = "EXAMPEN.exam.lifecycle"
EVENT_VERSION = "1.0.0"


async def publish_lifecycle_event(
    nats: NatsClient,
    *,
    exam_id: str,
    from_state: str,
    to_state: str,
    actor_id: str,
    reason: str | None = None,
) -> None:
    """Publish a lifecycle transition event to NATS JetStream.

    Must be called AFTER the PostgreSQL transaction that performed the
    state change has been committed.
    """
    payload = {
        "event_id": str(uuid.uuid4()),
        "event_type": SUBJECT,
        "event_version": EVENT_VERSION,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "exam_id": exam_id,
        "from_state": from_state,
        "to_state": to_state,
        "actor_id": actor_id,
    }
    if reason:
        payload["reason"] = reason

    await nats.publish(SUBJECT, payload)
    _log.info(
        "Published %s: %s -> %s (exam=%s)",
        SUBJECT, from_state, to_state, exam_id,
    )
