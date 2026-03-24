"""Publish ``copy.ready`` events to NATS JetStream.

Event schema: contracts/events/copy.ready.schema.json
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from exampen_common.logging import get_logger
from exampen_common.nats_client import NatsClient

_log = get_logger(__name__)

_SUBJECT = "EXAMPEN.copy.ready"


async def publish_copy_ready(
    nats: NatsClient,
    *,
    exam_id: str,
    student_id: str,
    page_number: int,
    copy_image_uri: str,
) -> None:
    """Publish a ``copy.ready`` event after PG commit.

    Payload matches ``copy.ready.schema.json``.
    """
    payload = {
        "event_id": uuid.uuid4().hex,
        "event_type": "copy.ready",
        "event_version": "1.0.0",
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "exam_id": exam_id,
        "student_id": student_id,
        "page_number": page_number,
        "copy_image_uri": copy_image_uri,
        "authoritative_candidate": "copy_image",
    }
    await nats.publish(_SUBJECT, payload)
    _log.info(
        "published copy.ready exam=%s student=%s page=%d",
        exam_id, student_id, page_number,
    )
