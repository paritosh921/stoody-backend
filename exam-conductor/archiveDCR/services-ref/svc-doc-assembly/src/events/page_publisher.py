"""Publish page.ready events to NATS.

Event payload matches contracts/events/page.ready.schema.json.
Published AFTER both S3 upload and PG metadata write succeed.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from nats.js import JetStreamContext

logger = logging.getLogger(__name__)

SUBJECT = "EXAMPEN.page.ready"


class PagePublisher:
    """Publishes page.ready events to NATS JetStream."""

    def __init__(self, js: JetStreamContext) -> None:
        self._js = js

    async def publish_page_ready(
        self,
        exam_id: str,
        student_id: str,
        page_id: str,
        page_number: int,
        image_uri: str,
        vector_uri: str,
        question_ids: list[str] | None = None,
    ) -> None:
        """Publish a page.ready event.

        Payload conforms to page.ready.schema.json v1.0.0.
        """
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "page.ready",
            "event_version": "1.0.0",
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "exam_id": exam_id,
            "student_id": student_id,
            "page_id": page_id,
            "page_number": page_number,
            "image_uri": image_uri,
            "vector_uri": vector_uri,
            "authoritative_source": "strokes",
        }

        if question_ids:
            event["question_ids"] = question_ids

        payload = json.dumps(event).encode("utf-8")

        ack = await self._js.publish(SUBJECT, payload)
        logger.info(
            "Published page.ready: page_id=%s exam=%s page=%d seq=%d",
            page_id,
            exam_id,
            page_number,
            ack.seq,
        )
