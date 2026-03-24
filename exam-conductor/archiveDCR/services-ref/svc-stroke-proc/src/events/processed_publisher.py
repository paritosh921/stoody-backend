"""Publish ``stroke.processed`` events to NATS JetStream.

Event payload matches ``contracts/events/stroke.processed.schema.json``.
Published AFTER successful TimescaleDB commit -- never before.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from exampen_common.logging import get_logger
from exampen_common.nats_client import NatsClient

from src.config import STROKE_PROCESSED_SUBJECT

_log = get_logger(__name__)

EVENT_TYPE = "stroke.processed"
EVENT_VERSION = "1.0.0"


class ProcessedStrokePublisher:
    """Publish ``stroke.processed`` events to NATS JetStream."""

    def __init__(self, client: NatsClient) -> None:
        self._client = client

    async def publish_stroke_processed(
        self,
        exam_id: str,
        pen_mac: str,
        student_id: str | None,
        page_assignments: list[dict[str, Any]],
    ) -> None:
        """Build and publish a ``stroke.processed`` event.

        Parameters
        ----------
        exam_id:
            Exam session UUID.
        pen_mac:
            Pen MAC address.
        student_id:
            Optional student identifier (may be ``None`` if binding
            is still provisional).
        page_assignments:
            List of ``{page_number, question_id, point_count}`` dicts.
        """
        event: dict[str, Any] = {
            "event_id": uuid.uuid4().hex,
            "event_type": EVENT_TYPE,
            "event_version": EVENT_VERSION,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "exam_id": exam_id,
            "pen_mac": pen_mac,
            "normalized_stroke_uri": (
                f"timescaledb://strokes/{exam_id}/{pen_mac}"
            ),
            "page_assignments": page_assignments,
        }

        if student_id is not None:
            event["student_id"] = student_id

        await self._client.publish(STROKE_PROCESSED_SUBJECT, event)
        _log.debug(
            "published stroke.processed: exam=%s pen=%s pages=%d",
            exam_id,
            pen_mac,
            len(page_assignments),
        )
