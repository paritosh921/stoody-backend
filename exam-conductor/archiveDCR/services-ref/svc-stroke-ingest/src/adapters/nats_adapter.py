"""NATS JetStream publish wrapper for stroke.raw events.

Uses ``exampen_common.nats_client.NatsClient`` for the underlying
connection.  All publish calls are JetStream-acknowledged.
"""

from __future__ import annotations

from typing import Any

from exampen_common.logging import get_logger
from exampen_common.nats_client import NatsClient

from src.config import STROKE_RAW_SUBJECT

_log = get_logger(__name__)


class StrokePublisher:
    """Publish ``stroke.raw`` events to NATS JetStream."""

    def __init__(self, client: NatsClient) -> None:
        self._client = client

    async def publish_stroke_raw(self, event: dict[str, Any]) -> None:
        """Publish a single ``stroke.raw`` event.

        The underlying ``NatsClient.publish`` waits for JetStream ACK.
        Raises on publish failure so the route can return 503.
        """
        await self._client.publish(STROKE_RAW_SUBJECT, event)
        _log.debug(
            "published stroke.raw: exam=%s pen=%s chunk=%s",
            event.get("exam_id"),
            event.get("pen_mac"),
            event.get("chunk_index"),
        )
