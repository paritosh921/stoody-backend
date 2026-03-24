"""Subscribe to ``stroke.raw`` NATS JetStream subject and drive the
processing pipeline: dedup -> normalize -> DB commit -> publish processed.

ACKs after successful TimescaleDB commit.  NAKs on failure so NATS
redelivers the message.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from exampen_common.logging import get_logger
from exampen_common.nats_client import NatsClient

from src.config import (
    CONSUMER_DURABLE_NAME,
    STROKE_RAW_SUBJECT,
)
from src.domain.dedup import make_idempotency_key
from src.domain.normalizer import (
    compute_bbox_mm,
    normalize_coordinates,
)
from src.domain.page_assigner import build_page_assignments
from src.events.processed_publisher import ProcessedStrokePublisher
from src.storage.stroke_repo import StrokeRepo

_log = get_logger(__name__)

# Raw stroke payload is a JSON-serialized list of stroke objects.
# Each stroke has: stroke_id, page_number, book_type, points[], ...

_COORDINATE_FRAME_BYTES = 14


class RawStrokeConsumer:
    """NATS JetStream consumer for ``stroke.raw`` events."""

    def __init__(
        self,
        nats_client: NatsClient,
        stroke_repo: StrokeRepo,
        publisher: ProcessedStrokePublisher,
    ) -> None:
        self._nats = nats_client
        self._repo = stroke_repo
        self._publisher = publisher
        self._subscription: Any = None

    async def start(self) -> None:
        """Subscribe to the ``stroke.raw`` subject."""
        self._subscription = await self._nats.subscribe(
            subject=STROKE_RAW_SUBJECT,
            durable=CONSUMER_DURABLE_NAME,
            handler=self._handle_message,
        )
        _log.info(
            "subscribed to %s (durable=%s)",
            STROKE_RAW_SUBJECT,
            CONSUMER_DURABLE_NAME,
        )

    async def stop(self) -> None:
        """Unsubscribe and drain."""
        if self._subscription is not None:
            await self._subscription.unsubscribe()
            self._subscription = None

    async def _handle_message(self, msg: Any) -> None:
        """Process a single ``stroke.raw`` event."""
        try:
            event = json.loads(msg.data)
            await self._process_event(event)
            await msg.ack()
        except Exception:
            _log.exception(
                "failed to process stroke.raw event — NAK for redelivery"
            )
            await msg.nak()

    async def _process_event(self, event: dict[str, Any]) -> None:
        """Pipeline: dedup -> decode -> normalize -> commit -> publish."""
        exam_id: str = event["exam_id"]
        pen_mac: str = event["pen_mac"]
        chunk_index: int = event["chunk_index"]

        idem_key = make_idempotency_key(exam_id, pen_mac, chunk_index)

        # DB-level dedup: check if chunk already committed
        if await self._repo.chunk_exists(idem_key):
            _log.debug("duplicate chunk skipped: %s", idem_key)
            return

        # Decode raw payload
        raw_bytes = base64.b64decode(event["payload_base64"])
        strokes = _decode_stroke_payload(raw_bytes, event)

        # Normalize coordinates per stroke
        for stroke in strokes:
            book_type = stroke.get("book_type", "LS")
            normalized = normalize_coordinates(
                stroke["points"], book_type
            )
            stroke["normalized_points"] = normalized
            stroke["bbox"] = compute_bbox_mm(normalized)

        page_assignments = build_page_assignments(strokes)

        # Atomic DB commit — returns False if this was a concurrent
        # duplicate that lost the race (ON CONFLICT DO NOTHING).
        committed = await self._repo.commit_processed_strokes(
            exam_id=exam_id,
            pen_mac=pen_mac,
            chunk_index=chunk_index,
            idempotency_key=idem_key,
            strokes=strokes,
        )

        if not committed:
            _log.debug("concurrent duplicate skipped: %s", idem_key)
            return

        # Publish stroke.processed AFTER DB commit
        await self._publisher.publish_stroke_processed(
            exam_id=exam_id,
            pen_mac=pen_mac,
            student_id=event.get("student_id"),
            page_assignments=page_assignments,
        )

        _log.info(
            "processed chunk %s: %d strokes",
            idem_key,
            len(strokes),
        )


def _decode_stroke_payload(
    raw_bytes: bytes,
    event: dict[str, Any],
) -> list[dict[str, Any]]:
    """Decode raw payload bytes into a list of stroke dicts.

    The payload is expected to be a JSON-encoded array of stroke
    objects.  Each stroke contains ``stroke_id``, ``page_number``,
    ``book_type``, and ``points`` (list of raw coordinate dicts).

    Falls back to treating the payload as raw 14-byte coordinate
    frames if JSON decoding fails (hub firmware may send binary).
    """
    try:
        parsed = json.loads(raw_bytes)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    # Fallback: binary 14-byte coordinate frames -> single stroke
    points = _parse_binary_frames(raw_bytes)
    return [
        {
            "stroke_id": f"{event['exam_id']}:{event['pen_mac']}:{event['chunk_index']}:0",
            "page_number": event.get("page_number", 0),
            "book_type": event.get("book_type", "LS"),
            "points": points,
        }
    ]


def _parse_binary_frames(data: bytes) -> list[dict[str, Any]]:
    """Parse concatenated 14-byte coordinate frames into point dicts.

    Frame layout (14 bytes):
        bookType(1), pageNo(1), X(2 BE), Y(2 BE), pressure(2 BE),
        penProp(1), timestamp(5 — truncated to 4 bytes here for int32)
    """
    points: list[dict[str, Any]] = []
    offset = 0
    while offset + _COORDINATE_FRAME_BYTES <= len(data):
        frame = data[offset : offset + _COORDINATE_FRAME_BYTES]
        x = int.from_bytes(frame[2:4], "big")
        y = int.from_bytes(frame[4:6], "big")
        pressure = int.from_bytes(frame[6:8], "big") / 1023.0
        ts = int.from_bytes(frame[8:12], "big")
        points.append({
            "x": x,
            "y": y,
            "pressure": round(pressure, 4),
            "timestamp": ts,
        })
        offset += _COORDINATE_FRAME_BYTES
    return points
