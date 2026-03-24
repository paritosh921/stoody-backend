"""Build ``stroke.raw`` event payloads from validated chunk data.

The event schema matches ``contracts/events/stroke.raw.schema.json``.
This module builds the event dict; the adapter performs the actual
NATS publish.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

EVENT_TYPE = "stroke.raw"
EVENT_VERSION = "1.0.0"


def build_stroke_raw_event(chunk: dict[str, Any]) -> dict[str, Any]:
    """Build a ``stroke.raw`` event payload from a validated chunk.

    Parameters
    ----------
    chunk:
        Validated ``StrokeChunkUploadRequest`` dict.

    Returns
    -------
    dict matching ``stroke.raw.schema.json``.
    """
    event: dict[str, Any] = {
        "event_id": uuid.uuid4().hex,
        "event_type": EVENT_TYPE,
        "event_version": EVENT_VERSION,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "exam_id": chunk["exam_id"],
        "pen_mac": chunk["pen_mac"],
        "chunk_index": chunk["chunk_index"],
        "total_chunks": chunk["total_chunks"],
        "payload_base64": chunk["payload_base64"],
        "checksum_crc32": chunk["checksum_crc32"],
        "upload_path": chunk["upload_path"],
    }

    binding_status = chunk.get("binding_status")
    if binding_status is not None:
        event["binding_status"] = binding_status

    return event
