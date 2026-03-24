"""JetStream stream initialization for the ExamPen event bus.

Creates (or verifies) the ``EXAMPEN`` stream that captures all
``EXAMPEN.>`` subjects.  Safe to call multiple times — if the stream
already exists with matching config it is a no-op.
"""

from __future__ import annotations

import logging
from typing import Any

from nats.js.api import StreamConfig, StorageType, RetentionPolicy

logger = logging.getLogger(__name__)

STREAM_NAME = "EXAMPEN"
STREAM_SUBJECTS = ["EXAMPEN.>"]
MAX_AGE_SECONDS = 7 * 24 * 60 * 60  # 7 days


async def ensure_exampen_stream(nats: Any) -> None:
    """Create the ``EXAMPEN`` JetStream stream if it does not exist.

    Parameters
    ----------
    nats:
        A connected :class:`NatsClient` instance (from
        ``exampen.dcr.core.nats_client``).

    The stream is configured with:
    - **Subjects**: ``EXAMPEN.>`` (all ExamPen events)
    - **Storage**: File-backed for durability
    - **Retention**: Limits-based (oldest messages discarded when limits hit)
    - **Max age**: 7 days
    """
    if nats is None:
        logger.warning("NATS not available — skipping stream setup")
        return

    js = nats.jetstream

    config = StreamConfig(
        name=STREAM_NAME,
        subjects=STREAM_SUBJECTS,
        storage=StorageType.FILE,
        retention=RetentionPolicy.LIMITS,
        max_age=MAX_AGE_SECONDS,
        description="ExamPen DCR event bus — stroke, page, AI, score pipeline",
    )

    try:
        await js.find_stream_info_by_subject("EXAMPEN.>")
        logger.info("JetStream stream '%s' already exists", STREAM_NAME)
    except Exception:
        # Stream does not exist — create it
        await js.add_stream(config)
        logger.info(
            "Created JetStream stream '%s' (subjects=%s, max_age=%ds)",
            STREAM_NAME,
            STREAM_SUBJECTS,
            MAX_AGE_SECONDS,
        )
