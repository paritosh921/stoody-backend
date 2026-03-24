"""NATS event consumers for svc-notify.

Subscribes to EXAMPEN.score.updated, EXAMPEN.objection.*, and
EXAMPEN.exam.lifecycle.
For each event: determine notifications -> render -> dispatch.
"""

from __future__ import annotations

import json
import logging

import nats
from nats.aio.msg import Msg

from src.adapters.dispatcher import dispatch
from src.domain.trigger_rules import determine_notifications

logger = logging.getLogger(__name__)


async def _handle_message(msg: Msg) -> None:
    """Process a single NATS message end-to-end."""
    subject = msg.subject
    try:
        payload = json.loads(msg.data.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.error("Invalid JSON on subject %s, dropping message", subject)
        return

    event_type = payload.get("event_type", subject)
    logger.info("Received event %s (id=%s)", event_type, payload.get("event_id", "?"))

    actions = determine_notifications(event_type, payload)
    if not actions:
        logger.debug("No notifications for event %s", event_type)
        return

    logger.info("Dispatching %d notifications for event %s", len(actions), event_type)
    await dispatch(actions)


async def start_consumers(nats_url: str) -> nats.NATS:
    """Connect to NATS and subscribe to notification-relevant subjects.

    Returns the NATS connection so the caller can close it on shutdown.
    """
    nc = await nats.connect(nats_url)
    logger.info("Connected to NATS at %s", nats_url)

    subjects = [
        "EXAMPEN.score.updated",
        "EXAMPEN.objection.*",
        "EXAMPEN.exam.lifecycle",
    ]

    for subj in subjects:
        await nc.subscribe(subj, cb=_handle_message)
        logger.info("Subscribed to %s", subj)

    return nc
