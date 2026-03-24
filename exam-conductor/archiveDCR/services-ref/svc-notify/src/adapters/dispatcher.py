"""Route NotificationActions to the correct channel adapter.

Retry with exponential backoff on transient failures.
Dead-letter logging for permanently failed notifications.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.config import settings
from src.domain.templates import RenderedNotification, render_template
from src.domain.trigger_rules import (
    NotificationAction,
    NotificationChannel,
    WebhookAction,
)

from . import email_sender, push_sender, sms_sender
from .webhook_sender import get_sender as get_webhook_sender

logger = logging.getLogger(__name__)


async def _send_via_email(action: NotificationAction, rendered: RenderedNotification) -> bool:
    return await email_sender.send_email(
        to=action.recipient_id,
        subject=rendered.subject,
        body_html=rendered.body_html,
        body_text=rendered.body_text,
    )


async def _send_via_push(action: NotificationAction, rendered: RenderedNotification) -> bool:
    return await push_sender.send_push(
        device_token=action.recipient_id,
        title=rendered.subject,
        body=rendered.body_text,
    )


async def _send_via_sms(action: NotificationAction, rendered: RenderedNotification) -> bool:
    return await sms_sender.send_sms(
        phone=action.recipient_id,
        message=rendered.body_text,
    )


_CHANNEL_SENDERS: dict[str, Any] = {
    "email": _send_via_email,
    "push": _send_via_push,
    "sms": _send_via_sms,
}


async def _dispatch_webhook(action: WebhookAction) -> bool:
    """Dispatch a webhook action to the StoodyWebhookSender.

    The sender handles its own retry logic, so we do NOT apply the
    generic retry loop here.  Returns True on success, False on failure.
    Failures are never raised.
    """
    sender = get_webhook_sender()
    wh_type = action.webhook_type
    data = action.webhook_data

    try:
        if wh_type == "score":
            return await sender.send_score_webhook(
                exam_id=data.get("exam_id", ""),
                scores=data.get("scores", []),
            )
        elif wh_type == "exam_created":
            return await sender.send_exam_webhook(
                exam_id=data.get("exam_id", ""),
                status="created",
                data=data,
            )
        elif wh_type == "exam_completed":
            return await sender.send_exam_webhook(
                exam_id=data.get("exam_id", ""),
                status="completed",
                data=data,
            )
        else:
            logger.error("Unknown webhook_type %r", wh_type)
            return False
    except Exception:
        logger.exception("Unhandled error dispatching webhook %s", wh_type)
        return False


async def _dispatch_one(action: NotificationAction | WebhookAction) -> bool:
    """Dispatch a single notification with exponential backoff retries.

    WebhookActions are handled by the dedicated webhook path which has
    its own retry logic built into the sender.
    """
    # Webhook actions bypass the template-render + generic-retry path
    if isinstance(action, WebhookAction):
        return await _dispatch_webhook(action)

    rendered = render_template(action.template_name, action.template_data)
    sender = _CHANNEL_SENDERS.get(action.channel)
    if sender is None:
        logger.error("Unknown channel %r for action %s", action.channel, action)
        return False

    max_retries = settings.dispatch_max_retries
    base_delay = settings.dispatch_base_delay_s

    for attempt in range(max_retries):
        success = await sender(action, rendered)
        if success:
            return True
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "Retry %d/%d for %s/%s in %.1fs",
                attempt + 1,
                max_retries,
                action.channel,
                action.template_name,
                delay,
            )
            await asyncio.sleep(delay)

    # All retries exhausted — dead-letter
    logger.error(
        "DEAD_LETTER: permanently failed notification channel=%s template=%s recipient=%s",
        action.channel,
        action.template_name,
        action.recipient_id,
    )
    return False


async def dispatch(actions: list[NotificationAction | WebhookAction]) -> None:
    """Dispatch a batch of notifications concurrently.

    Failures are logged but never raised — notification dispatch must not
    block the event pipeline.
    """
    if not actions:
        return

    tasks = [asyncio.create_task(_dispatch_one(a)) for a in actions]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for action, result in zip(actions, results):
        if isinstance(result, Exception):
            logger.error(
                "DEAD_LETTER: exception dispatching channel=%s template=%s: %s",
                action.channel,
                action.template_name,
                result,
            )
