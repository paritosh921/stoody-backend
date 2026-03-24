"""Push notification adapter (FCM stub).

This is a pluggable stub. Replace the implementation when a real FCM
or APNs integration is wired up.
"""

from __future__ import annotations

import logging

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


async def send_push(device_token: str, title: str, body: str) -> bool:
    """Send a push notification via FCM HTTP v1 API.

    Returns True on success, False on failure.
    """
    if not settings.push_fcm_server_key:
        logger.warning("PUSH_FCM_SERVER_KEY not configured; skipping push to %s", device_token)
        return False

    headers = {
        "Authorization": f"key={settings.push_fcm_server_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "to": device_token,
        "notification": {
            "title": title,
            "body": body,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(settings.push_fcm_api_url, json=payload, headers=headers)
            resp.raise_for_status()
        logger.info("Push sent to device %s: %s", device_token, title)
        return True
    except httpx.HTTPError:
        logger.exception("Failed to send push to device %s", device_token)
        return False
