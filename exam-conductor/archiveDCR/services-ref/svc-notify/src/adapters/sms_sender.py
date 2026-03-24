"""SMS adapter (Twilio stub).

This is a pluggable stub. Replace the implementation when a real Twilio
or other SMS provider is wired up.
"""

from __future__ import annotations

import logging

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

_TWILIO_API_BASE = "https://api.twilio.com/2010-04-01"


async def send_sms(phone: str, message: str) -> bool:
    """Send an SMS via Twilio REST API.

    Returns True on success, False on failure.
    """
    if not settings.sms_twilio_account_sid or not settings.sms_twilio_auth_token:
        logger.warning("Twilio credentials not configured; skipping SMS to %s", phone)
        return False

    url = f"{_TWILIO_API_BASE}/Accounts/{settings.sms_twilio_account_sid}/Messages.json"
    data = {
        "To": phone,
        "From": settings.sms_twilio_from_number,
        "Body": message,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                url,
                data=data,
                auth=(settings.sms_twilio_account_sid, settings.sms_twilio_auth_token),
            )
            resp.raise_for_status()
        logger.info("SMS sent to %s", phone)
        return True
    except httpx.HTTPError:
        logger.exception("Failed to send SMS to %s", phone)
        return False
