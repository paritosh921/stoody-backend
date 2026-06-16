"""Transactional email delivery providers."""

import html
import logging
from typing import Optional

import httpx

from config_async import settings

logger = logging.getLogger(__name__)


async def send_password_reset_otp_email(
    *,
    to_email: str,
    otp: str,
    username: str,
    role: str,
    expire_minutes: int,
) -> bool:
    """Send a password reset OTP through the configured transactional provider."""
    provider = (getattr(settings, "EMAIL_PROVIDER", "") or "").lower().strip()
    if provider != "resend":
        logger.error("Password reset OTP requires EMAIL_PROVIDER=resend")
        return False
    return await _send_resend_password_reset_otp(
        to_email=to_email,
        otp=otp,
        username=username,
        role=role,
        expire_minutes=expire_minutes,
    )


async def _send_resend_password_reset_otp(
    *,
    to_email: str,
    otp: str,
    username: str,
    role: str,
    expire_minutes: int,
) -> bool:
    api_key = getattr(settings, "RESEND_API_KEY", "")
    if not api_key:
        logger.error("RESEND_API_KEY is not configured")
        return False

    from_address = getattr(settings, "EMAIL_FROM_ADDRESS", "noreply@stoody.in")
    from_name = getattr(settings, "EMAIL_FROM_NAME", "Stoody")
    reply_to = getattr(settings, "EMAIL_REPLY_TO", "")
    api_base = getattr(settings, "RESEND_API_BASE_URL", "https://api.resend.com").rstrip("/")

    safe_name = html.escape(username or "Stoody user")
    safe_role = html.escape(role.title())
    safe_otp = html.escape(otp)
    html_body = f"""
    <div style="font-family:Arial,sans-serif;line-height:1.5;color:#1f2937">
      <h2>Stoody password reset</h2>
      <p>Hi {safe_name},</p>
      <p>Use this one-time code to reset your {safe_role} password:</p>
      <p style="font-size:28px;font-weight:700;letter-spacing:4px">{safe_otp}</p>
      <p>This code expires in {expire_minutes} minutes. If you did not request this, ignore this email.</p>
    </div>
    """
    text_body = (
        f"Stoody password reset\n\n"
        f"Hi {username or 'Stoody user'},\n\n"
        f"Use this one-time code to reset your {role} password: {otp}\n\n"
        f"This code expires in {expire_minutes} minutes. If you did not request this, ignore this email."
    )

    payload = {
        "from": f"{from_name} <{from_address}>",
        "to": [to_email],
        "subject": "Your Stoody password reset code",
        "html": html_body,
        "text": text_body,
    }
    if reply_to:
        payload["reply_to"] = reply_to

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"{api_base}/emails",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        if response.status_code >= 400:
            logger.error("Resend OTP email failed: %s %s", response.status_code, response.text[:300])
            return False
        return True
    except Exception as exc:
        logger.error("Resend OTP email failed: %s", exc)
        return False
