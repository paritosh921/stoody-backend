"""SMTP email adapter using aiosmtplib."""

from __future__ import annotations

import logging
from email.message import EmailMessage

import aiosmtplib

from src.config import settings

logger = logging.getLogger(__name__)


async def send_email(
    to: str,
    subject: str,
    body_html: str,
    body_text: str,
) -> bool:
    """Send an email via SMTP. Returns True on success, False on failure."""
    msg = EmailMessage()
    msg["From"] = settings.smtp_from_email
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body_text)
    msg.add_alternative(body_html, subtype="html")

    try:
        await aiosmtplib.send(
            msg,
            hostname=settings.smtp_host,
            port=settings.smtp_port,
            username=settings.smtp_username or None,
            password=settings.smtp_password or None,
            use_tls=settings.smtp_use_tls,
        )
        logger.info("Email sent to %s: %s", to, subject)
        return True
    except aiosmtplib.SMTPException:
        logger.exception("Failed to send email to %s", to)
        return False
