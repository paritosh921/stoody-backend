"""
Email Service for SkillBot Backend

Handles email sending for:
- Password reset links
- Account verification
- Notifications

Uses SMTP for reliability and supports both sync and async sending.
"""

import logging
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List
import asyncio
from concurrent.futures import ThreadPoolExecutor

from config_async import (
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USER,
    SMTP_PASSWORD,
    SMTP_FROM_EMAIL,
    SMTP_FROM_NAME,
    SMTP_USE_TLS,
    DEBUG_MODE,
)

logger = logging.getLogger(__name__)

# Thread pool for async email sending
_email_executor = ThreadPoolExecutor(max_workers=3)


class EmailService:
    """
    Email service for sending transactional emails.

    Features:
    - Async email sending (non-blocking)
    - HTML and plain text support
    - TLS encryption
    - Graceful fallback for development
    """

    def __init__(
        self,
        host: str = SMTP_HOST,
        port: int = SMTP_PORT,
        username: str = SMTP_USER,
        password: str = SMTP_PASSWORD,
        from_email: str = SMTP_FROM_EMAIL,
        from_name: str = SMTP_FROM_NAME,
        use_tls: bool = SMTP_USE_TLS,
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.from_name = from_name
        self.use_tls = use_tls
        self._configured = bool(host and username and password)

    @property
    def is_configured(self) -> bool:
        """Check if email service is properly configured"""
        return self._configured

    def _send_email_sync(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None,
    ) -> bool:
        """
        Send email synchronously (internal use).

        Returns True if successful, False otherwise.
        """
        if not self._configured:
            if DEBUG_MODE:
                logger.info(
                    f"📧 [DEV] Would send email to {to_email}\n"
                    f"   Subject: {subject}\n"
                    f"   Body preview: {html_body[:200]}..."
                )
                return True
            logger.error("Email service not configured - cannot send email")
            return False

        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = f"{self.from_name} <{self.from_email}>"
            message["To"] = to_email

            # Add plain text part (fallback)
            if text_body:
                part1 = MIMEText(text_body, "plain")
                message.attach(part1)

            # Add HTML part
            part2 = MIMEText(html_body, "html")
            message.attach(part2)

            # Send email
            context = ssl.create_default_context()

            if self.use_tls:
                with smtplib.SMTP(self.host, self.port) as server:
                    server.starttls(context=context)
                    server.login(self.username, self.password)
                    server.sendmail(self.from_email, to_email, message.as_string())
            else:
                with smtplib.SMTP_SSL(self.host, self.port, context=context) as server:
                    server.login(self.username, self.password)
                    server.sendmail(self.from_email, to_email, message.as_string())

            logger.info(f"✅ Email sent successfully to {to_email}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to send email to {to_email}: {str(e)}")
            return False

    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None,
    ) -> bool:
        """
        Send email asynchronously (non-blocking).

        Returns True if successful, False otherwise.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _email_executor,
            self._send_email_sync,
            to_email,
            subject,
            html_body,
            text_body,
        )

    async def send_password_reset_email(
        self,
        to_email: str,
        reset_token: str,
        username: str,
        frontend_url: str = "https://app.stoody.in",
        expire_minutes: int = 30,
    ) -> bool:
        """
        Send password reset email with secure token link.

        Args:
            to_email: Recipient email address
            reset_token: Secure reset token
            username: User's username for personalization
            frontend_url: Frontend URL for reset link
            expire_minutes: Token expiry time for display

        Returns:
            True if email sent successfully
        """
        reset_link = f"{frontend_url}/reset-password?token={reset_token}"

        subject = "Reset Your Stoody Password"

        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; text-align: center; border-radius: 10px 10px 0 0;">
                <h1 style="color: white; margin: 0;">Stoody</h1>
            </div>

            <div style="background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px;">
                <h2 style="color: #333; margin-top: 0;">Password Reset Request</h2>

                <p>Hi <strong>{username}</strong>,</p>

                <p>We received a request to reset your password. Click the button below to create a new password:</p>

                <div style="text-align: center; margin: 30px 0;">
                    <a href="{reset_link}"
                       style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                              color: white;
                              padding: 15px 30px;
                              text-decoration: none;
                              border-radius: 5px;
                              font-weight: bold;
                              display: inline-block;">
                        Reset Password
                    </a>
                </div>

                <p style="color: #666; font-size: 14px;">
                    <strong>⏰ This link expires in {expire_minutes} minutes.</strong>
                </p>

                <p style="color: #666; font-size: 14px;">
                    If you didn't request this password reset, you can safely ignore this email.
                    Your password will remain unchanged.
                </p>

                <hr style="border: none; border-top: 1px solid #ddd; margin: 30px 0;">

                <p style="color: #999; font-size: 12px;">
                    If the button doesn't work, copy and paste this link into your browser:<br>
                    <a href="{reset_link}" style="color: #667eea; word-break: break-all;">{reset_link}</a>
                </p>
            </div>

            <div style="text-align: center; margin-top: 20px; color: #999; font-size: 12px;">
                <p>
                    This is an automated message from Stoody.<br>
                    Please do not reply to this email.
                </p>
            </div>
        </body>
        </html>
        """

        text_body = f"""
Password Reset Request

Hi {username},

We received a request to reset your password.

Click this link to reset your password:
{reset_link}

This link expires in {expire_minutes} minutes.

If you didn't request this password reset, you can safely ignore this email.

- The Stoody Team
        """

        return await self.send_email(to_email, subject, html_body, text_body)

    async def send_password_changed_notification(
        self,
        to_email: str,
        username: str,
    ) -> bool:
        """
        Send notification that password was changed.

        This is a security measure to alert users if their password
        was changed without their knowledge.
        """
        subject = "Your Stoody Password Was Changed"

        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
        </head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: #4CAF50; padding: 20px; text-align: center; border-radius: 10px 10px 0 0;">
                <h1 style="color: white; margin: 0;">✓ Password Changed</h1>
            </div>

            <div style="background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px;">
                <p>Hi <strong>{username}</strong>,</p>

                <p>Your Stoody password was successfully changed.</p>

                <p style="color: #666;">
                    If you made this change, you can ignore this email.
                </p>

                <p style="color: #d32f2f; font-weight: bold;">
                    If you did NOT change your password, please contact support immediately
                    as your account may have been compromised.
                </p>
            </div>
        </body>
        </html>
        """

        text_body = f"""
Password Changed

Hi {username},

Your Stoody password was successfully changed.

If you made this change, you can ignore this email.

If you did NOT change your password, please contact support immediately
as your account may have been compromised.

- The Stoody Team
        """

        return await self.send_email(to_email, subject, html_body, text_body)


# Global email service instance
_email_service: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """Get configured email service instance"""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
