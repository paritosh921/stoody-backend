"""Stoody webhook delivery adapter.

Sends score and exam lifecycle webhooks to the Stoody platform with
HMAC-SHA256 authentication and exponential-backoff retry.

Webhook failures are logged but never raised — they must not block the
notification pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from typing import Any

import aiohttp

from src.config import settings

logger = logging.getLogger(__name__)

_SIGNATURE_HEADER = "X-ExamPen-Signature"
_CONTENT_TYPE = "application/json"


def _compute_hmac(body: bytes, secret: str) -> str:
    """Return hex-encoded HMAC-SHA256 of *body* using *secret*."""
    return hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()


class StoodyWebhookSender:
    """Delivers webhooks to the Stoody platform.

    Each public method builds the payload, signs it with HMAC-SHA256, and
    POSTs to the appropriate Stoody endpoint.  On transient failure the
    request is retried with exponential backoff (default 2s / 4s / 8s).
    After all retries are exhausted the full payload is logged at ERROR
    level for dead-letter investigation.
    """

    def __init__(
        self,
        base_url: str | None = None,
        secret: str | None = None,
        max_retries: int | None = None,
        base_delay: float | None = None,
    ) -> None:
        self._base_url = (base_url or settings.stoody_webhook_url).rstrip("/")
        self._secret = secret or settings.stoody_webhook_secret
        self._max_retries = (
            max_retries if max_retries is not None
            else settings.stoody_webhook_max_retries
        )
        self._base_delay = (
            base_delay if base_delay is not None
            else settings.stoody_webhook_base_delay_s
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send_score_webhook(
        self,
        exam_id: str,
        scores: list[dict[str, Any]],
    ) -> bool:
        """POST score publication payload to Stoody.

        *scores* is a list of dicts each containing at minimum:
        ``student_id``, ``total``, ``percentage``, ``percentile``.
        """
        url = f"{self._base_url}/api/webhooks/exampen/scores"
        payload = {"exam_id": exam_id, "scores": scores}
        return await self._deliver(url, payload, tag="scores")

    async def send_exam_webhook(
        self,
        exam_id: str,
        status: str,
        data: dict[str, Any],
    ) -> bool:
        """POST exam lifecycle payload to Stoody.

        *status* is ``"created"`` or ``"completed"``.
        *data* carries the event-specific fields:
          - created:   subject_id, class_id, date, duration
          - completed: pens_synced, upload_status
        """
        url = f"{self._base_url}/api/webhooks/exampen/exams"
        payload: dict[str, Any] = {"exam_id": exam_id}

        if status == "created":
            payload.update({
                "subject_id": data.get("subject_id", ""),
                "class_id": data.get("class_id", ""),
                "date": data.get("date", ""),
                "duration": data.get("duration", 0),
            })
        elif status == "completed":
            payload.update({
                "status": "completed",
                "pens_synced": data.get("pens_synced", 0),
                "upload_status": data.get("upload_status", ""),
            })
        else:
            logger.warning("Unknown exam webhook status %r, sending raw data", status)
            payload["status"] = status
            payload.update(data)

        return await self._deliver(url, payload, tag="exams")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _deliver(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        tag: str,
    ) -> bool:
        """POST *payload* to *url* with HMAC signature and retry logic."""
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        signature = _compute_hmac(body, self._secret) if self._secret else ""

        headers: dict[str, str] = {"Content-Type": _CONTENT_TYPE}
        if signature:
            headers[_SIGNATURE_HEADER] = signature

        for attempt in range(self._max_retries):
            success = await self._post(url, body, headers, attempt, tag)
            if success:
                return True
            if attempt < self._max_retries - 1:
                delay = self._base_delay * (2 ** attempt)
                logger.warning(
                    "Webhook retry %d/%d for %s in %.1fs",
                    attempt + 1,
                    self._max_retries,
                    tag,
                    delay,
                )
                await asyncio.sleep(delay)

        # All retries exhausted — dead-letter log
        logger.error(
            "DEAD_LETTER: webhook permanently failed url=%s tag=%s payload=%s",
            url,
            tag,
            payload,
        )
        return False

    async def _post(
        self,
        url: str,
        body: bytes,
        headers: dict[str, str],
        attempt: int,
        tag: str,
    ) -> bool:
        """Execute a single HTTP POST.  Returns True on 2xx."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data=body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if 200 <= resp.status < 300:
                        logger.info(
                            "Webhook %s delivered (%d) attempt=%d",
                            tag,
                            resp.status,
                            attempt + 1,
                        )
                        return True
                    logger.warning(
                        "Webhook %s returned HTTP %d on attempt %d",
                        tag,
                        resp.status,
                        attempt + 1,
                    )
                    return False
        except (aiohttp.ClientError, asyncio.TimeoutError):
            logger.exception(
                "Webhook %s network error on attempt %d",
                tag,
                attempt + 1,
            )
            return False


# Module-level singleton, lazily initialised on first use.
_sender: StoodyWebhookSender | None = None


def get_sender() -> StoodyWebhookSender:
    """Return (or create) the module-level StoodyWebhookSender."""
    global _sender  # noqa: PLW0603
    if _sender is None:
        _sender = StoodyWebhookSender()
    return _sender
