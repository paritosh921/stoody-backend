"""HTTP client for svc-chat — message thread read and send relay.

svc-chat is the single writable owner of chat messages (append-only).
The BFF reads threads and relays new message posts.
"""

from __future__ import annotations

from typing import Any

import aiohttp

from exampen_common.logging import get_logger

from src.config import CLIENT_RETRIES, CLIENT_TIMEOUT

_log = get_logger(__name__)


class ChatClient:
    """Client for svc-chat REST API."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=CLIENT_TIMEOUT)
        self._retries = CLIENT_RETRIES

    async def get_thread(
        self,
        exam_id: str,
        student_id: str,
        teacher_id: str,
        token: str,
    ) -> list[dict[str, Any]]:
        """Fetch chat messages for a student-teacher thread."""
        url = (
            f"{self._base_url}/api/v1/chat"
            f"/exams/{exam_id}/threads/{student_id}/{teacher_id}"
        )
        data = await self._get(url, token)
        if data is None:
            return []
        return data.get("items", data.get("messages", []))

    async def send_message(
        self,
        exam_id: str,
        student_id: str,
        teacher_id: str,
        payload: dict[str, Any],
        token: str,
    ) -> dict[str, Any] | None:
        """Relay a message send to svc-chat (POST)."""
        url = (
            f"{self._base_url}/api/v1/chat"
            f"/exams/{exam_id}/threads/{student_id}/{teacher_id}"
        )
        return await self._post(url, token, payload)

    async def _get(
        self, url: str, token: str,
    ) -> dict[str, Any] | None:
        """GET with retry and auth forwarding."""
        headers = {"Authorization": f"Bearer {token}"}
        last_err: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                async with aiohttp.ClientSession(
                    timeout=self._timeout,
                ) as session:
                    async with session.get(url, headers=headers) as resp:
                        if resp.status == 404:
                            return None
                        resp.raise_for_status()
                        return await resp.json()
            except Exception as exc:
                last_err = exc
                _log.warning(
                    "chat request failed (%d/%d): %s %s",
                    attempt, self._retries, url, exc,
                )
        _log.error(
            "chat unreachable after %d attempts: %s",
            self._retries, last_err,
        )
        return None

    async def _post(
        self,
        url: str,
        token: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        """POST with retry and auth forwarding."""
        headers = {"Authorization": f"Bearer {token}"}
        last_err: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                async with aiohttp.ClientSession(
                    timeout=self._timeout,
                ) as session:
                    async with session.post(
                        url, json=payload, headers=headers,
                    ) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except Exception as exc:
                last_err = exc
                _log.warning(
                    "chat POST failed (%d/%d): %s %s",
                    attempt, self._retries, url, exc,
                )
        _log.error(
            "chat POST unreachable after %d attempts: %s",
            self._retries, last_err,
        )
        return None
