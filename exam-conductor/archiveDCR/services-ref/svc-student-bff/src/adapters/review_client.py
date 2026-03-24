"""HTTP client for svc-review — objection read and relay.

The student BFF relays objection filing to svc-review (the writable
owner of objection state).  All reads also go through svc-review.
"""

from __future__ import annotations

from typing import Any

import aiohttp

from exampen_common.logging import get_logger

from src.config import CLIENT_RETRIES, CLIENT_TIMEOUT

_log = get_logger(__name__)


class ReviewClient:
    """Client for svc-review REST API."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=CLIENT_TIMEOUT)
        self._retries = CLIENT_RETRIES

    async def list_objections(
        self,
        token: str,
        exam_id: str | None = None,
        student_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List objections, optionally filtered by exam or student."""
        params: dict[str, str] = {}
        if exam_id:
            params["exam_id"] = exam_id
        if student_id:
            params["student_id"] = student_id
        url = f"{self._base_url}/api/v1/objections"
        data = await self._get(url, token, params=params)
        if data is None:
            return []
        return data.get("items", [])

    async def get_objection(
        self,
        objection_id: str,
        token: str,
    ) -> dict[str, Any] | None:
        """Fetch a single objection by ID."""
        url = f"{self._base_url}/api/v1/objections/{objection_id}"
        return await self._get(url, token)

    async def file_objection(
        self,
        payload: dict[str, Any],
        token: str,
    ) -> dict[str, Any] | None:
        """Relay an objection filing to svc-review (POST)."""
        url = f"{self._base_url}/api/v1/objections"
        return await self._post(url, token, payload)

    async def _get(
        self,
        url: str,
        token: str,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """GET with retry and auth forwarding."""
        headers = {"Authorization": f"Bearer {token}"}
        last_err: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                async with aiohttp.ClientSession(
                    timeout=self._timeout,
                ) as session:
                    async with session.get(
                        url, headers=headers, params=params,
                    ) as resp:
                        if resp.status == 404:
                            return None
                        resp.raise_for_status()
                        return await resp.json()
            except Exception as exc:
                last_err = exc
                _log.warning(
                    "review request failed (%d/%d): %s %s",
                    attempt, self._retries, url, exc,
                )
        _log.error(
            "review unreachable after %d attempts: %s",
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
                    "review POST failed (%d/%d): %s %s",
                    attempt, self._retries, url, exc,
                )
        _log.error(
            "review POST unreachable after %d attempts: %s",
            self._retries, last_err,
        )
        return None
