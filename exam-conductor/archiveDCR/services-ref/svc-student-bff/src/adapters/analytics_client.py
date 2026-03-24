"""HTTP client for svc-analytics — performance history, trends, strengths.

Reads percentiles, score history, and AI-generated strength/weakness
summaries.  All data is read-only.
"""

from __future__ import annotations

from typing import Any

import aiohttp

from exampen_common.logging import get_logger

from src.config import CLIENT_RETRIES, CLIENT_TIMEOUT

_log = get_logger(__name__)


class AnalyticsClient:
    """Read-only client for svc-analytics REST API."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=CLIENT_TIMEOUT)
        self._retries = CLIENT_RETRIES

    async def get_score_history(
        self,
        student_id: str,
        token: str,
    ) -> list[dict[str, Any]]:
        """Fetch historical scores across all exams."""
        url = (
            f"{self._base_url}/api/v1/analytics"
            f"/students/{student_id}/history"
        )
        data = await self._get(url, token)
        if data is None:
            return []
        return data.get("history", data.get("items", []))

    async def get_trends(
        self,
        student_id: str,
        token: str,
    ) -> dict[str, Any] | None:
        """Fetch trend data for charts (scores + percentiles over time)."""
        url = (
            f"{self._base_url}/api/v1/analytics"
            f"/students/{student_id}/trends"
        )
        return await self._get(url, token)

    async def get_strengths(
        self,
        student_id: str,
        token: str,
    ) -> dict[str, Any] | None:
        """Fetch AI-generated strength/weakness summary."""
        url = (
            f"{self._base_url}/api/v1/analytics"
            f"/students/{student_id}/strengths"
        )
        return await self._get(url, token)

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
                    "analytics request failed (%d/%d): %s %s",
                    attempt, self._retries, url, exc,
                )
        _log.error(
            "analytics unreachable after %d attempts: %s",
            self._retries, last_err,
        )
        return None
