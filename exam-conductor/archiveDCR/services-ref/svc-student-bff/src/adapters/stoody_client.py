"""HTTP client for Stoody platform API — parent-child resolution.

Used by the auth middleware to resolve which student IDs a parent
account may view.  Graceful degradation: returns None on failure.
"""

from __future__ import annotations

from typing import Any

import aiohttp

from exampen_common.logging import get_logger

from src.config import CLIENT_RETRIES, CLIENT_TIMEOUT

_log = get_logger(__name__)


class StoodyClient:
    """Async HTTP client for Stoody REST endpoints."""

    def __init__(
        self,
        base_url: str,
        timeout: int = CLIENT_TIMEOUT,
        retries: int = CLIENT_RETRIES,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._retries = retries

    async def get_parent_children(
        self, user_id: str,
    ) -> list[str] | None:
        """Fetch parent's child student IDs.

        Returns None if Stoody is unreachable, or a list of child
        user IDs on success.
        """
        url = f"{self._base_url}/api/parents/{user_id}/children"
        data = await self._get_with_retry(url)
        if data is None:
            return None
        children = data.get("children", [])
        return [
            str(c.get("student_id", c.get("user_id", "")))
            for c in children
        ]

    async def _get_with_retry(
        self, url: str,
    ) -> dict[str, Any] | None:
        """GET with retry and graceful degradation."""
        last_err: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                async with aiohttp.ClientSession(
                    timeout=self._timeout,
                ) as session:
                    async with session.get(url) as resp:
                        if resp.status == 404:
                            _log.warning("Stoody 404: %s", url)
                            return None
                        resp.raise_for_status()
                        return await resp.json()
            except Exception as exc:
                last_err = exc
                _log.warning(
                    "Stoody request failed (%d/%d): %s %s",
                    attempt, self._retries, url, exc,
                )
        _log.error(
            "Stoody unreachable after %d attempts: %s",
            self._retries, last_err,
        )
        return None
