"""HTTP client for the Stoody platform API.

Fetches user profiles and parent-child relationships.
Graceful degradation: if Stoody is unreachable, returns None
so that callers (BFF) can degrade (show user_id instead of name).
"""

from __future__ import annotations

from typing import Any

import aiohttp

from exampen_common.logging import get_logger

from src.config import STOODY_API_URL, STOODY_CLIENT_RETRIES, STOODY_CLIENT_TIMEOUT

_log = get_logger(__name__)


class StoodyClient:
    """Async HTTP client for Stoody REST endpoints."""

    def __init__(
        self,
        base_url: str = STOODY_API_URL,
        timeout: int = STOODY_CLIENT_TIMEOUT,
        retries: int = STOODY_CLIENT_RETRIES,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._retries = retries

    async def get_user_profile(self, user_id: str) -> dict[str, Any] | None:
        """Fetch user profile from ``GET /api/users/{user_id}``.

        Returns None if Stoody is unreachable or returns an error.
        """
        url = f"{self._base_url}/api/users/{user_id}"
        return await self._get_with_retry(url)

    async def get_parent_children(self, user_id: str) -> list[str] | None:
        """Fetch parent's child student IDs from ``GET /api/parents/{user_id}/children``.

        Returns None if Stoody is unreachable, or a list of child user IDs.
        """
        url = f"{self._base_url}/api/parents/{user_id}/children"
        data = await self._get_with_retry(url)
        if data is None:
            return None
        # Stoody returns {"children": [{"student_id": ...}, ...]}
        children = data.get("children", [])
        return [str(c.get("student_id", c.get("user_id", ""))) for c in children]

    async def _get_with_retry(self, url: str) -> dict[str, Any] | None:
        """GET *url* with retry and graceful degradation."""
        last_err: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=self._timeout) as session:
                    async with session.get(url) as resp:
                        if resp.status == 404:
                            _log.warning("Stoody 404: %s", url)
                            return None
                        resp.raise_for_status()
                        return await resp.json()
            except Exception as exc:
                last_err = exc
                _log.warning(
                    "Stoody request failed (attempt %d/%d): %s %s",
                    attempt,
                    self._retries,
                    url,
                    exc,
                )
        _log.error("Stoody unreachable after %d attempts: %s", self._retries, last_err)
        return None
