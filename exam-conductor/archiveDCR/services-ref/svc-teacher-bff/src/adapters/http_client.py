"""Shared async HTTP client used by all backing-service adapters.

Creates a single ``aiohttp.ClientSession`` reused across requests.
Provides a typed ``request()`` helper that:
- Forwards the caller's Authorization header (pass-through auth)
- Returns parsed JSON on success
- Returns None on failure (graceful degradation)
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from src.config import BACKING_SERVICE_TIMEOUT

logger = logging.getLogger(__name__)


class BackingClients:
    """Manages a shared aiohttp session for outbound backing-service calls."""

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=BACKING_SERVICE_TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def request(
        self,
        method: str,
        url: str,
        *,
        auth_token: str | None = None,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """Send HTTP request to a backing service.

        Returns parsed JSON on 2xx, None on failure.
        On 4xx/5xx the caller decides whether to raise or degrade.
        """
        headers: dict[str, str] = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        session = self._get_session()
        try:
            async with session.request(
                method, url, headers=headers, json=json, params=params,
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.warning(
                        "Backing service %s %s returned %d: %s",
                        method, url, resp.status, body[:200],
                    )
                    return None
                return await resp.json()
        except Exception:
            logger.exception("Backing service request failed: %s %s", method, url)
            return None

    async def request_or_raise(
        self,
        method: str,
        url: str,
        *,
        auth_token: str | None = None,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Like request() but raises on failure instead of returning None."""
        headers: dict[str, str] = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        session = self._get_session()
        async with session.request(
            method, url, headers=headers, json=json, params=params,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                from fastapi import HTTPException
                raise HTTPException(status_code=resp.status, detail=body[:500])
            return await resp.json()
