"""HTTP client to svc-exam-orch for exam session data.

All data is read-only from svc-invig-console's perspective.
This service never writes exam state.

Sync progress and dongle health come from the hub via NATS relay
(``hub_relay.py``), NOT from exam-orch.  This adapter only fetches
exam metadata.
"""

from __future__ import annotations

from typing import Any

import aiohttp

from exampen_common.logging import get_logger

_log = get_logger(__name__)


class ExamOrchClient:
    """Thin async wrapper around svc-exam-orch REST endpoints."""

    def __init__(self, base_url: str, timeout: int = 5) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)

    # -- Session data -------------------------------------------------------

    async def get_session(
        self, exam_id: str, token: str,
    ) -> dict[str, Any]:
        """Fetch the current backend view of an exam session.

        Returns a dict matching the exam-orch ``GET /exams/{exam_id}``
        response, or an empty dict on error.
        """
        url = f"{self._base}/api/v1/exams/{exam_id}"
        return await self._get_json(url, token=token)

    async def list_active_sessions(
        self, token: str,
    ) -> list[dict[str, Any]]:
        """Fetch all active exam sessions from the orchestrator.

        Returns a list of session summaries, or an empty list on error.
        """
        url = f"{self._base}/api/v1/exams"
        params = {"state": "active"}
        data = await self._get_json(url, params=params, token=token)
        if isinstance(data, list):
            return data
        return data.get("items", data.get("data", []))

    # -- Internal -----------------------------------------------------------

    async def _get_json(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
        token: str = "",
    ) -> Any:
        """Perform a GET and return deserialized JSON."""
        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            async with aiohttp.ClientSession(
                timeout=self._timeout,
            ) as session:
                async with session.get(
                    url, params=params, headers=headers,
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except aiohttp.ClientError:
            _log.exception("exam-orch request failed: %s", url)
            return {}
