"""HTTP client to check exam score workflow state via svc-score-engine.

Used to enforce the objection window — objections can only be filed
when the exam's scores are in the ``objection_window`` state.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import aiohttp

logger = logging.getLogger(__name__)


class ScoreWorkflowChecker(Protocol):
    """Protocol for checking objection window — allows test injection."""

    async def is_objection_window_open(self, exam_id: str) -> bool: ...


class ScoreEngineClient:
    """Checks score workflow state via svc-score-engine REST API."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    async def is_objection_window_open(self, exam_id: str) -> bool:
        """Query svc-score-engine for the exam's score workflow state.

        Returns True only if scores are in ``published`` or
        ``objection_window`` state.  Returns False on any error
        (fail-closed: deny objections if score engine is unreachable).
        """
        url = f"{self._base_url}/api/v1/scores/{exam_id}/workflow-state"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "Score engine returned %d for %s", resp.status, exam_id,
                        )
                        return False
                    data: dict[str, Any] = await resp.json()
                    state = data.get("lifecycle_state", "")
                    return state in ("published", "objection_window")
        except Exception:
            logger.exception("Failed to check objection window for %s", exam_id)
            return False


class AlwaysOpenChecker:
    """Test stub that always allows objections."""

    async def is_objection_window_open(self, exam_id: str) -> bool:
        return True
