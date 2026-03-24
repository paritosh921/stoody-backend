"""HTTP client for Stoody platform APIs.

Fetches student roster, tutor list, and class/subject reference data.
All data is read-only from ExamPen's perspective.
"""

from __future__ import annotations

from typing import Any

import aiohttp

from exampen_common.logging import get_logger

_log = get_logger(__name__)


class StoodyClient:
    """Thin async wrapper around Stoody REST endpoints."""

    def __init__(self, base_url: str, timeout: int = 5) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)

    # -- Student roster ----------------------------------------------------

    async def get_students(
        self, class_id: str, section_id: str,
    ) -> list[dict[str, Any]]:
        """Fetch student roster for a class/section.

        Returns list of ``{student_id, name, roll, section_id}``.
        """
        url = f"{self._base}/api/students"
        params = {"class_id": class_id, "section_id": section_id}
        return await self._get_list(url, params)

    # -- Tutor list --------------------------------------------------------

    async def get_tutors(
        self, subject_id: str,
    ) -> list[dict[str, Any]]:
        """Fetch tutors for a given subject."""
        url = f"{self._base}/api/tutors"
        params = {"subject_id": subject_id}
        return await self._get_list(url, params)

    # -- Reference data ----------------------------------------------------

    async def get_classes(self) -> list[dict[str, Any]]:
        """Fetch all classes."""
        return await self._get_list(f"{self._base}/api/classes")

    async def get_subjects(self) -> list[dict[str, Any]]:
        """Fetch all subjects."""
        return await self._get_list(f"{self._base}/api/subjects")

    # -- Internal ----------------------------------------------------------

    async def _get_list(
        self,
        url: str,
        params: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform a GET and return a JSON list."""
        try:
            async with aiohttp.ClientSession(
                timeout=self._timeout,
            ) as session:
                async with session.get(url, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    # Stoody may wrap in {items: [...]} or return bare list
                    if isinstance(data, list):
                        return data
                    return data.get("items", data.get("data", []))
        except aiohttp.ClientError:
            _log.exception("Stoody request failed: %s", url)
            return []
