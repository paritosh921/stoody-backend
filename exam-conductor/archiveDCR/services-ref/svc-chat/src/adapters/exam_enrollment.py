"""Adapter to fetch exam enrollment data from svc-exam-orch.

Used by chat RBAC to verify that a teacher/student belongs to the exam
before allowing messages.
"""

from __future__ import annotations

from typing import Any

import httpx

from exampen_common.logging import get_logger

_log = get_logger(__name__)


class ExamEnrollmentAdapter:
    """HTTP client for svc-exam-orch enrollment queries."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    async def get_teacher_ids(
        self, exam_id: str, token: str,
    ) -> list[str]:
        """Return teacher/tutor IDs assigned to *exam_id*."""
        url = f"{self._base_url}/api/v1/exams/{exam_id}/teachers"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    url, headers={"Authorization": f"Bearer {token}"},
                )
                resp.raise_for_status()
                data: list[dict[str, Any]] = resp.json().get("items", [])
                return [t["user_id"] for t in data]
        except Exception:
            _log.warning(
                "Failed to fetch teachers for exam=%s, "
                "falling back to empty list",
                exam_id,
            )
            return []

    async def get_student_ids(
        self, exam_id: str, teacher_id: str, token: str,
    ) -> list[str]:
        """Return student IDs enrolled under *teacher_id* for *exam_id*."""
        url = (
            f"{self._base_url}/api/v1/exams/{exam_id}"
            f"/teachers/{teacher_id}/students"
        )
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    url, headers={"Authorization": f"Bearer {token}"},
                )
                resp.raise_for_status()
                data: list[dict[str, Any]] = resp.json().get("items", [])
                return [s["user_id"] for s in data]
        except Exception:
            _log.warning(
                "Failed to fetch students for exam=%s teacher=%s, "
                "falling back to empty list",
                exam_id,
                teacher_id,
            )
            return []
