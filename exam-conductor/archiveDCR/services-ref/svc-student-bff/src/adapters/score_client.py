"""HTTP client for svc-score-engine — score summary and question breakdown.

Forwards the caller's auth token to the backing service so that
svc-score-engine can enforce its own RBAC.

Actual score-engine endpoints used:
  GET /api/v1/scores/{exam_id}/students/{student_id}
      -> StudentScoreDetailOut (total_score, lifecycle_state, questions[])
"""

from __future__ import annotations

from typing import Any

import aiohttp

from exampen_common.logging import get_logger

from src.config import CLIENT_RETRIES, CLIENT_TIMEOUT

_log = get_logger(__name__)


class ScoreClient:
    """Read-only client for svc-score-engine REST API."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=CLIENT_TIMEOUT)
        self._retries = CLIENT_RETRIES

    # ------------------------------------------------------------------
    # Public helpers consumed by student-bff routes
    # ------------------------------------------------------------------

    async def get_score_summary(
        self,
        exam_id: str,
        student_id: str,
        token: str,
    ) -> dict[str, Any] | None:
        """Fetch score summary for a student in an exam.

        Calls ``GET /scores/{exam_id}/students/{student_id}`` and
        reshapes the response into the ``StudentScoreView`` contract
        expected by the BFF route layer.
        """
        detail = await self._get_student_detail(exam_id, student_id, token)
        if detail is None:
            return None

        total = detail.get("total_score", 0.0)
        max_score = detail.get("max_score")
        percentage = (total / max_score * 100) if max_score else 0.0

        return {
            "exam_id": detail.get("exam_id", exam_id),
            "total_score": total,
            "percentage": round(percentage, 2),
            # Percentile requires analytics; default to 0 until enriched.
            "percentile": 0.0,
            "pass_fail": None,
            "questions": detail.get("questions", []),
        }

    async def get_question_breakdown(
        self,
        exam_id: str,
        student_id: str,
        token: str,
    ) -> list[dict[str, Any]]:
        """Fetch per-question score breakdown.

        Calls the same student-detail endpoint and extracts the
        ``questions`` list.
        """
        detail = await self._get_student_detail(exam_id, student_id, token)
        if detail is None:
            return []
        return detail.get("questions", [])

    async def get_answer_insight(
        self,
        exam_id: str,
        student_id: str,
        question_id: str,
        token: str,
    ) -> dict[str, Any] | None:
        """Fetch score data for one question.

        Calls the student-detail endpoint and filters to the requested
        question.  Full answer-image and AI-analysis fields require
        integration with svc-doc-assembly / svc-ai-pipeline (not yet
        wired); this method returns the score-engine data that is
        available today.
        """
        detail = await self._get_student_detail(exam_id, student_id, token)
        if detail is None:
            return None

        questions = detail.get("questions", [])
        match = next(
            (q for q in questions if q.get("question_id") == question_id),
            None,
        )
        if match is None:
            return None

        return {
            "question_id": question_id,
            "answer_image_uri": "",
            "recognized_text": "",
            "confidence": match.get("confidence", 0.0) or 0.0,
            "step_breakdown": [],
            "feedback": None,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _get_student_detail(
        self,
        exam_id: str,
        student_id: str,
        token: str,
    ) -> dict[str, Any] | None:
        """GET /api/v1/scores/{exam_id}/students/{student_id}."""
        url = (
            f"{self._base_url}/api/v1/scores/{exam_id}"
            f"/students/{student_id}"
        )
        return await self._get(url, token)

    async def _get(
        self, url: str, token: str,
    ) -> dict[str, Any] | None:
        """GET with retry, forwarding the caller's auth token."""
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
                    "score-engine request failed (%d/%d): %s %s",
                    attempt, self._retries, url, exc,
                )
        _log.error(
            "score-engine unreachable after %d attempts: %s",
            self._retries, last_err,
        )
        return None
