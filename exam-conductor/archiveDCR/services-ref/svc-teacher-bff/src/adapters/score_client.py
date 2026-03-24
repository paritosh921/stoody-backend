"""Adapter for svc-score-engine — scores, overrides, finalize, publish."""

from __future__ import annotations

from typing import Any

from src.adapters.http_client import BackingClients
from src.config import SCORE_ENGINE_URL


async def get_exam_scores(
    clients: BackingClients,
    token: str,
    exam_id: str,
) -> list[dict[str, Any]]:
    """Fetch per-student score totals for an exam (class overview)."""
    url = f"{SCORE_ENGINE_URL}/api/v1/scores/{exam_id}/overview"
    data = await clients.request("GET", url, auth_token=token)
    if data is None:
        return []
    return data.get("items", [])


async def get_student_detail(
    clients: BackingClients,
    token: str,
    exam_id: str,
    student_id: str,
) -> dict[str, Any] | None:
    """Fetch per-student score breakdown."""
    url = f"{SCORE_ENGINE_URL}/api/v1/scores/{exam_id}/students/{student_id}"
    return await clients.request("GET", url, auth_token=token)


async def relay_score_override(
    clients: BackingClients,
    token: str,
    exam_id: str,
    student_id: str,
    question_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Forward a score override to svc-score-engine."""
    url = (
        f"{SCORE_ENGINE_URL}/api/v1/scores/{exam_id}"
        f"/students/{student_id}/questions/{question_id}"
    )
    return await clients.request_or_raise(
        "PATCH", url, auth_token=token, json=payload,
    )


async def relay_finalize(
    clients: BackingClients,
    token: str,
    exam_id: str,
) -> dict[str, Any]:
    """Forward finalization to svc-score-engine."""
    url = f"{SCORE_ENGINE_URL}/api/v1/scores/{exam_id}/finalize"
    return await clients.request_or_raise("POST", url, auth_token=token)


async def relay_publish(
    clients: BackingClients,
    token: str,
    exam_id: str,
) -> dict[str, Any]:
    """Forward score publication to svc-score-engine."""
    url = f"{SCORE_ENGINE_URL}/api/v1/scores/{exam_id}/publish"
    return await clients.request_or_raise("POST", url, auth_token=token)


async def relay_bulk_approve(
    clients: BackingClients,
    token: str,
    exam_id: str,
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Forward bulk score approval to svc-score-engine.

    NOTE: If svc-score-engine does not yet expose a native
    ``/bulk-approve`` endpoint, this will return a 404 from the
    backing service.  In that case the teacher-bff ``workflows``
    route should be updated to fan-out individual overrides.
    """
    url = f"{SCORE_ENGINE_URL}/api/v1/scores/{exam_id}/bulk-approve"
    return await clients.request_or_raise(
        "POST", url, auth_token=token,
        json={"decisions": decisions},
    )


async def relay_step_marking(
    clients: BackingClients,
    token: str,
    exam_id: str,
    student_id: str,
    question_id: str,
    *,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Forward step-level marking to svc-score-engine."""
    url = (
        f"{SCORE_ENGINE_URL}/api/v1/scores/{exam_id}"
        f"/students/{student_id}/questions/{question_id}/steps"
    )
    return await clients.request_or_raise(
        "PATCH", url, auth_token=token,
        json={"steps": steps},
    )
