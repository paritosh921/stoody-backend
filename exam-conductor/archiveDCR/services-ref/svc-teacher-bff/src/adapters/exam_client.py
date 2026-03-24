"""Adapter for svc-exam-orch — exam lifecycle and roster data."""

from __future__ import annotations

from typing import Any

from src.adapters.http_client import BackingClients
from src.config import EXAM_ORCH_URL


async def list_exams(
    clients: BackingClients,
    token: str,
    *,
    subject_id: str | None = None,
    class_id: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch exams visible to the authenticated teacher."""
    params: dict[str, str] = {}
    if subject_id:
        params["subject_id"] = subject_id
    if class_id:
        params["class_id"] = class_id

    url = f"{EXAM_ORCH_URL}/api/v1/exams"
    data = await clients.request("GET", url, auth_token=token, params=params)
    if data is None:
        return []
    return data.get("items", [])


async def get_exam_detail(
    clients: BackingClients,
    token: str,
    exam_id: str,
) -> dict[str, Any] | None:
    """Fetch exam detail with roster from svc-exam-orch."""
    url = f"{EXAM_ORCH_URL}/api/v1/exams/{exam_id}"
    return await clients.request("GET", url, auth_token=token)


async def create_exam(
    clients: BackingClients, token: str, body: dict[str, Any],
) -> dict[str, Any] | None:
    url = f"{EXAM_ORCH_URL}/api/v1/exams"
    return await clients.request_or_raise("POST", url, auth_token=token, json=body)


async def save_rubric(
    clients: BackingClients, token: str, exam_id: str, body: dict[str, Any],
) -> dict[str, Any] | None:
    url = f"{EXAM_ORCH_URL}/api/v1/exams/{exam_id}/rubric"
    return await clients.request_or_raise("PUT", url, auth_token=token, json=body)


async def get_rubric(
    clients: BackingClients, token: str, exam_id: str,
) -> dict[str, Any] | None:
    url = f"{EXAM_ORCH_URL}/api/v1/exams/{exam_id}/rubric"
    return await clients.request("GET", url, auth_token=token)


async def save_question_regions(
    clients: BackingClients, token: str, exam_id: str, body: dict[str, Any],
) -> dict[str, Any] | None:
    url = f"{EXAM_ORCH_URL}/api/v1/exams/{exam_id}/regions"
    return await clients.request_or_raise("PUT", url, auth_token=token, json=body)


async def get_question_regions(
    clients: BackingClients, token: str, exam_id: str,
) -> dict[str, Any] | None:
    url = f"{EXAM_ORCH_URL}/api/v1/exams/{exam_id}/regions"
    return await clients.request("GET", url, auth_token=token)


async def assign_staff(
    clients: BackingClients, token: str, exam_id: str, body: dict[str, Any],
) -> dict[str, Any] | None:
    """Relay to svc-exam-orch single assignment endpoint (both invigilators + evaluators)."""
    url = f"{EXAM_ORCH_URL}/api/v1/exams/{exam_id}/invigilators"
    return await clients.request_or_raise("POST", url, auth_token=token, json=body)
