"""Adapter for svc-chat — teacher/student messaging proxy."""

from __future__ import annotations

from typing import Any

from src.adapters.http_client import BackingClients
from src.config import CHAT_URL


async def get_thread(
    clients: BackingClients,
    token: str,
    exam_id: str,
    student_id: str,
) -> list[dict[str, Any]]:
    """Fetch chat thread between teacher and student for an exam."""
    url = f"{CHAT_URL}/api/v1/chat/{exam_id}/{student_id}"
    data = await clients.request("GET", url, auth_token=token)
    if data is None:
        return []
    return data.get("items", [])


async def relay_message(
    clients: BackingClients,
    token: str,
    exam_id: str,
    student_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Forward a new message to svc-chat."""
    url = f"{CHAT_URL}/api/v1/chat/{exam_id}/{student_id}"
    return await clients.request_or_raise(
        "POST", url, auth_token=token, json=payload,
    )
