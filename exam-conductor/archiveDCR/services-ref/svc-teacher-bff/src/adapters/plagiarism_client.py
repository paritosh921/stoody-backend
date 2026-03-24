"""Adapter for svc-plagiarism — plagiarism flags and teacher verdicts."""

from __future__ import annotations

from typing import Any

from src.adapters.http_client import BackingClients
from src.config import PLAGIARISM_URL


async def list_flags(
    clients: BackingClients,
    token: str,
    exam_id: str,
) -> list[dict[str, Any]]:
    """Fetch plagiarism flags for an exam."""
    url = f"{PLAGIARISM_URL}/api/v1/plagiarism/{exam_id}/flags"
    data = await clients.request("GET", url, auth_token=token)
    if data is None:
        return []
    return data.get("items", [])


async def relay_verdict(
    clients: BackingClients,
    token: str,
    flag_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Forward a teacher verdict to svc-plagiarism."""
    url = f"{PLAGIARISM_URL}/api/v1/plagiarism/flags/{flag_id}/verdict"
    return await clients.request_or_raise(
        "PATCH", url, auth_token=token, json=payload,
    )
