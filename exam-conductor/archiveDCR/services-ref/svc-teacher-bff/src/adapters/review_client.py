"""Adapter for svc-review — objection lifecycle proxy."""

from __future__ import annotations

from typing import Any

from src.adapters.http_client import BackingClients
from src.config import REVIEW_URL


async def list_objections(
    clients: BackingClients,
    token: str,
    exam_id: str,
) -> list[dict[str, Any]]:
    """Fetch objections for an exam."""
    url = f"{REVIEW_URL}/api/v1/objections"
    params = {"exam_id": exam_id}
    data = await clients.request("GET", url, auth_token=token, params=params)
    if data is None:
        return []
    return data.get("items", [])


async def get_objection_detail(
    clients: BackingClients,
    token: str,
    objection_id: str,
) -> dict[str, Any] | None:
    """Fetch a single objection with full context."""
    url = f"{REVIEW_URL}/api/v1/objections/{objection_id}"
    return await clients.request("GET", url, auth_token=token)


async def relay_resolve(
    clients: BackingClients,
    token: str,
    objection_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Forward an objection resolution to svc-review."""
    url = f"{REVIEW_URL}/api/v1/objections/{objection_id}/resolve"
    return await clients.request_or_raise(
        "POST", url, auth_token=token, json=payload,
    )


async def relay_escalate(
    clients: BackingClients,
    token: str,
    objection_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Forward an objection escalation to svc-review."""
    url = f"{REVIEW_URL}/api/v1/objections/{objection_id}/escalate"
    return await clients.request_or_raise(
        "POST", url, auth_token=token, json=payload,
    )
