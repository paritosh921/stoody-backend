"""Adapter for svc-doc-assembly — page images and miss indicators."""

from __future__ import annotations

from typing import Any

from src.adapters.http_client import BackingClients
from src.config import DOC_ASSEMBLY_URL


async def get_answer_pages(
    clients: BackingClients,
    token: str,
    exam_id: str,
    student_id: str,
) -> list[str]:
    """Fetch answer page image URIs for a student's exam submission."""
    url = (
        f"{DOC_ASSEMBLY_URL}/api/v1/documents/{exam_id}"
        f"/students/{student_id}/pages"
    )
    data = await clients.request("GET", url, auth_token=token)
    if data is None:
        return []
    return data.get("pages", [])


async def get_miss_indicators(
    clients: BackingClients,
    token: str,
    exam_id: str,
) -> dict[str, Any] | None:
    """Fetch miss indicator matrix for an exam."""
    url = f"{DOC_ASSEMBLY_URL}/api/v1/documents/{exam_id}/miss-indicators"
    return await clients.request("GET", url, auth_token=token)
