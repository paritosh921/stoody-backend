"""Adapter for svc-analytics — leaderboards, class stats, question analysis."""

from __future__ import annotations

from typing import Any

from src.adapters.http_client import BackingClients
from src.config import ANALYTICS_URL


async def get_leaderboard(
    clients: BackingClients,
    token: str,
    exam_id: str,
) -> dict[str, Any] | None:
    """Fetch leaderboard for an exam."""
    url = f"{ANALYTICS_URL}/api/v1/analytics/{exam_id}/leaderboard"
    return await clients.request("GET", url, auth_token=token)


async def get_class_stats(
    clients: BackingClients,
    token: str,
    exam_id: str,
) -> dict[str, Any] | None:
    """Fetch class-level statistics for an exam."""
    url = f"{ANALYTICS_URL}/api/v1/analytics/{exam_id}/class-stats"
    return await clients.request("GET", url, auth_token=token)


async def get_question_analysis(
    clients: BackingClients,
    token: str,
    exam_id: str,
) -> dict[str, Any] | None:
    """Fetch per-question analysis for an exam."""
    url = f"{ANALYTICS_URL}/api/v1/analytics/{exam_id}/questions"
    return await clients.request("GET", url, auth_token=token)


async def trigger_export(
    clients: BackingClients,
    token: str,
    exam_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Trigger an exam results export via svc-analytics.

    The payload typically includes ``format`` (pdf/csv/xlsx) and
    ``include_answer_sheets`` (bool).  The analytics service returns
    a job reference that can be polled for completion.
    """
    url = f"{ANALYTICS_URL}/api/v1/analytics/{exam_id}/export"
    return await clients.request_or_raise(
        "POST", url, auth_token=token, json=payload,
    )
