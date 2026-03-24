"""
Shared HTTP helpers for BFF-level E2E tests (E2E-10, E2E-11, E2E-13).

Provides thin wrappers around aiohttp calls targeting teacher-bff,
student-bff, review, score-engine, exam-orch, analytics, and the
Stoody webhook mock.  Also provides a score-seeding helper that
publishes AI results and waits for the resulting ``score.updated``
NATS events.
"""

from __future__ import annotations

import asyncio
import json

from conftest import (
    ANALYTICS_URL,
    EXAM_ORCH_URL,
    REVIEW_URL,
    SCORE_ENGINE_URL,
    STOODY_WEBHOOK_URL,
    STUDENT_BFF_URL,
    TEACHER_BFF_URL,
)


# ── generic helpers ─────────────────────────────────────────────────────

async def http_post(http_session, url: str, payload: dict):
    """POST JSON and return ``(status, body|None)``."""
    async with http_session.post(url, json=payload) as resp:
        body = (
            await resp.json()
            if "application/json" in (resp.content_type or "")
            else None
        )
        return resp.status, body


async def http_get(http_session, url: str, *, headers: dict | None = None):
    """GET and return ``(status, body|None)``."""
    async with http_session.get(url, headers=headers or {}) as resp:
        body = await resp.json() if resp.status == 200 else None
        return resp.status, body


# ── service-scoped helpers ──────────────────────────────────────────────

async def teacher_get(http_session, path: str, *, token: str | None = None):
    """GET a teacher-bff endpoint."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return await http_get(
        http_session,
        f"{TEACHER_BFF_URL}{path}",
        headers=headers,
    )


async def student_get(http_session, path: str):
    """GET from a student-bff endpoint."""
    return await http_get(http_session, f"{STUDENT_BFF_URL}{path}")


async def student_post(http_session, path: str, payload: dict):
    """POST to a student-bff endpoint."""
    return await http_post(http_session, f"{STUDENT_BFF_URL}{path}", payload)


async def review_post(http_session, path: str, payload: dict):
    """POST to the review service."""
    return await http_post(http_session, f"{REVIEW_URL}{path}", payload)


async def exam_orch_post(http_session, path: str, payload: dict):
    """POST to the exam-orch service."""
    return await http_post(http_session, f"{EXAM_ORCH_URL}{path}", payload)


async def score_engine_post(http_session, path: str, payload: dict):
    """POST to the score-engine service."""
    return await http_post(
        http_session, f"{SCORE_ENGINE_URL}{path}", payload,
    )


async def analytics_get(http_session, path: str):
    """GET from the analytics service."""
    return await http_get(http_session, f"{ANALYTICS_URL}{path}")


async def webhook_get(http_session, path: str = "/received-webhooks"):
    """GET from the Stoody webhook mock."""
    return await http_get(http_session, f"{STOODY_WEBHOOK_URL}{path}")


# ── score seeding ───────────────────────────────────────────────────────

async def seed_student_score(
    publish_event,
    nats_client,
    ai_result_factory,
    *,
    exam_id: str,
    student_id: str,
    question_results: list[dict] | None = None,
    timeout: float = 30,
) -> dict | None:
    """Publish an AI result and wait for the ``ai_draft`` score event."""
    done = asyncio.Event()
    holder: list[dict] = []

    async def _handler(msg):
        data = json.loads(msg.data.decode())
        if (
            data.get("exam_id") == exam_id
            and data.get("student_id") == student_id
            and data.get("lifecycle_state") == "ai_draft"
        ):
            holder.append(data)
            done.set()

    sub = await nats_client.subscribe("score.updated", cb=_handler)
    try:
        ai_event = ai_result_factory.create_event(
            exam_id=exam_id,
            student_id=student_id,
            question_results=question_results,
        )
        await publish_event("ai.result", ai_event)
        await asyncio.wait_for(done.wait(), timeout=timeout)
        return holder[0] if holder else None
    finally:
        await sub.unsubscribe()


async def seed_scores_for_students(
    publish_event,
    nats_client,
    ai_result_factory,
    *,
    exam_id: str,
    students: list[dict],
    timeout: float = 30,
) -> list[dict]:
    """Publish AI results for multiple students and collect scores."""
    expected = len(students)
    received: list[dict] = []
    done = asyncio.Event()

    async def _handler(msg):
        data = json.loads(msg.data.decode())
        if data.get("exam_id") != exam_id:
            return
        received.append(data)
        if len(received) >= expected:
            done.set()

    sub = await nats_client.subscribe("score.updated", cb=_handler)
    try:
        for student in students:
            ai_event = ai_result_factory.create_event(
                exam_id=exam_id,
                student_id=student["id"],
            )
            await publish_event("ai.result", ai_event)
        await asyncio.wait_for(done.wait(), timeout=timeout)
        return received
    finally:
        await sub.unsubscribe()
