"""Tests for the objections route — proxy with mocked backing services.

Test IDs: U-TBFF-OBJ-01 through U-TBFF-OBJ-04
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from src.config import REVIEW_URL
from tests.conftest import MockBackingClients


@pytest.mark.asyncio
async def test_list_objections(
    teacher_client: AsyncClient,
    mock_clients: MockBackingClients,
) -> None:
    """U-TBFF-OBJ-01: List objections proxies to svc-review."""
    exam_id = "exam-obj-1"

    mock_clients.responses[
        f"{REVIEW_URL}/api/v1/objections"
    ] = {
        "items": [
            {
                "objection_id": "obj-1",
                "student_id": "s1",
                "question_id": "q1",
                "status": "filed",
                "filed_at": "2026-03-15T10:00:00Z",
            },
            {
                "objection_id": "obj-2",
                "student_id": "s2",
                "question_id": "q3",
                "status": "reviewing",
                "filed_at": "2026-03-15T11:00:00Z",
            },
        ],
    }

    resp = await teacher_client.get(
        f"/api/v1/teacher/exams/{exam_id}/objections",
    )
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 2
    assert items[0]["objection_id"] == "obj-1"


@pytest.mark.asyncio
async def test_get_objection_detail(
    teacher_client: AsyncClient,
    mock_clients: MockBackingClients,
) -> None:
    """U-TBFF-OBJ-02: Objection detail returns enriched data."""
    objection_id = "obj-detail-1"

    mock_clients.responses[
        f"{REVIEW_URL}/api/v1/objections/{objection_id}"
    ] = {
        "objection_id": objection_id,
        "student_id": "s1",
        "question_id": "q2",
        "status": "reviewing",
        "student_text": "I believe my answer was correct because...",
        "current_score": 3.0,
        "answer_image_uri": "https://s3.example.com/page1.png",
    }

    resp = await teacher_client.get(
        f"/api/v1/teacher/objections/{objection_id}",
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["objection_id"] == objection_id
    assert data["current_score"] == 3.0


@pytest.mark.asyncio
async def test_objection_detail_not_found(
    teacher_client: AsyncClient,
    mock_clients: MockBackingClients,
) -> None:
    """U-TBFF-OBJ-03: Returns 404 when objection does not exist."""
    resp = await teacher_client.get(
        "/api/v1/teacher/objections/nonexistent",
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_resolve_objection_relay(
    teacher_client: AsyncClient,
    mock_clients: MockBackingClients,
) -> None:
    """U-TBFF-OBJ-04: Resolve relay forwards to svc-review."""
    objection_id = "obj-resolve-1"

    mock_clients.responses[
        f"{REVIEW_URL}/api/v1/objections/{objection_id}/resolve"
    ] = {
        "objection_id": objection_id,
        "status": "resolved",
        "verdict": "approved",
        "new_score": 5.0,
    }

    resp = await teacher_client.post(
        f"/api/v1/teacher/objections/{objection_id}/resolve",
        json={
            "verdict": "approved",
            "new_score": 5.0,
            "reason": "Student's interpretation is valid",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "resolved"
    assert data["verdict"] == "approved"
