"""Tests for the scores route — class overview aggregation.

Test IDs: U-TBFF-SCR-01 through U-TBFF-SCR-04
Validates that the BFF correctly aggregates data from svc-score-engine,
svc-doc-assembly (miss indicators), and svc-plagiarism into a merged response.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from src.config import DOC_ASSEMBLY_URL, PLAGIARISM_URL, SCORE_ENGINE_URL
from tests.conftest import MockBackingClients


@pytest.mark.asyncio
async def test_class_overview_merges_all_sources(
    teacher_client: AsyncClient,
    mock_clients: MockBackingClients,
) -> None:
    """U-TBFF-SCR-01: Class overview merges scores + miss + plagiarism."""
    exam_id = "exam-aaa"

    mock_clients.responses[
        f"{SCORE_ENGINE_URL}/api/v1/scores/{exam_id}"
    ] = {
        "items": [
            {
                "student_id": "s1",
                "student_name": "Alice",
                "total_score": 85.0,
                "ai_confidence": 0.92,
            },
            {
                "student_id": "s2",
                "student_name": "Bob",
                "total_score": 72.0,
                "ai_confidence": 0.88,
            },
        ],
    }

    mock_clients.responses[
        f"{DOC_ASSEMBLY_URL}/api/v1/documents/{exam_id}/miss-indicators"
    ] = {
        "exam_id": exam_id,
        "cells": [
            {"student_id": "s1", "question_id": "q3", "state": "miss_no_strokes"},
            {"student_id": "s2", "question_id": "q1", "state": "answered"},
        ],
    }

    mock_clients.responses[
        f"{PLAGIARISM_URL}/api/v1/plagiarism/{exam_id}/flags"
    ] = {
        "items": [
            {
                "flag_id": "f1",
                "student_a_id": "s1",
                "student_b_id": "s2",
                "question_id": "q2",
                "composite_score": 0.87,
                "severity": "high",
            },
        ],
    }

    resp = await teacher_client.get(f"/api/v1/teacher/exams/{exam_id}/scores")
    assert resp.status_code == 200
    data = resp.json()
    rows = data["rows"]

    assert len(rows) == 2

    alice = next(r for r in rows if r["student_id"] == "s1")
    assert alice["total_score"] == 85.0
    assert alice["miss_indicator_count"] == 1  # one miss_no_strokes
    assert alice["plagiarism_flag_count"] == 1  # involved in one flag

    bob = next(r for r in rows if r["student_id"] == "s2")
    assert bob["miss_indicator_count"] == 0  # "answered" is not a miss
    assert bob["plagiarism_flag_count"] == 1  # same flag


@pytest.mark.asyncio
async def test_class_overview_degrades_on_missing_services(
    teacher_client: AsyncClient,
    mock_clients: MockBackingClients,
) -> None:
    """U-TBFF-SCR-02: Graceful degradation when miss/plag services are down."""
    exam_id = "exam-bbb"

    # Only score-engine responds; miss indicators and plagiarism return None
    mock_clients.responses[
        f"{SCORE_ENGINE_URL}/api/v1/scores/{exam_id}"
    ] = {
        "items": [
            {
                "student_id": "s1",
                "student_name": "Charlie",
                "total_score": 90.0,
                "ai_confidence": 0.95,
            },
        ],
    }
    # miss_indicators and plagiarism URLs not set -> return None -> defaults 0

    resp = await teacher_client.get(f"/api/v1/teacher/exams/{exam_id}/scores")
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    assert len(rows) == 1
    assert rows[0]["miss_indicator_count"] == 0
    assert rows[0]["plagiarism_flag_count"] == 0


@pytest.mark.asyncio
async def test_student_drill_down(
    teacher_client: AsyncClient,
    mock_clients: MockBackingClients,
) -> None:
    """U-TBFF-SCR-03: Student drill-down merges scores + answer pages."""
    exam_id = "exam-ccc"
    student_id = "s1"

    mock_clients.responses[
        f"{SCORE_ENGINE_URL}/api/v1/scores/{exam_id}/students/{student_id}"
    ] = {
        "student_id": student_id,
        "student_name": "Diana",
        "total_score": 78.0,
        "questions": [
            {"question_id": "q1", "current_score": 10, "confidence": 0.9},
        ],
    }

    mock_clients.responses[
        f"{DOC_ASSEMBLY_URL}/api/v1/documents/{exam_id}/students/{student_id}/pages"
    ] = {
        "pages": [
            "https://s3.example.com/page1.png",
            "https://s3.example.com/page2.png",
        ],
    }

    resp = await teacher_client.get(
        f"/api/v1/teacher/exams/{exam_id}/students/{student_id}",
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["student_id"] == student_id
    assert len(data["answer_pages"]) == 2
    assert len(data["questions"]) == 1


@pytest.mark.asyncio
async def test_student_drill_down_404(
    teacher_client: AsyncClient,
    mock_clients: MockBackingClients,
) -> None:
    """U-TBFF-SCR-04: 404 when score-engine has no data for the student."""
    resp = await teacher_client.get(
        "/api/v1/teacher/exams/exam-xxx/students/unknown",
    )
    assert resp.status_code == 404
