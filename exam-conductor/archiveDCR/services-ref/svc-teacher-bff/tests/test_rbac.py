"""RBAC enforcement tests — student and parent tokens must get 403.

Test IDs: U-TBFF-RBAC-01 through U-TBFF-RBAC-05
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_student_gets_403_on_exam_list(
    student_client: AsyncClient,
) -> None:
    """U-TBFF-RBAC-01: Student token rejected on GET /teacher/exams."""
    resp = await student_client.get("/api/v1/teacher/exams")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_student_gets_403_on_scores(
    student_client: AsyncClient,
) -> None:
    """U-TBFF-RBAC-02: Student token rejected on class score overview."""
    resp = await student_client.get("/api/v1/teacher/exams/any-exam/scores")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_student_gets_403_on_objections(
    student_client: AsyncClient,
) -> None:
    """U-TBFF-RBAC-03: Student token rejected on objection inbox."""
    resp = await student_client.get("/api/v1/teacher/exams/any-exam/objections")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_student_gets_403_on_analytics(
    student_client: AsyncClient,
) -> None:
    """U-TBFF-RBAC-04: Student token rejected on leaderboard."""
    resp = await student_client.get("/api/v1/teacher/exams/any-exam/leaderboard")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_student_gets_403_on_plagiarism(
    student_client: AsyncClient,
) -> None:
    """U-TBFF-RBAC-05: Student token rejected on plagiarism flags."""
    resp = await student_client.get("/api/v1/teacher/exams/any-exam/plagiarism")
    assert resp.status_code == 403
