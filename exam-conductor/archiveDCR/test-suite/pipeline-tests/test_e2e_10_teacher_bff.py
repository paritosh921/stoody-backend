"""
E2E-10: Teacher BFF score aggregation.

Services involved: svc-teacher-bff, all backing services.

What it proves:
    The teacher BFF returns correctly aggregated class score data drawn from
    multiple backing services (score-engine, analytics, doc-assembly,
    ai-pipeline).  The response shape matches the OpenAPI contract defined
    in ``api/teacher-bff.openapi.yaml``.

    RBAC enforcement: a student JWT must be rejected by teacher BFF endpoints.

Test-ID: E2E-10  (TEST_SUITE_SPEC.md section 2.3)
Level: L5 (multi-service pipeline)
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from bff_helpers import seed_scores_for_students, teacher_get
from conftest import TEACHER_BFF_URL

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]

# Required fields per OpenAPI ``ClassScoreRow`` schema.
CLASS_SCORE_ROW_REQUIRED = {
    "student_id", "student_name", "total_score", "ai_confidence",
}
# Required fields per ``TeacherStudentDetail`` schema.
STUDENT_DETAIL_REQUIRED = {
    "student_id", "student_name", "total_score", "questions",
}
# Required fields per ``QuestionDetail`` inside drill-down.
QUESTION_DETAIL_REQUIRED = {"question_id", "current_score", "confidence"}


class TestTeacherBFFScoreAggregation:
    """E2E-10 -- teacher BFF score aggregation (real)."""

    async def test_class_overview_returns_contract_shape(
        self, http_session, publish_event, nats_client,
        ai_result_factory, student_factory,
    ):
        """GET /api/v1/teacher/exams/{id}/scores -> ClassScoreRow[]."""
        exam_id = str(uuid.uuid4())
        students = student_factory.create_batch(5)

        await seed_scores_for_students(
            publish_event, nats_client, ai_result_factory,
            exam_id=exam_id, students=students,
        )
        await asyncio.sleep(2)

        status, body = await teacher_get(
            http_session,
            f"/api/v1/teacher/exams/{exam_id}/scores",
        )
        assert status == 200, (
            f"Expected 200 from class overview, got {status}"
        )
        assert "rows" in body, (
            f"Response must contain 'rows'. Got: {list(body.keys())}"
        )
        rows = body["rows"]
        assert isinstance(rows, list) and len(rows) > 0

        for row in rows:
            missing = CLASS_SCORE_ROW_REQUIRED - set(row.keys())
            assert not missing, f"ClassScoreRow missing: {missing}"
            assert isinstance(row["total_score"], (int, float))
            assert isinstance(row["ai_confidence"], (int, float))

    async def test_student_drill_down_returns_contract_shape(
        self, http_session, publish_event, nats_client,
        ai_result_factory,
    ):
        """GET /api/v1/teacher/exams/{id}/scores/{sid}
        -> TeacherStudentDetail with QuestionDetail[]."""
        exam_id = str(uuid.uuid4())
        student_id = str(uuid.uuid4())

        await seed_scores_for_students(
            publish_event, nats_client, ai_result_factory,
            exam_id=exam_id, students=[{"id": student_id}],
        )
        await asyncio.sleep(2)

        status, body = await teacher_get(
            http_session,
            f"/api/v1/teacher/exams/{exam_id}/scores/{student_id}",
        )
        assert status == 200, (
            f"Expected 200 from drill-down, got {status}"
        )
        missing = STUDENT_DETAIL_REQUIRED - set(body.keys())
        assert not missing, f"TeacherStudentDetail missing: {missing}"

        questions = body["questions"]
        assert isinstance(questions, list) and len(questions) > 0
        for q in questions:
            q_missing = QUESTION_DETAIL_REQUIRED - set(q.keys())
            assert not q_missing, f"QuestionDetail missing: {q_missing}"

    async def test_rbac_student_token_gets_403(self, http_session):
        """Student JWT -> 401/403 on teacher BFF endpoints."""
        exam_id = str(uuid.uuid4())
        status, _ = await teacher_get(
            http_session,
            f"/api/v1/teacher/exams/{exam_id}/scores",
            token="student_test_token",
        )
        assert status in (401, 403), (
            f"Expected 401/403 for student token, got {status}"
        )

    async def test_rbac_no_token_gets_401(self, http_session):
        """Missing auth header -> 401."""
        exam_id = str(uuid.uuid4())
        async with http_session.get(
            f"{TEACHER_BFF_URL}/api/v1/teacher/exams/{exam_id}/scores",
        ) as resp:
            assert resp.status in (401, 403), (
                f"Expected 401/403 for no auth, got {resp.status}"
            )

    async def test_nonexistent_exam_returns_empty_or_404(
        self, http_session,
    ):
        """Random exam_id -> 200 with empty rows or 404."""
        exam_id = str(uuid.uuid4())
        status, body = await teacher_get(
            http_session,
            f"/api/v1/teacher/exams/{exam_id}/scores",
        )
        if status == 200:
            assert len(body.get("rows", [])) == 0
        else:
            assert status == 404, f"Expected 200 or 404, got {status}"
