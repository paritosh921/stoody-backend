"""Integration tests for RBAC enforcement at route level.

Test IDs: I-RBAC-01 through I-RBAC-04.

Verifies that route handlers reject forbidden callers with HTTP 403
and accept authorized callers.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.routes.assignments import router as assignments_router
from src.routes.bindings import router as bindings_router
from src.routes.exams import router as exams_router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXAM_ID = str(uuid.uuid4())

_EXAM_ROW: dict[str, Any] = {
    "exam_id": _EXAM_ID,
    "title": "Math Final",
    "subject_id": "math-101",
    "class_id": "cls-10",
    "section_id": "sec-a",
    "scheduled_at": datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc),
    "duration_min": 90,
    "question_count": 5,
    "total_marks": 50.0,
    "negative_marking": False,
    "variants": [],
    "state": "created",
    "created_by": "teacher-1",
    "created_at": datetime(2026, 3, 1, tzinfo=timezone.utc),
    "updated_at": datetime(2026, 3, 1, tzinfo=timezone.utc),
}


def _mock_user_obj(
    roles: list[str],
    user_id: str = "user-1",
    stoody_role: str = "tutor",
) -> MagicMock:
    """MagicMock that quacks like ExamPenUser."""
    u = MagicMock()
    u.user_id = user_id
    u.tenant_id = "tenant-1"
    u.stoody_role = stoody_role
    u.exampen_roles = roles
    return u


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(exams_router, prefix="/api/v1/exams")
    app.include_router(
        bindings_router, prefix="/api/v1/exams/{exam_id}/bindings",
    )
    app.include_router(
        assignments_router, prefix="/api/v1/exams/{exam_id}/invigilators",
    )
    mock_stoody = AsyncMock()
    mock_stoody.get_students = AsyncMock(return_value=[])
    app.state.session_factory = MagicMock()
    app.state.nats_client = AsyncMock()
    app.state.stoody_client = mock_stoody
    return app


@pytest.fixture()
async def rbac_client() -> AsyncGenerator[AsyncClient, None]:
    app = _build_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _fake_rls(*_a: Any, **_k: Any) -> AsyncGenerator:
    yield AsyncMock()


_CREATE_EXAM_BODY: dict[str, Any] = {
    "title": "Math Final",
    "subject_id": "math-101",
    "class_id": "cls-10",
    "section_id": "sec-a",
    "scheduled_at": "2026-04-01T09:00:00Z",
    "duration_min": 90,
    "question_count": 5,
    "total_marks": 50,
}


# ---------------------------------------------------------------------------
# I-RBAC-01: Student token gets 403 on exam creation
# ---------------------------------------------------------------------------


class TestStudentCannotCreateExam:
    @pytest.mark.asyncio
    async def test_student_403_on_create(
        self, rbac_client: AsyncClient,
    ) -> None:
        student = _mock_user_obj(["student"], stoody_role="student")

        with (
            patch("src.routes.exams.get_current_user", return_value=student),
            patch("src.routes.exams.rls_session", side_effect=_fake_rls),
        ):
            resp = await rbac_client.post(
                "/api/v1/exams", json=_CREATE_EXAM_BODY,
            )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# I-RBAC-02: Parent token gets 403 on binding creation
# ---------------------------------------------------------------------------


class TestParentCannotCreateBinding:
    @pytest.mark.asyncio
    async def test_parent_403_on_binding(
        self, rbac_client: AsyncClient,
    ) -> None:
        parent = _mock_user_obj(
            ["parent"], user_id="parent-1", stoody_role="parent",
        )

        from src.storage.assignment_repo import AssignmentRepo
        from src.storage.exam_repo import ExamRepo

        with (
            patch(
                "src.routes.bindings.get_current_user", return_value=parent,
            ),
            patch("src.routes.bindings.rls_session", side_effect=_fake_rls),
            patch.object(ExamRepo, "get_by_id", return_value=_EXAM_ROW),
            patch.object(AssignmentRepo, "list_by_exam", return_value=[]),
        ):
            resp = await rbac_client.post(
                f"/api/v1/exams/{_EXAM_ID}/bindings",
                json={
                    "pen_mac": "AA:BB:CC:DD:EE:FF",
                    "student_id": "student-1",
                    "source": "manual_register",
                },
            )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# I-RBAC-03: Evaluator can create exam but not assign invigilators
# ---------------------------------------------------------------------------


class TestEvaluatorPermissions:
    @pytest.mark.asyncio
    async def test_evaluator_can_create_exam(
        self, rbac_client: AsyncClient,
    ) -> None:
        evaluator = _mock_user_obj(["evaluator"], user_id="eval-1")

        from src.storage.exam_repo import ExamRepo

        with (
            patch(
                "src.routes.exams.get_current_user", return_value=evaluator,
            ),
            patch("src.routes.exams.rls_session", side_effect=_fake_rls),
            patch.object(ExamRepo, "create", return_value=_EXAM_ROW),
        ):
            resp = await rbac_client.post(
                "/api/v1/exams", json=_CREATE_EXAM_BODY,
            )
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_evaluator_cannot_assign_invigilators(
        self, rbac_client: AsyncClient,
    ) -> None:
        evaluator = _mock_user_obj(["evaluator"], user_id="eval-1")

        with (
            patch(
                "src.routes.assignments.get_current_user",
                return_value=evaluator,
            ),
            patch(
                "src.routes.assignments.rls_session", side_effect=_fake_rls,
            ),
        ):
            resp = await rbac_client.post(
                f"/api/v1/exams/{_EXAM_ID}/invigilators",
                json={
                    "invigilator_ids": ["teacher-2"],
                    "evaluator_ids": [],
                },
            )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# I-RBAC-04: Invigilator can create bindings for assigned exams
# ---------------------------------------------------------------------------


class TestInvigilatorBindings:
    @pytest.mark.asyncio
    async def test_assigned_invigilator_can_bind(
        self, rbac_client: AsyncClient,
    ) -> None:
        invig = _mock_user_obj(
            ["invigilator"], user_id="inv-1", stoody_role="tutor",
        )
        assignment_rows = [
            {
                "exam_id": _EXAM_ID,
                "user_id": "inv-1",
                "role": "invigilator",
                "assigned_at": None,
            },
        ]

        from src.storage.assignment_repo import AssignmentRepo
        from src.storage.binding_repo import BindingRepo
        from src.storage.exam_repo import ExamRepo

        binding_row = {
            "exam_id": _EXAM_ID,
            "pen_mac": "AA:BB:CC:DD:EE:FF",
            "student_id": "student-1",
            "student_name": "",
            "student_roll": "",
            "status": "provisional",
            "source": "registration_scan",
            "bound_at": datetime.now(timezone.utc).isoformat(),
            "server_confirmed_at": None,
            "rejection_reason": None,
        }

        with (
            patch(
                "src.routes.bindings.get_current_user", return_value=invig,
            ),
            patch("src.routes.bindings.rls_session", side_effect=_fake_rls),
            patch.object(ExamRepo, "get_by_id", return_value=_EXAM_ROW),
            patch.object(
                AssignmentRepo, "list_by_exam",
                return_value=assignment_rows,
            ),
            patch.object(BindingRepo, "list_by_exam", return_value=[]),
            patch.object(BindingRepo, "create", return_value=binding_row),
        ):
            resp = await rbac_client.post(
                f"/api/v1/exams/{_EXAM_ID}/bindings",
                json={
                    "pen_mac": "AA:BB:CC:DD:EE:FF",
                    "student_id": "student-1",
                    "source": "registration_scan",
                },
            )
        assert resp.status_code == 202
