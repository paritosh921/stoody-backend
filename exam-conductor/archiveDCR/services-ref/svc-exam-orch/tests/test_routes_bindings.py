"""Integration tests for pen binding routes.

Test IDs: I-ORCH-BND-01 through I-ORCH-BND-05.

Uses mock DB sessions and mock Stoody client to exercise the full
route -> domain -> storage path.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.routes.bindings import router as bindings_router
from src.routes.exams import router as exams_router


# ---------------------------------------------------------------------------
# Fixtures
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
    "state": "armed",
    "created_by": "teacher-1",
    "created_at": datetime(2026, 3, 1, tzinfo=timezone.utc),
    "updated_at": datetime(2026, 3, 1, tzinfo=timezone.utc),
}

_BINDING_ROW: dict[str, Any] = {
    "exam_id": _EXAM_ID,
    "pen_mac": "AA:BB:CC:DD:EE:FF",
    "student_id": "student-1",
    "student_name": "Alice",
    "student_roll": "101",
    "status": "provisional",
    "source": "registration_scan",
    "bound_at": datetime(2026, 4, 1, 8, 30, tzinfo=timezone.utc).isoformat(),
    "server_confirmed_at": None,
    "rejection_reason": None,
}

_STUDENTS = [
    {"student_id": "student-1", "name": "Alice", "roll": "101"},
    {"student_id": "student-2", "name": "Bob", "roll": "102"},
]


_ASSIGNMENT_ROWS = [
    {"exam_id": _EXAM_ID, "user_id": "teacher-1", "role": "invigilator",
     "assigned_at": None},
]


def _mock_user() -> MagicMock:
    user = MagicMock()
    user.user_id = "teacher-1"
    user.tenant_id = "tenant-1"
    user.stoody_role = "tutor"
    user.exampen_roles = ["invigilator"]
    return user


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(exams_router, prefix="/api/v1/exams")
    app.include_router(bindings_router, prefix="/api/v1/exams/{exam_id}/bindings")
    return app


@pytest.fixture()
def app() -> FastAPI:
    application = _build_app()
    mock_stoody = AsyncMock()
    mock_stoody.get_students = AsyncMock(return_value=_STUDENTS)

    application.state.session_factory = MagicMock()
    application.state.nats_client = AsyncMock()
    application.state.stoody_client = mock_stoody
    return application


@pytest.fixture()
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test",
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# I-ORCH-BND-01: Create binding
# ---------------------------------------------------------------------------


class TestCreateBinding:
    @pytest.mark.asyncio
    async def test_create_returns_202(self, client: AsyncClient) -> None:
        from src.storage.assignment_repo import AssignmentRepo
        from src.storage.binding_repo import BindingRepo
        from src.storage.exam_repo import ExamRepo

        with (
            patch("src.routes.bindings.get_current_user", return_value=_mock_user()),
            patch("src.routes.bindings.rls_session") as mock_rls,
            patch.object(ExamRepo, "get_by_id", return_value=_EXAM_ROW),
            patch.object(AssignmentRepo, "list_by_exam", return_value=_ASSIGNMENT_ROWS),
            patch.object(BindingRepo, "list_by_exam", return_value=[]),
            patch.object(BindingRepo, "create", return_value=_BINDING_ROW),
        ):
            async def _fake_rls(*_a: Any, **_k: Any) -> AsyncGenerator:
                yield AsyncMock()

            mock_rls.side_effect = _fake_rls

            resp = await client.post(
                f"/api/v1/exams/{_EXAM_ID}/bindings",
                json={
                    "pen_mac": "AA:BB:CC:DD:EE:FF",
                    "student_id": "student-1",
                    "source": "registration_scan",
                },
            )

        assert resp.status_code == 202
        assert resp.json()["status"] == "provisional"


# ---------------------------------------------------------------------------
# I-ORCH-BND-02: Duplicate pen rejected
# ---------------------------------------------------------------------------


class TestDuplicatePen:
    @pytest.mark.asyncio
    async def test_duplicate_pen_returns_409(self, client: AsyncClient) -> None:
        from src.storage.assignment_repo import AssignmentRepo
        from src.storage.binding_repo import BindingRepo
        from src.storage.exam_repo import ExamRepo

        with (
            patch("src.routes.bindings.get_current_user", return_value=_mock_user()),
            patch("src.routes.bindings.rls_session") as mock_rls,
            patch.object(ExamRepo, "get_by_id", return_value=_EXAM_ROW),
            patch.object(AssignmentRepo, "list_by_exam", return_value=_ASSIGNMENT_ROWS),
            patch.object(BindingRepo, "list_by_exam", return_value=[_BINDING_ROW]),
        ):
            async def _fake_rls(*_a: Any, **_k: Any) -> AsyncGenerator:
                yield AsyncMock()

            mock_rls.side_effect = _fake_rls

            resp = await client.post(
                f"/api/v1/exams/{_EXAM_ID}/bindings",
                json={
                    "pen_mac": "AA:BB:CC:DD:EE:FF",
                    "student_id": "student-2",
                    "source": "manual_register",
                },
            )

        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# I-ORCH-BND-03: List bindings
# ---------------------------------------------------------------------------


class TestListBindings:
    @pytest.mark.asyncio
    async def test_list_returns_items(self, client: AsyncClient) -> None:
        from src.storage.binding_repo import BindingRepo

        with (
            patch("src.routes.bindings.get_current_user", return_value=_mock_user()),
            patch("src.routes.bindings.rls_session") as mock_rls,
            patch.object(BindingRepo, "list_by_exam", return_value=[_BINDING_ROW]),
        ):
            async def _fake_rls(*_a: Any, **_k: Any) -> AsyncGenerator:
                yield AsyncMock()

            mock_rls.side_effect = _fake_rls

            resp = await client.get(f"/api/v1/exams/{_EXAM_ID}/bindings")

        assert resp.status_code == 200
        assert len(resp.json()["items"]) == 1


# ---------------------------------------------------------------------------
# I-ORCH-BND-04: Confirm binding
# ---------------------------------------------------------------------------


class TestConfirmBinding:
    @pytest.mark.asyncio
    async def test_confirm_provisional(self, client: AsyncClient) -> None:
        from src.storage.assignment_repo import AssignmentRepo
        from src.storage.binding_repo import BindingRepo

        confirmed = {**_BINDING_ROW, "status": "confirmed"}

        with (
            patch("src.routes.bindings.get_current_user", return_value=_mock_user()),
            patch("src.routes.bindings.rls_session") as mock_rls,
            patch.object(AssignmentRepo, "list_by_exam", return_value=_ASSIGNMENT_ROWS),
            patch.object(BindingRepo, "get_by_pen", return_value=_BINDING_ROW),
            patch.object(BindingRepo, "confirm_or_reject", return_value=confirmed),
        ):
            async def _fake_rls(*_a: Any, **_k: Any) -> AsyncGenerator:
                yield AsyncMock()

            mock_rls.side_effect = _fake_rls

            resp = await client.post(
                f"/api/v1/exams/{_EXAM_ID}/bindings/AA:BB:CC:DD:EE:FF/confirm",
                json={"status": "confirmed"},
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "confirmed"


# ---------------------------------------------------------------------------
# I-ORCH-BND-05: Reject already-confirmed returns 409
# ---------------------------------------------------------------------------


class TestRejectConfirmed:
    @pytest.mark.asyncio
    async def test_confirm_non_provisional_returns_409(
        self, client: AsyncClient,
    ) -> None:
        from src.storage.assignment_repo import AssignmentRepo
        from src.storage.binding_repo import BindingRepo

        confirmed_row = {**_BINDING_ROW, "status": "confirmed"}

        with (
            patch("src.routes.bindings.get_current_user", return_value=_mock_user()),
            patch("src.routes.bindings.rls_session") as mock_rls,
            patch.object(AssignmentRepo, "list_by_exam", return_value=_ASSIGNMENT_ROWS),
            patch.object(BindingRepo, "get_by_pen", return_value=confirmed_row),
        ):
            async def _fake_rls(*_a: Any, **_k: Any) -> AsyncGenerator:
                yield AsyncMock()

            mock_rls.side_effect = _fake_rls

            resp = await client.post(
                f"/api/v1/exams/{_EXAM_ID}/bindings/AA:BB:CC:DD:EE:FF/confirm",
                json={"status": "rejected", "rejection_reason": "wrong pen"},
            )

        assert resp.status_code == 409
