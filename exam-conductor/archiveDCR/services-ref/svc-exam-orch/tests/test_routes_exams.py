"""Integration tests for exam CRUD and FSM transition routes.

Test IDs: I-ORCH-EXAM-01 through I-ORCH-EXAM-08.

These tests use FastAPI's TestClient with a mock DB session and
mock NATS client, so they exercise the full route -> domain -> storage
call path without requiring real infrastructure.
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
# Fixtures
# ---------------------------------------------------------------------------

_EXAM_ROW: dict[str, Any] = {
    "exam_id": str(uuid.uuid4()),
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
    "late_entry_cutoff_min": None,
    "objection_window_days": None,
    "created_at": datetime(2026, 3, 1, tzinfo=timezone.utc),
    "updated_at": datetime(2026, 3, 1, tzinfo=timezone.utc),
}


def _mock_user() -> MagicMock:
    user = MagicMock()
    user.user_id = "teacher-1"
    user.tenant_id = "tenant-1"
    user.stoody_role = "tutor"
    user.exampen_roles = ["evaluator"]
    return user


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(exams_router, prefix="/api/v1/exams")
    app.include_router(bindings_router, prefix="/api/v1/exams/{exam_id}/bindings")
    app.include_router(assignments_router, prefix="/api/v1/exams/{exam_id}/invigilators")
    return app


# We patch auth and DB at module level for all integration tests in this file.


@pytest.fixture()
def mock_session() -> AsyncMock:
    session = AsyncMock()
    return session


@pytest.fixture()
def app(mock_session: AsyncMock) -> FastAPI:
    application = _build_app()

    mock_nats = AsyncMock()
    mock_stoody = AsyncMock()

    application.state.session_factory = MagicMock()
    application.state.nats_client = mock_nats
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
# I-ORCH-EXAM-01: Create exam
# ---------------------------------------------------------------------------


class TestCreateExam:
    @pytest.mark.asyncio
    async def test_create_returns_201(self, client: AsyncClient, app: FastAPI) -> None:
        """POST /api/v1/exams should return 201 on success."""
        from src.storage.exam_repo import ExamRepo

        with (
            patch("src.routes.exams.get_current_user", return_value=_mock_user()),
            patch("src.routes.exams.rls_session") as mock_rls,
            patch.object(ExamRepo, "create", return_value=_EXAM_ROW),
        ):
            async def _fake_rls(*_a: Any, **_k: Any) -> AsyncGenerator:
                yield AsyncMock()

            mock_rls.side_effect = _fake_rls

            resp = await client.post("/api/v1/exams", json={
                "title": "Math Final",
                "subject_id": "math-101",
                "class_id": "cls-10",
                "section_id": "sec-a",
                "scheduled_at": "2026-04-01T09:00:00Z",
                "duration_min": 90,
                "question_count": 5,
                "total_marks": 50,
            })

        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "Math Final"
        assert data["state"] == "created"


# ---------------------------------------------------------------------------
# I-ORCH-EXAM-02: List exams
# ---------------------------------------------------------------------------


class TestListExams:
    @pytest.mark.asyncio
    async def test_list_returns_items(self, client: AsyncClient) -> None:
        from src.storage.exam_repo import ExamRepo

        with (
            patch("src.routes.exams.get_current_user", return_value=_mock_user()),
            patch("src.routes.exams.rls_session") as mock_rls,
            patch.object(ExamRepo, "list_exams", return_value=[_EXAM_ROW]),
        ):
            async def _fake_rls(*_a: Any, **_k: Any) -> AsyncGenerator:
                yield AsyncMock()

            mock_rls.side_effect = _fake_rls
            resp = await client.get("/api/v1/exams")

        assert resp.status_code == 200
        assert len(resp.json()["items"]) == 1


# ---------------------------------------------------------------------------
# I-ORCH-EXAM-03: Get exam detail
# ---------------------------------------------------------------------------


class TestGetExam:
    @pytest.mark.asyncio
    async def test_get_existing(self, client: AsyncClient) -> None:
        from src.storage.exam_repo import ExamRepo

        with (
            patch("src.routes.exams.get_current_user", return_value=_mock_user()),
            patch("src.routes.exams.rls_session") as mock_rls,
            patch.object(ExamRepo, "get_by_id", return_value=_EXAM_ROW),
        ):
            async def _fake_rls(*_a: Any, **_k: Any) -> AsyncGenerator:
                yield AsyncMock()

            mock_rls.side_effect = _fake_rls
            resp = await client.get(f"/api/v1/exams/{_EXAM_ROW['exam_id']}")

        assert resp.status_code == 200
        assert resp.json()["exam_id"] == _EXAM_ROW["exam_id"]

    @pytest.mark.asyncio
    async def test_get_missing_returns_404(self, client: AsyncClient) -> None:
        from src.storage.exam_repo import ExamRepo

        with (
            patch("src.routes.exams.get_current_user", return_value=_mock_user()),
            patch("src.routes.exams.rls_session") as mock_rls,
            patch.object(ExamRepo, "get_by_id", return_value=None),
        ):
            async def _fake_rls(*_a: Any, **_k: Any) -> AsyncGenerator:
                yield AsyncMock()

            mock_rls.side_effect = _fake_rls
            resp = await client.get(f"/api/v1/exams/{uuid.uuid4()}")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# I-ORCH-EXAM-04: FSM transition
# ---------------------------------------------------------------------------


class TestTransition:
    @pytest.mark.asyncio
    async def test_valid_transition(self, client: AsyncClient, app: FastAPI) -> None:
        from src.storage.assignment_repo import AssignmentRepo
        from src.storage.exam_repo import ExamRepo

        # Use principal role — bypasses all transition RBAC checks
        principal_user = MagicMock()
        principal_user.user_id = "teacher-1"
        principal_user.tenant_id = "tenant-1"
        principal_user.stoody_role = "tutor"
        principal_user.exampen_roles = ["principal"]

        updated = {**_EXAM_ROW, "state": "armed", "updated_at": datetime.now(timezone.utc)}

        with (
            patch("src.routes.exams.get_current_user", return_value=principal_user),
            patch("src.routes.exams.rls_session") as mock_rls,
            patch.object(ExamRepo, "get_by_id", return_value=_EXAM_ROW),
            patch.object(AssignmentRepo, "list_by_exam", return_value=[]),
            patch.object(ExamRepo, "transition_state", return_value=updated),
            patch("src.routes.exams.publish_lifecycle_event", new_callable=AsyncMock),
        ):
            async def _fake_rls(*_a: Any, **_k: Any) -> AsyncGenerator:
                yield AsyncMock()

            mock_rls.side_effect = _fake_rls

            resp = await client.post(
                f"/api/v1/exams/{_EXAM_ROW['exam_id']}/transitions",
                json={"to_state": "armed", "actor_id": "teacher-1"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["from_state"] == "created"
        assert data["to_state"] == "armed"

    @pytest.mark.asyncio
    async def test_invalid_transition_returns_409(self, client: AsyncClient) -> None:
        from src.storage.assignment_repo import AssignmentRepo
        from src.storage.exam_repo import ExamRepo

        with (
            patch("src.routes.exams.get_current_user", return_value=_mock_user()),
            patch("src.routes.exams.rls_session") as mock_rls,
            patch.object(ExamRepo, "get_by_id", return_value=_EXAM_ROW),
            patch.object(AssignmentRepo, "list_by_exam", return_value=[]),
        ):
            async def _fake_rls(*_a: Any, **_k: Any) -> AsyncGenerator:
                yield AsyncMock()

            mock_rls.side_effect = _fake_rls

            resp = await client.post(
                f"/api/v1/exams/{_EXAM_ROW['exam_id']}/transitions",
                json={"to_state": "locked", "actor_id": "teacher-1"},
            )

        assert resp.status_code == 409
