"""Integration tests for routes/messages.py — append + read receipt cycle.

Test IDs: I-CHAT-RT-01 through I-CHAT-RT-07

These tests use FastAPI's TestClient with a mocked MessageRepo,
mocked enrollment adapter, and mocked auth dependency to verify
the HTTP layer without a real DB.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from exampen_common.auth import ExamPenUser

from src.main import create_app


# -- Fixtures --------------------------------------------------------------


def _make_user(
    user_id: str = "teacher-1",
    tenant_id: str = "tenant-a",
    stoody_role: str = "tutor",
    exampen_roles: list[str] | None = None,
) -> ExamPenUser:
    return ExamPenUser(
        user_id=user_id,
        tenant_id=tenant_id,
        stoody_role=stoody_role,
        exampen_roles=exampen_roles or ["teacher"],
    )


def _mock_repo() -> AsyncMock:
    repo = AsyncMock()
    repo.append_message.return_value = {
        "message_id": "msg-001",
        "sender_id": "teacher-1",
        "recipient_id": "student-1",
        "exam_id": "exam-abc",
        "content": "Check Q3 again.",
        "attachment_uri": None,
        "sent_at": "2026-03-19T10:00:00+00:00",
    }
    repo.get_thread.return_value = [
        {
            "message_id": "msg-001",
            "sender_id": "teacher-1",
            "recipient_id": "student-1",
            "exam_id": "exam-abc",
            "content": "Check Q3 again.",
            "attachment_uri": None,
            "sent_at": "2026-03-19T10:00:00+00:00",
        },
    ]
    repo.append_read_receipt.return_value = {
        "exam_id": "exam-abc",
        "other_user_id": "student-1",
        "read_at": "2026-03-19T10:05:00+00:00",
    }
    repo.list_threads.return_value = [
        {
            "exam_id": "exam-abc",
            "other_user_id": "student-1",
            "last_message_at": "2026-03-19T10:00:00+00:00",
        },
    ]
    return repo


def _mock_enrollment(
    teacher_ids: list[str] | None = None,
    student_ids: list[str] | None = None,
) -> AsyncMock:
    """Mock ExamEnrollmentAdapter with configurable enrollment lists."""
    enrollment = AsyncMock()
    enrollment.get_teacher_ids.return_value = (
        teacher_ids if teacher_ids is not None else ["teacher-1"]
    )
    enrollment.get_student_ids.return_value = (
        student_ids if student_ids is not None else ["student-1", "student-2"]
    )
    return enrollment


@pytest.fixture()
def teacher_client() -> TestClient:
    """TestClient with teacher auth and mocked repo."""
    user = _make_user()
    app = create_app()

    async def _fake_user() -> ExamPenUser:
        return user

    from exampen_common.auth import get_current_user
    app.dependency_overrides[get_current_user] = _fake_user
    app.state.message_repo = _mock_repo()
    app.state.enrollment = _mock_enrollment()
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def student_client() -> TestClient:
    """TestClient with student auth and mocked repo."""
    user = _make_user(
        user_id="student-1",
        stoody_role="student",
        exampen_roles=["student"],
    )
    app = create_app()

    async def _fake_user() -> ExamPenUser:
        return user

    from exampen_common.auth import get_current_user
    app.dependency_overrides[get_current_user] = _fake_user
    app.state.message_repo = _mock_repo()
    app.state.enrollment = _mock_enrollment()
    return TestClient(app, raise_server_exceptions=False)


# -- I-CHAT-RT-01: Teacher appends a message ------------------------------


def test_teacher_appends_message(teacher_client: TestClient):
    """I-CHAT-RT-01: POST /threads/{exam}/{student} returns 201."""
    resp = teacher_client.post(
        "/api/v1/chat/threads/exam-abc/student-1",
        json={"content": "Check Q3 again."},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["message_id"] == "msg-001"
    assert body["sender_id"] == "teacher-1"
    assert body["recipient_id"] == "student-1"


# -- I-CHAT-RT-02: Student appends a message ------------------------------


def test_student_appends_message(student_client: TestClient):
    """I-CHAT-RT-02: Student can send message to teacher."""
    repo = student_client.app.state.message_repo  # type: ignore[union-attr]
    repo.append_message.return_value = {
        "message_id": "msg-002",
        "sender_id": "student-1",
        "recipient_id": "teacher-1",
        "exam_id": "exam-abc",
        "content": "I believe Q3 deserves partial marks.",
        "attachment_uri": None,
        "sent_at": "2026-03-19T10:01:00+00:00",
    }
    resp = student_client.post(
        "/api/v1/chat/threads/exam-abc/teacher-1",
        json={"content": "I believe Q3 deserves partial marks."},
    )
    assert resp.status_code == 201
    assert resp.json()["sender_id"] == "student-1"


# -- I-CHAT-RT-03: Get thread messages ------------------------------------


def test_get_thread_returns_messages(teacher_client: TestClient):
    """I-CHAT-RT-03: GET /threads/{exam}/{student} returns thread."""
    resp = teacher_client.get("/api/v1/chat/threads/exam-abc/student-1")
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    assert len(body["items"]) == 1
    assert body["items"][0]["content"] == "Check Q3 again."


# -- I-CHAT-RT-04: Mark thread as read ------------------------------------


def test_mark_thread_read(teacher_client: TestClient):
    """I-CHAT-RT-04: POST /threads/{exam}/{student}/read returns receipt."""
    resp = teacher_client.post(
        "/api/v1/chat/threads/exam-abc/student-1/read",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["exam_id"] == "exam-abc"
    assert body["other_user_id"] == "student-1"
    assert "read_at" in body


# -- I-CHAT-RT-05: Empty content rejected ---------------------------------


def test_empty_content_returns_422(teacher_client: TestClient):
    """I-CHAT-RT-05: POST with empty content returns 422."""
    resp = teacher_client.post(
        "/api/v1/chat/threads/exam-abc/student-1",
        json={"content": ""},
    )
    assert resp.status_code == 422


# -- I-CHAT-RT-06: Self-message rejected ----------------------------------


def test_self_message_returns_422(teacher_client: TestClient):
    """I-CHAT-RT-06: Sending message to self returns 422."""
    resp = teacher_client.post(
        "/api/v1/chat/threads/exam-abc/teacher-1",
        json={"content": "Hello myself"},
    )
    assert resp.status_code == 422


# -- I-CHAT-RT-07: Teacher messaging unrelated student gets 403 -----------


def test_teacher_messaging_unrelated_student_returns_403():
    """I-CHAT-RT-07: Teacher messaging a student not in their exam gets 403."""
    user = _make_user(user_id="teacher-1", exampen_roles=["teacher"])
    app = create_app()

    async def _fake_user() -> ExamPenUser:
        return user

    from exampen_common.auth import get_current_user
    app.dependency_overrides[get_current_user] = _fake_user
    app.state.message_repo = _mock_repo()
    # Enrollment returns students that do NOT include the target
    app.state.enrollment = _mock_enrollment(
        student_ids=["student-5", "student-6"],
    )

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/v1/chat/threads/exam-abc/student-99",
        json={"content": "You are not my student."},
    )
    assert resp.status_code == 403
    assert "students in their exam" in resp.json()["detail"]
