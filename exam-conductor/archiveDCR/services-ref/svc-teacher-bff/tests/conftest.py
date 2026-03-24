"""Shared test fixtures for svc-teacher-bff tests.

Provides a configured HTTPX AsyncClient against the FastAPI app with
mocked backing services (no real network calls).
"""

from __future__ import annotations

from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from exampen_common.auth import ExamPenUser
from src.main import app


def _make_user(
    *,
    role: str = "teacher",
    user_id: str = "teacher-001",
    tenant_id: str = "tenant-001",
) -> ExamPenUser:
    return ExamPenUser(
        user_id=user_id,
        tenant_id=tenant_id,
        stoody_role="tutor" if role == "teacher" else role,
        exampen_roles=[role],
        name="Test Teacher",
        email="teacher@example.com",
    )


TEACHER_USER = _make_user(role="teacher")
HOD_USER = _make_user(role="hod", user_id="hod-001")
STUDENT_USER = _make_user(role="student", user_id="student-001")
PARENT_USER = _make_user(role="parent", user_id="parent-001")


class MockBackingClients:
    """In-memory backing client that records calls and returns canned data."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.responses: dict[str, dict[str, Any] | None] = {}

    async def request(
        self,
        method: str,
        url: str,
        *,
        auth_token: str | None = None,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        self.calls.append((method, url, json))
        return self.responses.get(url)

    async def request_or_raise(
        self,
        method: str,
        url: str,
        *,
        auth_token: str | None = None,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((method, url, json))
        data = self.responses.get(url)
        if data is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=502, detail="mock: no canned response")
        return data

    async def close(self) -> None:
        pass


@pytest_asyncio.fixture
async def mock_clients() -> MockBackingClients:
    return MockBackingClients()


@pytest_asyncio.fixture
async def teacher_client(
    mock_clients: MockBackingClients,
) -> AsyncGenerator[AsyncClient, None]:
    """HTTPX client authenticated as a teacher."""
    from exampen_common.auth import get_current_user

    app.state.clients = mock_clients
    app.dependency_overrides[get_current_user] = lambda: TEACHER_USER
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"Authorization": "Bearer fake-teacher-token"},
        ) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_current_user, None)


@pytest_asyncio.fixture
async def student_client(
    mock_clients: MockBackingClients,
) -> AsyncGenerator[AsyncClient, None]:
    """HTTPX client authenticated as a student (should get 403)."""
    from exampen_common.auth import get_current_user

    app.state.clients = mock_clients
    app.dependency_overrides[get_current_user] = lambda: STUDENT_USER
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"Authorization": "Bearer fake-student-token"},
        ) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_current_user, None)
