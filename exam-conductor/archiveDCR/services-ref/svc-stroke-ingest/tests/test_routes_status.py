"""Integration tests for GET /api/v1/exams/{exam_id}/upload-status.

Test IDs: I-SINGEST-10 through I-SINGEST-13
Markers: integration (requires mocked DB)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.routes.chunks import router as chunks_router
from src.routes.status import router as status_router

_EXAM_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(chunks_router, prefix="/api/v1/strokes")
    app.include_router(status_router, prefix="/api/v1/exams")

    status_repo = AsyncMock()
    status_repo.get_exam_status = AsyncMock(return_value=[])
    app.state.upload_status_repo = status_repo

    # Stubs for chunk route (not used in these tests but needed for mount)
    app.state.idempotency_repo = AsyncMock()
    app.state.stroke_publisher = AsyncMock()

    return app


@pytest.fixture
def app() -> FastAPI:
    return _make_app()


@pytest.fixture
async def client(app: FastAPI) -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _mock_auth():
    mock_user = MagicMock()
    mock_user.user_id = "user-1"
    mock_user.tenant_id = "tenant-1"

    with patch(
        "src.routes.status.get_current_user",
        return_value=mock_user,
    ), patch(
        "src.routes.chunks.get_current_user",
        return_value=mock_user,
    ):
        yield


# ---------------------------------------------------------------------------
# I-SINGEST-10: Empty exam returns empty pens list
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_upload_status_empty(client: AsyncClient):
    resp = await client.get(f"/api/v1/exams/{_EXAM_ID}/upload-status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["exam_id"] == _EXAM_ID
    assert body["pens"] == []


# ---------------------------------------------------------------------------
# I-SINGEST-11: Exam with pen data returns correct structure
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_upload_status_with_pens(client: AsyncClient, app: FastAPI):
    app.state.upload_status_repo.get_exam_status = AsyncMock(
        return_value=[
            {
                "pen_mac": "AA:BB:CC:DD:EE:01",
                "acked_chunks": [0, 1, 2],
                "missing_chunks": [],
                "total_chunks": 3,
                "complete": True,
            },
            {
                "pen_mac": "AA:BB:CC:DD:EE:02",
                "acked_chunks": [0],
                "missing_chunks": [1, 2, 3],
                "total_chunks": 4,
                "complete": False,
            },
        ],
    )
    resp = await client.get(f"/api/v1/exams/{_EXAM_ID}/upload-status")
    assert resp.status_code == 200

    body = resp.json()
    assert len(body["pens"]) == 2
    assert body["pens"][0]["complete"] is True
    assert body["pens"][0]["missing_chunks"] == []
    assert body["pens"][1]["complete"] is False
    assert body["pens"][1]["missing_chunks"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# I-SINGEST-12: Invalid exam_id returns 422
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_upload_status_invalid_exam_id(client: AsyncClient):
    resp = await client.get("/api/v1/exams/not-a-uuid/upload-status")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# I-SINGEST-13: DB failure returns 503
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_upload_status_db_failure(client: AsyncClient, app: FastAPI):
    app.state.upload_status_repo.get_exam_status = AsyncMock(
        side_effect=RuntimeError("DB unreachable"),
    )
    resp = await client.get(f"/api/v1/exams/{_EXAM_ID}/upload-status")
    assert resp.status_code == 503
