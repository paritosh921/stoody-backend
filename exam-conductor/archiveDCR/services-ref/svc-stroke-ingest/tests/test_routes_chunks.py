"""Integration tests for POST /api/v1/strokes/ingest.

Test IDs: I-SINGEST-01 through I-SINGEST-09
Markers: integration (requires mocked NATS, Redis, DB)

These tests use ``httpx.AsyncClient`` against the FastAPI app with
mocked infrastructure dependencies injected via ``app.state``.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.routes.chunks import router as chunks_router
from src.routes.status import router as status_router
from src.storage.upload_status_repo import PenProgress

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_EXAM_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
_PEN_MAC = "AA:BB:CC:DD:EE:FF"
_PAYLOAD = b"stroke bytes here"
_PAYLOAD_B64 = base64.b64encode(_PAYLOAD).decode()
_CRC32 = f"{binascii.crc32(_PAYLOAD) & 0xFFFFFFFF:08x}"


def _valid_body(**overrides) -> dict[str, Any]:
    base = {
        "exam_id": _EXAM_ID,
        "pen_mac": _PEN_MAC,
        "chunk_index": 0,
        "total_chunks": 5,
        "payload_base64": _PAYLOAD_B64,
        "checksum_crc32": _CRC32,
        "upload_path": "wifi",
        "idempotency_key": f"{_EXAM_ID}:{_PEN_MAC}:0",
    }
    base.update(overrides)
    return base


def _pen_progress(
    indices: set[int],
    total: int = 5,
    mac: str = _PEN_MAC,
) -> PenProgress:
    return PenProgress(
        pen_mac=mac,
        total_chunks=total,
        received_indices=frozenset(indices),
    )


def _make_app(
    pen_progress: PenProgress | None = None,
) -> FastAPI:
    """Create a minimal app with mocked state for testing."""
    app = FastAPI()
    app.include_router(chunks_router, prefix="/api/v1/strokes")
    app.include_router(status_router, prefix="/api/v1/exams")

    # Mock idempotency repo
    idem = AsyncMock()
    idem.check_and_mark = AsyncMock(return_value=True)  # new key
    app.state.idempotency_repo = idem

    # Mock NATS publisher
    publisher = AsyncMock()
    publisher.publish_stroke_raw = AsyncMock()
    app.state.stroke_publisher = publisher

    # Mock upload status repo -- must include get_pen_progress
    progress = pen_progress or _pen_progress({0})
    status_repo = AsyncMock()
    status_repo.record_chunk = AsyncMock()
    status_repo.get_pen_progress = AsyncMock(return_value=progress)
    app.state.upload_status_repo = status_repo

    return app


@pytest.fixture
def app() -> FastAPI:
    return _make_app()


@pytest.fixture
async def client(app: FastAPI) -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# Patch auth for all tests in this module
@pytest.fixture(autouse=True)
def _mock_auth():
    mock_user = MagicMock()
    mock_user.user_id = "user-1"
    mock_user.tenant_id = "tenant-1"

    with patch(
        "src.routes.chunks.get_current_user",
        return_value=mock_user,
    ), patch(
        "src.routes.status.get_current_user",
        return_value=mock_user,
    ):
        yield


# ---------------------------------------------------------------------------
# I-SINGEST-01: Successful chunk ingest returns 202
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_ingest_chunk_success(client: AsyncClient, app: FastAPI):
    resp = await client.post("/api/v1/strokes/ingest", json=_valid_body())
    assert resp.status_code == 202

    body = resp.json()
    assert body["accepted"] is True
    assert body["deduplicated"] is False
    assert body["exam_id"] == _EXAM_ID
    assert body["pen_mac"] == _PEN_MAC
    assert body["chunk_index"] == 0

    # Verify NATS publish was called
    app.state.stroke_publisher.publish_stroke_raw.assert_awaited_once()

    # Verify DB record was attempted
    app.state.upload_status_repo.record_chunk.assert_awaited_once()

    # Verify progress was queried from persisted state
    app.state.upload_status_repo.get_pen_progress.assert_awaited_once()


# ---------------------------------------------------------------------------
# I-SINGEST-02: Duplicate chunk returns 202 with deduplicated=True
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_ingest_duplicate_chunk(client: AsyncClient, app: FastAPI):
    app.state.idempotency_repo.check_and_mark = AsyncMock(return_value=False)

    resp = await client.post("/api/v1/strokes/ingest", json=_valid_body())
    assert resp.status_code == 202

    body = resp.json()
    assert body["accepted"] is True
    assert body["deduplicated"] is True

    # NATS publish must NOT be called for duplicates
    app.state.stroke_publisher.publish_stroke_raw.assert_not_awaited()

    # Progress must still be queried from DB (not in-memory)
    app.state.upload_status_repo.get_pen_progress.assert_awaited_once()


# ---------------------------------------------------------------------------
# I-SINGEST-03: Invalid CRC returns 422
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_ingest_invalid_crc(client: AsyncClient):
    resp = await client.post(
        "/api/v1/strokes/ingest",
        json=_valid_body(checksum_crc32="00000000"),
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# I-SINGEST-04: Missing required field returns 422
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_ingest_missing_field(client: AsyncClient):
    body = _valid_body()
    del body["pen_mac"]
    resp = await client.post("/api/v1/strokes/ingest", json=body)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# I-SINGEST-05: NATS publish failure returns 503
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_ingest_nats_failure(client: AsyncClient, app: FastAPI):
    app.state.stroke_publisher.publish_stroke_raw = AsyncMock(
        side_effect=RuntimeError("NATS down"),
    )
    resp = await client.post("/api/v1/strokes/ingest", json=_valid_body())
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# I-SINGEST-06: DB write failure is non-fatal (still returns 202)
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_ingest_db_failure_non_fatal(client: AsyncClient, app: FastAPI):
    app.state.upload_status_repo.record_chunk = AsyncMock(
        side_effect=RuntimeError("DB down"),
    )
    resp = await client.post("/api/v1/strokes/ingest", json=_valid_body())
    assert resp.status_code == 202
    assert resp.json()["accepted"] is True


# ---------------------------------------------------------------------------
# I-SINGEST-07: Out-of-order upload reflects true cumulative progress
#   Upload chunk 0, then chunk 2 (skipping 1).
#   ACK for chunk 2 must show next_expected=1, NOT 3.
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_ingest_out_of_order_shows_correct_next(
    _mock_auth,
):
    """After chunk 0 and chunk 2, next_expected should be 1 (the gap)."""
    # Simulate DB state: chunks {0, 2} received out of 5
    progress_after_c2 = _pen_progress({0, 2}, total=5)
    app = _make_app(pen_progress=progress_after_c2)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            "/api/v1/strokes/ingest",
            json=_valid_body(chunk_index=2),
        )

    assert resp.status_code == 202
    body = resp.json()
    assert body["next_expected_chunk"] == 1
    assert body["pen_upload_complete"] is False


# ---------------------------------------------------------------------------
# I-SINGEST-08: Duplicate after prior progress shows cumulative state
#   Chunks 0-2 already received; duplicate of chunk 0 should reflect all 3.
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_ingest_duplicate_reflects_cumulative_progress(
    _mock_auth,
):
    """Duplicate of chunk 0 must still show 3 chunks received."""
    progress = _pen_progress({0, 1, 2}, total=5)
    app = _make_app(pen_progress=progress)
    app.state.idempotency_repo.check_and_mark = AsyncMock(return_value=False)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            "/api/v1/strokes/ingest",
            json=_valid_body(chunk_index=0),
        )

    assert resp.status_code == 202
    body = resp.json()
    assert body["deduplicated"] is True
    assert body["next_expected_chunk"] == 3
    assert body["pen_upload_complete"] is False


# ---------------------------------------------------------------------------
# I-SINGEST-09: All chunks uploaded — final ACK shows complete=true
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_ingest_all_chunks_complete(_mock_auth):
    """When all 5 chunks are persisted, pen_upload_complete must be True."""
    progress = _pen_progress({0, 1, 2, 3, 4}, total=5)
    app = _make_app(pen_progress=progress)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            "/api/v1/strokes/ingest",
            json=_valid_body(chunk_index=4, total_chunks=5),
        )

    assert resp.status_code == 202
    body = resp.json()
    assert body["pen_upload_complete"] is True
    # next_expected == total_chunks means nothing is missing
    assert body["next_expected_chunk"] == 5
