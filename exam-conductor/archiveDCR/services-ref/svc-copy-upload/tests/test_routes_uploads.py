"""Integration tests for upload routes (S3 mocked).

Test IDs: I-COPY-01 through I-COPY-05
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.main import app

# ---------------------------------------------------------------------------
# JPEG test fixture (minimal valid JPEG: SOI + APP0 marker + EOI)
# ---------------------------------------------------------------------------

_JPEG_HEADER = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
_JPEG_BODY = _JPEG_HEADER + b"\xff\xd9"

_PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_user() -> MagicMock:
    user = MagicMock()
    user.user_id = "user-001"
    user.tenant_id = "tenant-001"
    return user


@pytest.fixture(autouse=True)
def _patch_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override auth dependency to return a mock user."""
    from exampen_common.auth import get_current_user

    async def _fake_user() -> MagicMock:
        return _mock_user()

    app.dependency_overrides[get_current_user] = _fake_user


@pytest.fixture(autouse=True)
def _patch_s3() -> Any:
    """Replace S3 adapter on app state with an async mock."""
    mock_s3 = AsyncMock()
    mock_s3.ensure_bucket = AsyncMock()
    mock_s3.upload = AsyncMock(return_value="s3://exampen-copies/copies/e/s/page_1.jpg")
    mock_s3.presigned_get_url = AsyncMock(return_value="https://minio/presigned/page_1.jpg")
    mock_s3.build_key = MagicMock(return_value="copies/e/s/page_1.jpg")
    app.state.s3 = mock_s3
    return mock_s3


@pytest.fixture(autouse=True)
def _patch_nats() -> Any:
    """Replace NATS client on app state with an async mock."""
    mock_nats = AsyncMock()
    mock_nats.publish = AsyncMock()
    mock_nats.close = AsyncMock()
    app.state.nats = mock_nats
    return mock_nats


@pytest.fixture(autouse=True)
def _patch_settings() -> None:
    """Provide settings on app state."""
    settings = MagicMock()
    settings.database_url = "postgresql+asyncpg://test:test@localhost/test"
    app.state.settings = settings


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Upload tests
# ---------------------------------------------------------------------------


class TestUploadCopy:
    """I-COPY-01..03 — upload endpoint integration."""

    @pytest.mark.asyncio
    @patch("src.routes.uploads.create_pool")
    @patch("src.routes.uploads.session_factory")
    @patch("src.routes.uploads.rls_session")
    @patch("src.routes.uploads.insert_copy_image", new_callable=AsyncMock)
    @patch("src.routes.uploads.publish_copy_ready", new_callable=AsyncMock)
    async def test_successful_upload(
        self,
        mock_publish: AsyncMock,
        mock_insert: AsyncMock,
        mock_rls: MagicMock,
        mock_sf: MagicMock,
        mock_pool: MagicMock,
        client: AsyncClient,
    ) -> None:
        """I-COPY-01: Successful JPEG upload returns 201 with copy_image data_source."""
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()
        mock_pool.return_value = mock_engine

        mock_session = AsyncMock()
        mock_insert.return_value = {
            "exam_id": "e-1",
            "student_id": "s-1",
            "page_number": 1,
            "s3_path": "copies/e-1/s-1/page_1.jpg",
            "uploaded_at": "2026-03-19T00:00:00Z",
        }

        async def _fake_rls(*args: Any, **kwargs: Any) -> AsyncGenerator:
            yield mock_session

        mock_rls.return_value = _fake_rls()

        response = await client.post(
            "/api/v1/exams/e-1/copies/upload",
            data={
                "student_id": "s-1",
                "page_number": "1",
                "captured_at": "2026-03-19T10:00:00Z",
            },
            files={"image": ("page1.jpg", BytesIO(_JPEG_BODY), "image/jpeg")},
        )

        assert response.status_code == 201
        body = response.json()
        assert body["exam_id"] == "e-1"
        assert body["data_source"] == "copy_image"
        assert body["page_number"] == 1

    @pytest.mark.asyncio
    async def test_reject_gif(self, client: AsyncClient) -> None:
        """I-COPY-02: Reject unsupported image format (GIF)."""
        response = await client.post(
            "/api/v1/exams/e-1/copies/upload",
            data={
                "student_id": "s-1",
                "page_number": "1",
                "captured_at": "2026-03-19T10:00:00Z",
            },
            files={"image": ("page1.gif", BytesIO(b"GIF89a"), "image/gif")},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_reject_oversized(self, client: AsyncClient) -> None:
        """I-COPY-03: Reject files over 10 MB."""
        big = _JPEG_HEADER + (b"\x00" * (11 * 1024 * 1024))
        response = await client.post(
            "/api/v1/exams/e-1/copies/upload",
            data={
                "student_id": "s-1",
                "page_number": "1",
                "captured_at": "2026-03-19T10:00:00Z",
            },
            files={"image": ("page1.jpg", BytesIO(big), "image/jpeg")},
        )
        assert response.status_code == 400
        assert "10 MB" in response.json()["detail"]


# ---------------------------------------------------------------------------
# List / Get tests
# ---------------------------------------------------------------------------


class TestListAndGetCopies:
    """I-COPY-04..05 — list and get endpoints."""

    @pytest.mark.asyncio
    @patch("src.routes.uploads.create_pool")
    @patch("src.routes.uploads.session_factory")
    @patch("src.routes.uploads.rls_session")
    @patch("src.routes.uploads.list_copies_for_student", new_callable=AsyncMock)
    async def test_list_copies(
        self,
        mock_list: AsyncMock,
        mock_rls: MagicMock,
        mock_sf: MagicMock,
        mock_pool: MagicMock,
        client: AsyncClient,
    ) -> None:
        """I-COPY-04: List copies returns items array."""
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()
        mock_pool.return_value = mock_engine

        mock_list.return_value = [
            {"page_number": 1, "s3_path": "copies/e/s/page_1.jpg"},
            {"page_number": 2, "s3_path": "copies/e/s/page_2.jpg"},
        ]

        async def _fake_rls(*args: Any, **kwargs: Any) -> AsyncGenerator:
            yield AsyncMock()

        mock_rls.return_value = _fake_rls()

        response = await client.get("/api/v1/exams/e-1/copies/s-1")
        assert response.status_code == 200
        body = response.json()
        assert len(body["items"]) == 2
        assert body["items"][0]["page_number"] == 1

    @pytest.mark.asyncio
    @patch("src.routes.uploads.create_pool")
    @patch("src.routes.uploads.session_factory")
    @patch("src.routes.uploads.rls_session")
    @patch("src.routes.uploads.get_copy_image", new_callable=AsyncMock)
    async def test_get_copy_not_found(
        self,
        mock_get: AsyncMock,
        mock_rls: MagicMock,
        mock_sf: MagicMock,
        mock_pool: MagicMock,
        client: AsyncClient,
    ) -> None:
        """I-COPY-05: 404 when copy image does not exist."""
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()
        mock_pool.return_value = mock_engine
        mock_get.return_value = None

        async def _fake_rls(*args: Any, **kwargs: Any) -> AsyncGenerator:
            yield AsyncMock()

        mock_rls.return_value = _fake_rls()

        response = await client.get("/api/v1/exams/e-1/copies/s-1/99")
        assert response.status_code == 404
