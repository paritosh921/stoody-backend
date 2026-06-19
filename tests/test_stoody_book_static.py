from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from api.v1.stoody_book_static import mount_stoody_book


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_stoody_book_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "stoody-book"
    assets = root / "web" / "assets"
    assets.mkdir(parents=True)
    (root / "web" / "index.html").write_text(
        "<!doctype html><title>Stoody Book</title><h1>Stoody Book</h1>",
        encoding="utf-8",
    )
    (assets / "stoody-book.css").write_text("body{color:#111827}", encoding="utf-8")
    return root


@pytest.mark.anyio
async def test_mount_serves_stoody_book_index_for_exact_and_slash_paths(tmp_path: Path):
    app = FastAPI()
    mount_stoody_book(app, _make_stoody_book_fixture(tmp_path))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        exact_response = await client.get("/stoody-book")
        slash_response = await client.get("/stoody-book/")

    assert exact_response.status_code == 200
    assert slash_response.status_code == 200
    assert "text/html" in exact_response.headers["content-type"]
    assert "Stoody Book" in exact_response.text
    assert slash_response.text == exact_response.text


@pytest.mark.anyio
async def test_mount_serves_stoody_book_assets(tmp_path: Path):
    app = FastAPI()
    mount_stoody_book(app, _make_stoody_book_fixture(tmp_path))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/stoody-book/assets/stoody-book.css")

    assert response.status_code == 200
    assert "body{color:#111827}" in response.text
