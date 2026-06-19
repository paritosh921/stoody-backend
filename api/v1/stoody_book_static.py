from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def _default_stoody_book_root() -> Path:
    return Path(__file__).resolve().parents[2] / "stoody-book"


def mount_stoody_book(app: FastAPI, root_dir: Path | None = None) -> None:
    root = root_dir or _default_stoody_book_root()
    web_dir = root / "web"
    assets_dir = web_dir / "assets"
    index_file = web_dir / "index.html"

    router = APIRouter()

    @router.get("/stoody-book", include_in_schema=False)
    @router.get("/stoody-book/", include_in_schema=False)
    async def stoody_book_index():
        return FileResponse(index_file, media_type="text/html")

    if assets_dir.exists():
        app.mount(
            "/stoody-book/assets",
            StaticFiles(directory=str(assets_dir)),
            name="stoody-book-assets",
        )
    app.include_router(router)
