from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

class PageResponse(BaseModel):
    pen_id: str
    page_no: int
    strokes: list


router = APIRouter(tags=["pages"])


def get_app_state():
    from ..main import app

    return app.state.app_state


@router.get("/v1/pages/{pen_id}/{page_no}", response_model=PageResponse)
async def get_page(pen_id: str, page_no: int, token: str | None = None, state=Depends(get_app_state)):
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid dashboard token")

    strokes = await state.storage.list_page_strokes(pen_id, page_no)
    return PageResponse(pen_id=pen_id, page_no=page_no, strokes=strokes)

