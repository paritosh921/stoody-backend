from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from models.raw_frames import RawFrameIn, RawFrameBatchIn

router = APIRouter()
logger = logging.getLogger(__name__)


def get_app_state(request: Request):
    try:
        return request.app.state.app_state
    except AttributeError as exc:
        raise RuntimeError("Application state not configured") from exc


async def _ingest_single_frame(frame: RawFrameIn, state) -> None:
    pen_id = await state.pen_router.resolve_pen_id(frame.pen_mac)
    canonical = frame.to_canonical(pen_id)
    await state.pen_workers.enqueue(canonical)


@router.post("/v1/pen-frame")
async def ingest_pen_frame(
    frame: RawFrameIn,
    request: Request,
    state=Depends(get_app_state),
):
    """Ingress endpoint for Pi hubs (single frame)."""

    try:
        await _ingest_single_frame(frame, state)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    logger.debug(
        "Ingested frame pen=%s hub=%s seq=%s", frame.pen_mac, frame.hub_id, frame.seq
    )
    return {"status": "ok"}


@router.post("/v1/pen-frame-batch")
async def ingest_pen_frame_batch(
    batch: RawFrameBatchIn,
    request: Request,
    state=Depends(get_app_state),
):
    accepted = 0
    errors = []

    for frame in batch.frames:
        try:
            await _ingest_single_frame(frame, state)
            accepted += 1
        except LookupError as exc:
            errors.append({"pen_mac": frame.pen_mac, "error": str(exc)})
        except Exception as exc:
            logger.exception("Failed to process frame from batch %s: %s", batch.batch_id, exc)
            errors.append({"pen_mac": frame.pen_mac, "error": str(exc)})

    return {"status": "ok", "accepted": accepted, "errors": errors}

