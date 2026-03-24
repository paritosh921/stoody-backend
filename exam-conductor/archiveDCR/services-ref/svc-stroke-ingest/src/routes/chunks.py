"""POST /api/v1/strokes/ingest -- Accept a single chunk upload from hub.

Flow:
1. Validate CRC-32 and required fields (domain layer)
2. Check idempotency key (Redis) -- duplicates get 202 with deduplicated=True
3. Publish ``stroke.raw`` event to NATS JetStream
4. Record chunk in upload_progress (PostgreSQL)
5. Query persisted progress and return ACK with cumulative state

Backpressure: if NATS publish fails, return 503 so the hub retries.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from exampen_common.logging import get_logger

from src.adapters.auth_adapter import ExamPenUser, get_current_user
from src.domain.chunk_validator import make_idempotency_key, validate_chunk
from src.events.stroke_publisher import build_stroke_raw_event
from src.storage.upload_status_repo import PenProgress

_log = get_logger(__name__)

router = APIRouter()


async def _build_progress_ack(
    status_repo: Any,
    exam_id: str,
    pen_mac: str,
    chunk_index: int,
    total_chunks: int,
) -> dict[str, int | bool]:
    """Query PostgreSQL for cumulative pen progress and derive ACK fields.

    Falls back to single-chunk estimate when the DB query fails.
    """
    try:
        pen = await status_repo.get_pen_progress(exam_id, pen_mac)
    except Exception:
        _log.warning(
            "progress query failed for %s/%s; falling back to single-chunk",
            exam_id, pen_mac,
        )
        pen = None

    if pen is not None:
        return {
            "next_expected_chunk": pen.next_expected_chunk,
            "pen_upload_complete": pen.complete,
        }

    # Fallback: assume only this chunk exists (degraded accuracy)
    fallback = PenProgress(
        pen_mac=pen_mac,
        total_chunks=total_chunks,
        received_indices=frozenset({chunk_index}),
    )
    return {
        "next_expected_chunk": fallback.next_expected_chunk,
        "pen_upload_complete": fallback.complete,
    }


@router.post(
    "/ingest",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest one uploaded chunk from the hub",
)
async def ingest_chunk(
    request: Request,
    body: dict[str, Any],
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Accept a stroke chunk, validate, deduplicate, and publish."""

    # ---- 1. Domain validation -----------------------------------------
    validation = validate_chunk(body)
    if not validation.valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[
                {"field": e.field, "message": e.message}
                for e in validation.errors
            ],
        )

    exam_id: str = body["exam_id"]
    pen_mac: str = body["pen_mac"]
    chunk_index: int = body["chunk_index"]
    total_chunks: int = body["total_chunks"]
    status_repo = request.app.state.upload_status_repo

    # ---- 2. Idempotency check -----------------------------------------
    idem_key = make_idempotency_key(exam_id, pen_mac, chunk_index)
    idem_repo = request.app.state.idempotency_repo

    is_new = await idem_repo.check_and_mark(idem_key)

    if not is_new:
        # Duplicate -- already persisted; query cumulative progress
        _log.info("duplicate chunk: %s", idem_key)
        progress = await _build_progress_ack(
            status_repo, exam_id, pen_mac, chunk_index, total_chunks,
        )
        return {
            "exam_id": exam_id,
            "pen_mac": pen_mac,
            "chunk_index": chunk_index,
            "accepted": True,
            "deduplicated": True,
            **progress,
        }

    # ---- 3. Publish stroke.raw to NATS --------------------------------
    event = build_stroke_raw_event(body)
    publisher = request.app.state.stroke_publisher

    try:
        await publisher.publish_stroke_raw(event)
    except Exception as exc:
        _log.error("NATS publish failed for %s: %s", idem_key, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Upstream publish failed; retry later",
        ) from exc

    # ---- 4. Record in upload_progress DB ------------------------------
    try:
        await status_repo.record_chunk(
            exam_id, pen_mac, chunk_index, total_chunks,
        )
    except Exception:
        # DB record is best-effort; NATS publish was the real write.
        # Upload-status can be rebuilt from NATS replay if needed.
        _log.warning(
            "upload_progress DB write failed for %s (non-fatal)", idem_key,
        )

    # ---- 5. Build ACK from persisted cumulative state -----------------
    progress = await _build_progress_ack(
        status_repo, exam_id, pen_mac, chunk_index, total_chunks,
    )

    _log.info(
        "chunk accepted: exam=%s pen=%s chunk=%d/%d",
        exam_id, pen_mac, chunk_index, total_chunks,
    )

    return {
        "exam_id": exam_id,
        "pen_mac": pen_mac,
        "chunk_index": chunk_index,
        "accepted": True,
        "deduplicated": False,
        **progress,
    }
