"""Stroke chunk upload and upload-status endpoints.

Routes are mounted at ``/api/v1/exampen/strokes``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from exampen.dcr.core.auth_bridge import (
    ExamPenUser,
    get_exampen_user,
    require_exampen_role,
)
from exampen.dcr.domain.chunk_validator import (
    make_idempotency_key,
    validate_chunk,
)
from exampen.dcr.storage.stroke_raw_repo import StrokeRawRepo

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_tenant_db(request: Request, user: ExamPenUser):
    db = await request.app.state.db.get_tenant_db(user.tenant_id)
    if db is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database unavailable")
    return db


async def _publish_event(request: Request, subject: str, data: dict) -> None:
    nats = getattr(request.app.state, "exampen_nats", None)
    if nats is None or not nats.is_connected:
        return
    try:
        await nats.publish(subject, data)
    except Exception:
        logger.warning("NATS publish to %s failed (non-fatal)", subject)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/ingest",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest one uploaded chunk from the hub",
)
async def ingest_chunk(
    body: dict[str, Any],
    request: Request,
    user: ExamPenUser = Depends(get_exampen_user),
) -> dict[str, Any]:
    """Accept a stroke chunk, validate CRC, deduplicate, and publish to NATS.

    Steps:
    1. Domain validation (CRC-32, required fields).
    2. Idempotent insert into ``exampen_strokes_raw`` collection.
    3. Publish ``exampen.stroke.raw`` event to NATS (best-effort).
    4. Return ACK with cumulative upload progress.
    """
    # 1. Domain validation
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

    db = await _get_tenant_db(request, user)
    repo = StrokeRawRepo(db)

    # 2. Idempotent insert
    is_new = await repo.record_chunk(
        exam_id=exam_id,
        pen_mac=pen_mac,
        chunk_index=chunk_index,
        tenant_id=user.tenant_id,
        data={
            "total_chunks": total_chunks,
            "payload_base64": body.get("payload_base64", ""),
            "checksum_crc32": body.get("checksum_crc32", ""),
            "upload_path": body.get("upload_path", ""),
            "binding_status": body.get("binding_status"),
        },
    )

    # 3. Publish NATS event
    if is_new:
        await _publish_event(request, "exampen.stroke.raw", {
            "exam_id": exam_id,
            "pen_mac": pen_mac,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "tenant_id": user.tenant_id,
        })

    # 4. Query cumulative progress for the pen
    progress = await repo.get_pen_progress(exam_id, pen_mac, user.tenant_id)
    received = progress.get("received_chunks", [])
    pen_complete = len(received) >= total_chunks

    next_expected = 0
    if received:
        received_set = set(received)
        while next_expected in received_set:
            next_expected += 1

    logger.info(
        "chunk %s: exam=%s pen=%s chunk=%d/%d new=%s",
        "accepted" if is_new else "deduplicated",
        exam_id, pen_mac, chunk_index, total_chunks, is_new,
    )

    return {
        "exam_id": exam_id,
        "pen_mac": pen_mac,
        "chunk_index": chunk_index,
        "accepted": True,
        "deduplicated": not is_new,
        "next_expected_chunk": next_expected,
        "pen_upload_complete": pen_complete,
    }


@router.get("/{exam_id}/upload-status")
async def get_upload_status(
    exam_id: str,
    request: Request,
    user: ExamPenUser = Depends(
        require_exampen_role("principal", "hod", "invigilator", "evaluator")
    ),
) -> dict[str, Any]:
    """Per-pen upload reconciliation for an exam."""
    db = await _get_tenant_db(request, user)
    repo = StrokeRawRepo(db)
    pens = await repo.get_exam_upload_status(exam_id, user.tenant_id)
    return {"exam_id": exam_id, "pens": pens}
