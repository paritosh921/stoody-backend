"""GET /api/v1/exams/{exam_id}/upload-status -- Per-pen reconciliation.

Returns per-pen upload progress so hub-uplink can resume after
disconnect. Response matches ``ExamUploadStatus`` schema.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from exampen_common.logging import get_logger

from src.adapters.auth_adapter import ExamPenUser, get_current_user

_log = get_logger(__name__)

router = APIRouter()

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


@router.get(
    "/{exam_id}/upload-status",
    summary="Return per-pen upload progress for reconciliation",
)
async def get_upload_status(
    exam_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_current_user),
) -> dict[str, Any]:
    """Query upload reconciliation state for an exam."""

    if not _UUID_RE.match(exam_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="exam_id must be a valid UUID",
        )

    status_repo = request.app.state.upload_status_repo

    try:
        pens = await status_repo.get_exam_status(exam_id)
    except Exception as exc:
        _log.error("DB query failed for exam %s: %s", exam_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Upload status unavailable; retry later",
        ) from exc

    return {
        "exam_id": exam_id,
        "pens": pens,
    }
