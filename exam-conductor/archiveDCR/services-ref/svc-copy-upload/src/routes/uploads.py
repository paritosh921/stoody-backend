"""HTTP routes for copy image upload and retrieval.

Endpoints match ``api/copy-upload.openapi.yaml``.
"""

from __future__ import annotations

from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from exampen_common.auth import ExamPenUser, get_current_user
from exampen_common.db import create_pool, rls_session, session_factory
from exampen_common.logging import get_logger

from src.adapters.s3_adapter import S3Adapter
from src.domain.upload_validator import (
    ValidationResult,
    extension_for_content_type,
    validate_magic_bytes,
    validate_upload,
)
from src.events.copy_publisher import publish_copy_ready
from src.storage.copy_repo import (
    get_copy_image,
    insert_copy_image,
    list_copies_for_student,
)

_log = get_logger(__name__)


def build_router() -> APIRouter:
    """Create and return the uploads router."""
    router = APIRouter()

    # ------------------------------------------------------------------
    # POST /exams/{exam_id}/copies/upload
    # ------------------------------------------------------------------

    @router.post(
        "/exams/{exam_id}/copies/upload",
        status_code=status.HTTP_201_CREATED,
    )
    async def upload_copy(
        request: Request,
        exam_id: str,
        student_id: str = Form(...),
        page_number: int = Form(...),
        captured_at: str = Form(...),
        image: UploadFile = File(...),
        user: ExamPenUser = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Upload one photographed answer page.

        Write order: S3 first, PG second.
        Publish ``copy.ready`` NATS event after PG commit.
        """
        content = await image.read()
        file_size = len(content)
        content_type = image.content_type or ""

        # --- domain validation (pure, no I/O) ---
        meta_result: ValidationResult = validate_upload(
            exam_id=exam_id,
            student_id=student_id,
            page_number=page_number,
            file_size=file_size,
            content_type=content_type,
        )
        if not meta_result.valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=meta_result.error,
            )

        magic_result = validate_magic_bytes(content[:8])
        if not magic_result.valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=magic_result.error,
            )

        ext = extension_for_content_type(content_type)
        s3: S3Adapter = request.app.state.s3
        settings = request.app.state.settings

        # --- S3 write FIRST ---
        key = S3Adapter.build_key(exam_id, student_id, page_number, ext)
        await s3.ensure_bucket()
        s3_uri = await s3.upload(key, content, content_type)

        # --- PG metadata SECOND ---
        engine = create_pool(url=settings.database_url)
        sf = session_factory(engine)
        try:
            async for session in rls_session(sf, user.tenant_id):
                await insert_copy_image(
                    session,
                    exam_id=exam_id,
                    student_id=student_id,
                    page_number=page_number,
                    s3_path=key,
                    content_type=content_type,
                    file_size=file_size,
                    uploaded_by=user.user_id,
                    tenant_id=user.tenant_id,
                )
        finally:
            await engine.dispose()

        # --- NATS publish after PG commit ---
        nats_client = request.app.state.nats
        await publish_copy_ready(
            nats_client,
            exam_id=exam_id,
            student_id=student_id,
            page_number=page_number,
            copy_image_uri=s3_uri,
        )

        return {
            "exam_id": exam_id,
            "student_id": student_id,
            "page_number": page_number,
            "copy_image_uri": s3_uri,
            "data_source": "copy_image",
        }

    # ------------------------------------------------------------------
    # GET /exams/{exam_id}/copies/{student_id}
    # ------------------------------------------------------------------

    @router.get("/exams/{exam_id}/copies/{student_id}")
    async def list_copies(
        request: Request,
        exam_id: str,
        student_id: str,
        user: ExamPenUser = Depends(get_current_user),
    ) -> dict[str, Any]:
        """List uploaded copy pages for one student."""
        settings = request.app.state.settings
        s3: S3Adapter = request.app.state.s3
        engine = create_pool(url=settings.database_url)
        sf = session_factory(engine)
        try:
            async for session in rls_session(sf, user.tenant_id):
                rows = await list_copies_for_student(
                    session, exam_id=exam_id, student_id=student_id,
                )
        finally:
            await engine.dispose()

        items = []
        for row in rows:
            url = await s3.presigned_get_url(row["s3_path"])
            items.append({
                "page_number": row["page_number"],
                "copy_image_uri": url,
                "authoritative_source": "copy_image",
            })
        return {"items": items}

    # ------------------------------------------------------------------
    # GET /exams/{exam_id}/copies/{student_id}/{page_number}
    # ------------------------------------------------------------------

    @router.get(
        "/exams/{exam_id}/copies/{student_id}/{page_number}",
    )
    async def get_copy(
        request: Request,
        exam_id: str,
        student_id: str,
        page_number: int,
        user: ExamPenUser = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Get a specific copy image as a presigned URL."""
        settings = request.app.state.settings
        s3: S3Adapter = request.app.state.s3
        engine = create_pool(url=settings.database_url)
        sf = session_factory(engine)
        try:
            async for session in rls_session(sf, user.tenant_id):
                row = await get_copy_image(
                    session,
                    exam_id=exam_id,
                    student_id=student_id,
                    page_number=page_number,
                )
        finally:
            await engine.dispose()

        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Copy image not found",
            )

        url = await s3.presigned_get_url(row["s3_path"])
        return {
            "page_number": row["page_number"],
            "copy_image_uri": url,
            "authoritative_source": "copy_image",
        }

    return router
