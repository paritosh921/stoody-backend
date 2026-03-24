"""PostgreSQL repository for copy image metadata.

Write order: S3 first, PG second (see STATE_OWNERSHIP_MAP.md §1).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from exampen_common.logging import get_logger

_log = get_logger(__name__)


async def insert_copy_image(
    session: AsyncSession,
    *,
    exam_id: str,
    student_id: str,
    page_number: int,
    s3_path: str,
    content_type: str,
    file_size: int,
    uploaded_by: str,
    tenant_id: str,
) -> dict[str, Any]:
    """Insert copy image metadata row. Called AFTER S3 write succeeds."""
    now = datetime.now(timezone.utc)
    result = await session.execute(
        text("""
            INSERT INTO copy_images
                (exam_id, student_id, page_number, tenant_id,
                 s3_path, content_type, file_size,
                 uploaded_at, uploaded_by)
            VALUES
                (:exam_id, :student_id, :page_number, :tenant_id,
                 :s3_path, :content_type, :file_size,
                 :uploaded_at, :uploaded_by)
            ON CONFLICT (exam_id, student_id, page_number)
            DO UPDATE SET
                s3_path = EXCLUDED.s3_path,
                content_type = EXCLUDED.content_type,
                file_size = EXCLUDED.file_size,
                uploaded_at = EXCLUDED.uploaded_at,
                uploaded_by = EXCLUDED.uploaded_by
            RETURNING exam_id, student_id, page_number, s3_path, uploaded_at
        """),
        {
            "exam_id": exam_id,
            "student_id": student_id,
            "page_number": page_number,
            "tenant_id": tenant_id,
            "s3_path": s3_path,
            "content_type": content_type,
            "file_size": file_size,
            "uploaded_at": now,
            "uploaded_by": uploaded_by,
        },
    )
    row = result.mappings().one()
    _log.info(
        "inserted copy_image exam=%s student=%s page=%d tenant=%s",
        exam_id, student_id, page_number, tenant_id,
    )
    return dict(row)


async def list_copies_for_student(
    session: AsyncSession,
    *,
    exam_id: str,
    student_id: str,
) -> list[dict[str, Any]]:
    """List all copy image rows for one student in an exam."""
    result = await session.execute(
        text("""
            SELECT page_number, s3_path, content_type, file_size,
                   uploaded_at, uploaded_by
            FROM copy_images
            WHERE exam_id = :exam_id AND student_id = :student_id
            ORDER BY page_number
        """),
        {"exam_id": exam_id, "student_id": student_id},
    )
    return [dict(row) for row in result.mappings().all()]


async def get_copy_image(
    session: AsyncSession,
    *,
    exam_id: str,
    student_id: str,
    page_number: int,
) -> dict[str, Any] | None:
    """Get a single copy image metadata row."""
    result = await session.execute(
        text("""
            SELECT page_number, s3_path, content_type, file_size,
                   uploaded_at, uploaded_by
            FROM copy_images
            WHERE exam_id = :exam_id
              AND student_id = :student_id
              AND page_number = :page_number
        """),
        {
            "exam_id": exam_id,
            "student_id": student_id,
            "page_number": page_number,
        },
    )
    row = result.mappings().one_or_none()
    return dict(row) if row else None
