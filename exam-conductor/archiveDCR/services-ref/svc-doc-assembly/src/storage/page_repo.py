"""PostgreSQL metadata storage for assembled pages.

Write order: S3 upload first, then PG metadata (per STATE_OWNERSHIP_MAP).
Orphaned S3 objects are acceptable. Dangling PG references are not.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from src.domain.models import MissAutoState, PageDocument, QuestionResult

logger = logging.getLogger(__name__)


class PageRepository:
    """Async repository for assembled_pages table."""

    def __init__(self, database_url: str) -> None:
        self._engine: AsyncEngine | None = None
        self._database_url = database_url

    async def connect(self) -> None:
        self._engine = create_async_engine(
            self._database_url,
            pool_size=5,
            max_overflow=5,
        )

    async def disconnect(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()

    async def save_page(
        self,
        doc: PageDocument,
        s3_uri: str,
        page_id: str,
    ) -> None:
        """Persist assembled page metadata.

        Must be called AFTER successful S3 upload.
        """
        if self._engine is None:
            raise RuntimeError("PageRepository not connected")

        question_results_json = json.dumps(
            [
                {
                    "question_id": qr.question_id,
                    "auto_state": qr.auto_state.value,
                    "override_state": (
                        qr.override_state.value if qr.override_state else None
                    ),
                }
                for qr in doc.question_results
            ]
        )

        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO assembled_pages (
                        page_id,
                        exam_id,
                        student_id,
                        page_number,
                        s3_uri,
                        question_results,
                        page_width_mm,
                        page_height_mm,
                        assembled_at
                    ) VALUES (
                        :page_id,
                        :exam_id,
                        :student_id,
                        :page_number,
                        :s3_uri,
                        :question_results::jsonb,
                        :page_width_mm,
                        :page_height_mm,
                        :assembled_at
                    )
                    ON CONFLICT (exam_id, student_id, page_number)
                    DO UPDATE SET
                        s3_uri = EXCLUDED.s3_uri,
                        question_results = EXCLUDED.question_results,
                        assembled_at = EXCLUDED.assembled_at
                    """
                ),
                {
                    "page_id": page_id,
                    "exam_id": doc.exam_id,
                    "student_id": doc.student_id,
                    "page_number": doc.page_number,
                    "s3_uri": s3_uri,
                    "question_results": question_results_json,
                    "page_width_mm": doc.page_width_mm,
                    "page_height_mm": doc.page_height_mm,
                    "assembled_at": datetime.now(timezone.utc),
                },
            )

        logger.info(
            "Saved page metadata: exam=%s student=%s page=%d",
            doc.exam_id,
            doc.student_id,
            doc.page_number,
        )
