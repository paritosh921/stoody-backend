"""MinIO/S3 adapter for page image uploads.

Write order: S3 upload first, then PG metadata (per STATE_OWNERSHIP_MAP).
Key format: {exam_id}/{student_id}/page_{page_number}.svg
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from miniopy_async import Minio

from src.config import Settings


@dataclass
class UploadResult:
    """Result of an S3 upload operation."""

    bucket: str
    key: str
    etag: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


class S3Adapter:
    """Async MinIO/S3 client for page image storage."""

    def __init__(self, settings: Settings) -> None:
        self._client = Minio(
            settings.minio_url,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self._bucket = settings.minio_bucket

    async def ensure_bucket(self) -> None:
        """Create the bucket if it does not exist."""
        exists = await self._client.bucket_exists(self._bucket)
        if not exists:
            await self._client.make_bucket(self._bucket)

    def _page_key(
        self,
        exam_id: str,
        student_id: str,
        page_number: int,
    ) -> str:
        return f"{exam_id}/{student_id}/page_{page_number}.svg"

    async def upload_page_svg(
        self,
        exam_id: str,
        student_id: str,
        page_number: int,
        svg_content: str,
    ) -> UploadResult:
        """Upload SVG page content to S3.

        Returns the S3 path on success. This must be called BEFORE
        writing PG metadata (orphaned S3 is acceptable; dangling
        PG reference is not).
        """
        key = self._page_key(exam_id, student_id, page_number)
        data = svg_content.encode("utf-8")

        from io import BytesIO

        stream = BytesIO(data)

        result = await self._client.put_object(
            bucket_name=self._bucket,
            object_name=key,
            data=stream,
            length=len(data),
            content_type="image/svg+xml",
        )

        return UploadResult(
            bucket=self._bucket,
            key=key,
            etag=result.etag if result.etag else "",
        )
