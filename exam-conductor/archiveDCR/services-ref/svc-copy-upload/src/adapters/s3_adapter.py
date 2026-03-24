"""MinIO / S3 adapter for copy image storage."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from miniopy_async import Minio

from exampen_common.logging import get_logger

if TYPE_CHECKING:
    from src.config import Settings

_log = get_logger(__name__)


class S3Adapter:
    """Thin async wrapper around ``miniopy-async`` for copy images."""

    def __init__(
        self,
        client: Minio,
        bucket: str,
        presigned_expiry: int,
    ) -> None:
        self._client = client
        self._bucket = bucket
        self._presigned_expiry = presigned_expiry

    # -- public API --------------------------------------------------------

    async def ensure_bucket(self) -> None:
        """Create the bucket if it does not already exist."""
        exists = await self._client.bucket_exists(self._bucket)
        if not exists:
            await self._client.make_bucket(self._bucket)
            _log.info("created bucket %s", self._bucket)

    async def upload(
        self,
        key: str,
        data: bytes,
        content_type: str,
    ) -> str:
        """Upload *data* to S3 under *key* and return the object path.

        Key format: ``copies/{exam_id}/{student_id}/page_{n}.{ext}``
        """
        from io import BytesIO

        stream = BytesIO(data)
        await self._client.put_object(
            self._bucket,
            key,
            stream,
            length=len(data),
            content_type=content_type,
        )
        uri = f"s3://{self._bucket}/{key}"
        _log.info("uploaded %s (%d bytes)", uri, len(data))
        return uri

    async def presigned_get_url(self, key: str) -> str:
        """Generate a presigned GET URL for the given *key*."""
        url: str = await self._client.presigned_get_object(
            self._bucket,
            key,
            expires=timedelta(seconds=self._presigned_expiry),
        )
        return url

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def build_key(
        exam_id: str,
        student_id: str,
        page_number: int,
        extension: str,
    ) -> str:
        """Build the S3 object key for a copy image."""
        return (
            f"copies/{exam_id}/{student_id}/page_{page_number}.{extension}"
        )


def create_s3_adapter(settings: Settings) -> S3Adapter:
    """Factory that creates an :class:`S3Adapter` from settings."""
    secure = settings.minio_url.startswith("https://")
    endpoint = (
        settings.minio_url
        .removeprefix("https://")
        .removeprefix("http://")
    )
    client = Minio(
        endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=secure,
        region=settings.minio_region,
    )
    return S3Adapter(
        client=client,
        bucket=settings.minio_bucket,
        presigned_expiry=settings.presigned_url_expiry,
    )
