"""S3/MinIO adapter for downloading page images."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import aiobotocore.session

logger = logging.getLogger(__name__)


async def download_page_image(
    minio_url: str,
    access_key: str,
    secret_key: str,
    bucket: str,
    image_uri: str,
) -> bytes:
    """Download a page image from MinIO/S3.

    Parameters
    ----------
    minio_url:
        Endpoint URL, e.g. ``http://localhost:9000``.
    access_key, secret_key:
        MinIO credentials.
    bucket:
        S3 bucket name.
    image_uri:
        Full URI or object key for the page image.

    Returns
    -------
    Raw image bytes.
    """
    # Extract object key from full URI if needed
    parsed = urlparse(image_uri)
    if parsed.scheme in ("s3", "http", "https"):
        object_key = parsed.path.lstrip("/")
        # Remove bucket prefix if present
        if object_key.startswith(f"{bucket}/"):
            object_key = object_key[len(bucket) + 1:]
    else:
        object_key = image_uri

    session = aiobotocore.session.get_session()
    async with session.create_client(
        "s3",
        endpoint_url=minio_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    ) as client:
        response = await client.get_object(Bucket=bucket, Key=object_key)
        data = await response["Body"].read()
        logger.info(
            "Downloaded %d bytes from s3://%s/%s",
            len(data), bucket, object_key,
        )
        return data
