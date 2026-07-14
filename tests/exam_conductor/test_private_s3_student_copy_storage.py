"""Regression coverage for private S3-backed student answer-copy storage."""

from __future__ import annotations

import io

import pytest


class _FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.put_calls: list[dict] = []

    def put_object(self, **kwargs):
        self.put_calls.append(kwargs)
        self.objects[(kwargs["Bucket"], kwargs["Key"])] = kwargs["Body"]

    def get_object(self, *, Bucket: str, Key: str):
        data = self.objects[(Bucket, Key)]
        return {"ContentLength": len(data), "Body": io.BytesIO(data)}

    def delete_object(self, *, Bucket: str, Key: str):
        self.objects.pop((Bucket, Key), None)

    def generate_presigned_url(self, operation: str, *, Params: dict, ExpiresIn: int):
        return f"https://private.test/{Params['Bucket']}/{Params['Key']}?expires={ExpiresIn}"


@pytest.mark.asyncio
async def test_private_student_copy_object_uses_s3_only_with_encryption(monkeypatch):
    from utils import s3_storage

    fake_s3 = _FakeS3()
    monkeypatch.setattr(s3_storage, "USE_S3_STORAGE", True)
    monkeypatch.setattr(s3_storage, "S3_BUCKET_NAME", "stoody-test")
    monkeypatch.setattr(s3_storage, "_s3_client", fake_s3)

    path = await s3_storage.upload_private_object(
        b"student-answer-page",
        object_key="private/exampen/student-answer-copies/tenant/exam/attempt/page-1.png",
        content_type="image/png",
        metadata={"purpose": "exam_answer_copy"},
    )

    assert path == (
        "s3://stoody-test/private/exampen/student-answer-copies/"
        "tenant/exam/attempt/page-1.png"
    )
    assert fake_s3.put_calls[0]["ServerSideEncryption"] == "AES256"
    assert fake_s3.put_calls[0]["CacheControl"] == "private, no-store, max-age=0"

    loaded = await s3_storage.download_private_object(
        path,
        allowed_key_prefix="private/exampen/",
    )
    assert loaded == b"student-answer-page"
    assert s3_storage.create_private_download_url(
        path,
        allowed_key_prefix="private/exampen/",
        expires_in=300,
    ).startswith("https://private.test/")

    await s3_storage.delete_private_object(
        path,
        allowed_key_prefix="private/exampen/",
    )
    assert fake_s3.objects == {}


@pytest.mark.asyncio
async def test_private_student_copy_storage_never_falls_back_to_ec2(monkeypatch):
    from utils import s3_storage

    monkeypatch.setattr(s3_storage, "USE_S3_STORAGE", False)
    monkeypatch.setattr(s3_storage, "S3_BUCKET_NAME", "stoody-test")
    monkeypatch.setattr(s3_storage, "_s3_client", None)

    with pytest.raises(s3_storage.PrivateObjectStorageError, match="disabled"):
        await s3_storage.upload_private_object(
            b"student-answer-page",
            object_key="private/exampen/student-answer-copies/tenant/exam/attempt/page-1.png",
            content_type="image/png",
        )
