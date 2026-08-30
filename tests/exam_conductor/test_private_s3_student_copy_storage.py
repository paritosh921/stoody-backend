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

    def head_object(self, *, Bucket: str, Key: str):
        data = self.objects[(Bucket, Key)]
        return {"ContentLength": len(data)}

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


@pytest.mark.asyncio
async def test_failed_local_cleanup_retries_only_after_all_s3_assets_are_verified():
    from api.v1.evalpen_student_submission_async import (
        reconcile_answer_copy_local_cleanup,
    )
    from mongomock_motor import AsyncMongoMockClient
    from unittest.mock import AsyncMock, patch

    db = AsyncMongoMockClient()["skb_test"]
    await db["exampen_student_copy_uploads"].insert_one({
        "attempt_id": "attempt-cleanup",
        "storage_backend": "s3",
        "storage_handoff_status": "complete",
        "original_asset": {
            "storage_path": "s3://stoody-test/private/exampen/student-answer-copies/original.pdf",
        },
        "pages": [{
            "storage_path": "s3://stoody-test/private/exampen/student-answer-copies/page-1.png",
        }],
        "local_scan_cleanup_status": "failed",
        "local_scan_cleanup_pending_paths": ["C:/scan-stage/original.pdf"],
    })
    cleanup = AsyncMock(return_value=[])
    exists = AsyncMock(return_value=True)

    with (
        patch(
            "api.v1.evalpen_student_submission_async.private_object_exists",
            new=exists,
        ),
        patch(
            "api.v1.evalpen_student_submission_async._cleanup_released_student_copy_paths",
            new=cleanup,
        ),
    ):
        result = await reconcile_answer_copy_local_cleanup(db)

    attempt = await db["exampen_student_copy_uploads"].find_one(
        {"attempt_id": "attempt-cleanup"}
    )
    assert result["attempts_scanned"] == 1
    assert result["attempts_cleaned"] == 1
    assert result["attempts_deferred"] == 0
    assert exists.await_count == 2
    cleanup.assert_awaited_once_with(["C:/scan-stage/original.pdf"])
    assert attempt["local_scan_cleanup_status"] == "complete"
    assert "local_scan_cleanup_pending_paths" not in attempt


@pytest.mark.asyncio
async def test_failed_local_cleanup_is_retained_when_s3_cannot_be_verified():
    from api.v1.evalpen_student_submission_async import (
        reconcile_answer_copy_local_cleanup,
    )
    from mongomock_motor import AsyncMongoMockClient
    from unittest.mock import AsyncMock, patch

    db = AsyncMongoMockClient()["skb_test"]
    await db["exampen_student_copy_uploads"].insert_one({
        "attempt_id": "attempt-deferred",
        "storage_backend": "s3",
        "storage_handoff_status": "complete",
        "pages": [{
            "storage_path": "s3://stoody-test/private/exampen/student-answer-copies/page-1.png",
        }],
        "local_scan_cleanup_status": "failed",
        "local_scan_cleanup_pending_paths": ["C:/scan-stage/page-1.png"],
    })
    cleanup = AsyncMock(return_value=[])

    with (
        patch(
            "api.v1.evalpen_student_submission_async.private_object_exists",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "api.v1.evalpen_student_submission_async._cleanup_released_student_copy_paths",
            new=cleanup,
        ),
    ):
        result = await reconcile_answer_copy_local_cleanup(db)

    attempt = await db["exampen_student_copy_uploads"].find_one(
        {"attempt_id": "attempt-deferred"}
    )
    assert result["attempts_deferred"] == 1
    cleanup.assert_not_awaited()
    assert attempt["local_scan_cleanup_status"] == "failed"


@pytest.mark.asyncio
async def test_legacy_staff_scan_staging_is_removed_only_with_canonical_s3_evidence(
    tmp_path,
    monkeypatch,
):
    from api.v1.evalpen_student_submission_async import (
        reconcile_answer_copy_local_cleanup,
    )
    from config_async import settings
    from mongomock_motor import AsyncMongoMockClient
    from unittest.mock import AsyncMock, patch

    monkeypatch.setattr(settings, "UPLOAD_PRIVATE_LOCAL_DIR", tmp_path)
    upload_id = "legacy-upload"
    released = (
        tmp_path
        / settings.UPLOAD_RELEASED_PREFIX
        / "skb_test"
        / "student_answer_copy_pdf"
        / upload_id
        / "student-copy.pdf"
    )
    released.parent.mkdir(parents=True)
    released.write_bytes(b"temporary-scan-staging")
    released.with_name(f"{released.name}.metadata.json").write_text("{}", encoding="utf-8")

    db = AsyncMongoMockClient()["skb_test"]
    await db["upload_security_verdicts"].insert_one({
        "upload_id": upload_id,
        "policy_id": "student_answer_copy_pdf",
        "tenant_db": "skb_test",
        "storage_backend": "s3",
        "storage_transfer_status": "complete",
        "released_storage_path": (
            "s3://stoody-test/private/exampen/student-answer-copies/original.pdf"
        ),
        "purpose_metadata": {"exam_id": "exam-1", "student_id": "student-1"},
    })
    await db["evalpen_submissions"].insert_one({
        "submission_id": "submission-1",
        "exam_id": "exam-1",
        "student_id": "student-1",
    })
    await db["evalpen_answer_pages"].insert_one({
        "submission_id": "submission-1",
        "raw_image_ref": (
            "s3://stoody-test/private/exampen/student-answer-copies/page-1.png"
        ),
    })

    with patch(
        "api.v1.evalpen_student_submission_async.private_object_exists",
        new=AsyncMock(return_value=True),
    ):
        result = await reconcile_answer_copy_local_cleanup(db)

    verdict = await db["upload_security_verdicts"].find_one({"upload_id": upload_id})
    assert result["legacy_cleaned"] == 1
    assert verdict["local_scan_cleanup_status"] == "complete"
    assert not released.exists()
    assert not released.with_name(f"{released.name}.metadata.json").exists()
