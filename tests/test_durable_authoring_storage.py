from types import SimpleNamespace

import pytest
from mongomock_motor import AsyncMongoMockClient

from core.upload_security.storage import PrivateUploadStorage


@pytest.mark.asyncio
async def test_clean_authoring_pdf_is_promoted_audited_and_removed(monkeypatch, tmp_path):
    import core.upload_security.durable_authoring as durable

    db = AsyncMongoMockClient()["skb_test"]
    storage = PrivateUploadStorage(local_root=tmp_path)
    released = await storage.write_released_bytes(
        data=b"%PDF-1.4 durable",
        tenant="skb_test",
        policy_id="pdf_document",
        upload_id="upload-1",
        safe_filename="paper.pdf",
        content_type="application/pdf",
    )
    await db["upload_security_verdicts"].insert_one(
        {"upload_id": "upload-1", "released_storage_path": released}
    )

    async def fake_upload(data, *, object_key, content_type, metadata=None):
        assert object_key.startswith("private/content/authoring/skb_test/doc-1/source_document/")
        assert content_type == "application/pdf"
        return f"s3://private-bucket/{object_key}"

    monkeypatch.setattr(durable, "upload_private_object", fake_upload)
    monkeypatch.setattr(durable, "PrivateUploadStorage", lambda: storage)

    import hashlib

    payload = b"%PDF-1.4 durable"
    clean = SimpleNamespace(
        released_storage_path=released,
        sha256=hashlib.sha256(payload).hexdigest(),
        original_filename="paper.pdf",
        content_type="application/pdf",
        upload_id="upload-1",
        bytes=payload,
    )
    promotion = await durable.promote_clean_authoring_pdf(
        db,
        clean,
        tenant_db="skb_test",
        document_id="doc-1",
        artifact_role="source_document",
    )

    assert promotion.storage_uri.startswith("s3://private-bucket/")
    assert not tmp_path.joinpath("clean", "skb_test", "pdf_document", "upload-1", "paper.pdf").exists()
    verdict = await db["upload_security_verdicts"].find_one({"upload_id": "upload-1"})
    assert verdict["storage_backend"] == "s3"
    assert verdict["released_storage_path"] == promotion.storage_uri


@pytest.mark.asyncio
async def test_failed_private_upload_keeps_local_released_file(monkeypatch, tmp_path):
    import core.upload_security.durable_authoring as durable

    storage = PrivateUploadStorage(local_root=tmp_path)
    released = await storage.write_released_bytes(
        data=b"%PDF-1.4 recoverable",
        tenant="skb_test",
        policy_id="answer_sheet_pdf",
        upload_id="upload-2",
        safe_filename="solution.pdf",
        content_type="application/pdf",
    )

    async def failed_upload(*args, **kwargs):
        raise RuntimeError("S3 unavailable")

    monkeypatch.setattr(durable, "upload_private_object", failed_upload)

    import hashlib

    with pytest.raises(RuntimeError, match="S3 unavailable"):
        await durable.stage_released_authoring_pdf(
            released_path=released,
            expected_sha256=hashlib.sha256(b"%PDF-1.4 recoverable").hexdigest(),
            filename="solution.pdf",
            content_type="application/pdf",
            tenant_db="skb_test",
            document_id="doc-2",
            artifact_role="teacher_solution",
            upload_id="upload-2",
            storage=storage,
        )

    assert tmp_path.joinpath("clean", "skb_test", "answer_sheet_pdf", "upload-2", "solution.pdf").exists()
