from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from services import canonical_asset_storage as storage


def test_linux_private_upload_path_rebases_to_configured_windows_root(tmp_path):
    backend_root = tmp_path / "backend"
    private_root = backend_root / "data" / "private_uploads"
    expected = (
        private_root
        / "clean"
        / "skb_sgtb-0001"
        / "pdf_document"
        / "upload-1"
        / "paper.pdf"
    ).resolve(strict=False)

    candidates = storage.canonical_local_asset_candidates(
        (
            r"C:\var\lib\stoody\uploads\clean\skb_sgtb-0001"
            r"\pdf_document\upload-1\paper.pdf"
        ),
        backend_root=backend_root,
        private_root=private_root,
    )

    assert expected in candidates
    assert all(
        private_root.resolve(strict=False) in candidate.parents
        or (backend_root / "uploads").resolve(strict=False) in candidate.parents
        for candidate in candidates
    )


def test_arbitrary_absolute_path_is_not_an_approved_candidate(tmp_path):
    assert (
        storage.canonical_local_asset_candidates(
            r"C:\Windows\System32\config\SAM",
            backend_root=tmp_path / "backend",
            private_root=tmp_path / "private",
        )
        == []
    )


@pytest.mark.asyncio
async def test_read_canonical_asset_uses_safe_legacy_rebase(tmp_path, monkeypatch):
    backend_root = Path(storage.__file__).resolve().parents[1]
    private_root = tmp_path / "private"
    paper = private_root / "clean" / "tenant-1" / "pdf_document" / "u1" / "paper.pdf"
    paper.parent.mkdir(parents=True)
    paper.write_bytes(b"%PDF-1.7\nimmutable")
    monkeypatch.setattr(storage.settings, "UPLOAD_PRIVATE_LOCAL_DIR", private_root)

    payload = await storage.read_canonical_asset(
        "/var/lib/stoody/uploads/clean/tenant-1/pdf_document/u1/paper.pdf"
    )

    assert payload == b"%PDF-1.7\nimmutable"
    assert backend_root.is_absolute()


@pytest.mark.asyncio
async def test_store_canonical_asset_promotes_to_private_s3(tmp_path, monkeypatch):
    payload = b"%PDF-1.7\npaper"
    digest = hashlib.sha256(payload).hexdigest()
    captured = {}

    async def fake_upload(data, *, object_key, content_type, metadata):
        captured.update(
            data=data,
            object_key=object_key,
            content_type=content_type,
            metadata=metadata,
        )
        return f"s3://test-bucket/{object_key}"

    monkeypatch.setattr(storage, "is_s3_enabled", lambda: True)
    monkeypatch.setattr(storage, "upload_private_object", fake_upload)

    transfer = await storage.store_canonical_asset(
        data=payload,
        local_path=str(tmp_path / "paper.pdf"),
        upload_id="upload-1",
        tenant_db="tenant-1",
        document_id="doc-1",
        artifact_kind="question-paper",
        filename="../../paper.pdf",
        content_type="application/pdf",
        sha256=digest,
    )

    assert transfer.promoted_to_s3 is True
    assert transfer.storage_path.startswith(
        "s3://test-bucket/private/exampen/canonical-assets/tenant-1/doc-1/"
    )
    assert ".." not in captured["object_key"]
    assert captured["metadata"]["sha256"] == digest


@pytest.mark.asyncio
async def test_store_fails_closed_when_object_storage_is_required(
    tmp_path,
    monkeypatch,
):
    payload = b"%PDF-1.7\npaper"
    monkeypatch.setattr(storage, "is_s3_enabled", lambda: False)
    monkeypatch.setenv("CANONICAL_ASSET_REQUIRE_OBJECT_STORAGE", "true")

    with pytest.raises(
        storage.CanonicalAssetStorageError,
        match="required",
    ):
        await storage.store_canonical_asset(
            data=payload,
            local_path=str(tmp_path / "paper.pdf"),
            upload_id="upload-1",
            tenant_db="tenant-1",
            document_id="doc-1",
            artifact_kind="question-paper",
            filename="paper.pdf",
            content_type="application/pdf",
            sha256=hashlib.sha256(payload).hexdigest(),
        )


@pytest.mark.asyncio
async def test_store_rejects_payload_that_does_not_match_scan_digest(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(storage, "is_s3_enabled", lambda: True)

    with pytest.raises(
        storage.CanonicalAssetStorageError,
        match="integrity",
    ):
        await storage.store_canonical_asset(
            data=b"%PDF-1.7\npaper",
            local_path=str(tmp_path / "paper.pdf"),
            upload_id="upload-1",
            tenant_db="tenant-1",
            document_id="doc-1",
            artifact_kind="question-paper",
            filename="paper.pdf",
            content_type="application/pdf",
            sha256="0" * 64,
        )
