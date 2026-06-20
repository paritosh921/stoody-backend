import pathlib
import json

import pytest

from core.upload_security.storage import PRIVATE_DIRECTORY_MODE, PrivateUploadStorage


@pytest.mark.asyncio
async def test_private_storage_writes_quarantine_and_clean_paths(tmp_path):
    storage = PrivateUploadStorage(local_root=tmp_path)

    quarantine = await storage.write_quarantine(
        data=b"abc",
        tenant="skb_ciel",
        upload_id="upload-1",
        original_filename="bad name.pdf",
    )
    released = await storage.release_clean(
        quarantine_path=quarantine,
        tenant="skb_ciel",
        policy_id="pdf_document",
        upload_id="upload-1",
        safe_filename="paper.pdf",
        content_type="application/pdf",
        metadata={"upload_id": "upload-1"},
    )

    assert pathlib.Path(quarantine).read_bytes() == b"abc"
    assert pathlib.Path(released).read_bytes() == b"abc"
    assert "quarantine" in quarantine
    assert "clean" in released
    assert "uploads" not in pathlib.Path(released).parts
    sidecar = pathlib.Path(f"{released}.metadata.json")
    assert sidecar.exists()
    assert json.loads(sidecar.read_text(encoding="utf-8"))["metadata"] == {"upload_id": "upload-1"}


@pytest.mark.asyncio
async def test_private_storage_sets_restrictive_directory_mode(monkeypatch, tmp_path):
    chmod_calls = []

    def fake_chmod(path, mode):
        chmod_calls.append((pathlib.Path(path), mode))

    monkeypatch.setattr("core.upload_security.storage.os.chmod", fake_chmod)
    storage = PrivateUploadStorage(local_root=tmp_path)

    quarantine = await storage.write_quarantine(
        data=b"abc",
        tenant="skb_ciel",
        upload_id="upload-1",
        original_filename="paper.pdf",
    )

    assert pathlib.Path(quarantine).exists()
    assert chmod_calls
    assert (pathlib.Path(quarantine).parent, PRIVATE_DIRECTORY_MODE) in chmod_calls
    assert (tmp_path / "quarantine", PRIVATE_DIRECTORY_MODE) in chmod_calls
    assert (tmp_path / "quarantine" / "skb_ciel", PRIVATE_DIRECTORY_MODE) in chmod_calls


@pytest.mark.asyncio
async def test_private_storage_writes_released_metadata_sidecar(tmp_path):
    storage = PrivateUploadStorage(local_root=tmp_path, released_prefix="derived")

    released = await storage.write_released_bytes(
        data=b"image",
        tenant="skb_ciel",
        policy_id="exam_template_file",
        upload_id="upload-1",
        safe_filename="template.png",
        content_type="image/png",
        metadata={"source_upload_id": "upload-1", "derived_kind": "dcr_exam_template"},
    )

    sidecar = pathlib.Path(f"{released}.metadata.json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metadata["content_type"] == "image/png"
    assert metadata["metadata"]["source_upload_id"] == "upload-1"
