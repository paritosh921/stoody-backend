import pathlib

import pytest

from core.upload_security.storage import PrivateUploadStorage


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
