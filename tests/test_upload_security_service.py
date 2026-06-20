import io
from dataclasses import dataclass

import pytest
from fastapi import HTTPException
from PIL import Image

from core.upload_security.scanner import ScanResult
from core.upload_security.service import secure_upload, secure_upload_many
from core.upload_security.storage import PrivateUploadStorage


class DummyUpload:
    def __init__(self, data: bytes, filename: str = "logo.png", content_type: str = "image/png"):
        self._buffer = io.BytesIO(data)
        self.filename = filename
        self.content_type = content_type

    async def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)


class FakeDb:
    def __init__(self):
        self.rows = []

    async def mongo_insert_one(self, collection, document):
        self.rows.append((collection, document))
        return {"inserted_id": document["upload_id"]}


@dataclass
class FakeScanner:
    result: ScanResult
    calls: int = 0

    async def scan_path(self, path, *, filename, policy_id):
        self.calls += 1
        return self.result


def make_png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_secure_upload_clean_flow_persists_verdict_and_releases(tmp_path):
    db = FakeDb()
    scanner = FakeScanner(ScanResult.clean(scanner_name="fake-av", scanner_version="1"))
    storage = PrivateUploadStorage(local_root=tmp_path)

    result = await secure_upload(
        file=DummyUpload(make_png()),
        policy_id="school_logo",
        actor={"user_id": "admin-1", "db_name": "skb_ciel"},
        db=db,
        purpose_metadata={"purpose": "school_logo", "admin_id": "admin-1"},
        authorization_subject="school:logo:admin-1",
        scanner=scanner,
        storage=storage,
    )

    assert result.upload_id
    assert result.released_storage_path.startswith(str(tmp_path))
    assert scanner.calls == 1
    assert db.rows[-1][0] == "upload_security_verdicts"
    verdict = db.rows[-1][1]
    assert verdict["status"] == "clean"
    assert verdict["policy_id"] == "school_logo"
    assert verdict["purpose_metadata"]["purpose"] == "school_logo"


@pytest.mark.asyncio
async def test_secure_upload_scanner_rejection_prevents_parser(monkeypatch, tmp_path):
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("parser guard should not run after malware rejection")

    monkeypatch.setattr("core.upload_security.service.run_post_scan_parser_guards", fail_if_called)
    db = FakeDb()
    scanner = FakeScanner(ScanResult.rejected("Eicar-Test-Signature", scanner_name="fake-av"))

    with pytest.raises(HTTPException) as exc:
        await secure_upload(
            file=DummyUpload(make_png()),
            policy_id="school_logo",
            actor={"user_id": "admin-1", "db_name": "skb_ciel"},
            db=db,
            purpose_metadata={"purpose": "school_logo"},
            authorization_subject="school:logo:admin-1",
            scanner=scanner,
            storage=PrivateUploadStorage(local_root=tmp_path),
        )

    assert exc.value.status_code == 400
    assert db.rows[-1][1]["status"] == "rejected"


@pytest.mark.asyncio
async def test_secure_upload_scanner_failure_is_503(tmp_path):
    db = FakeDb()
    scanner = FakeScanner(ScanResult.scan_failed("clamd unavailable", scanner_name="fake-av"))

    with pytest.raises(HTTPException) as exc:
        await secure_upload(
            file=DummyUpload(make_png()),
            policy_id="school_logo",
            actor={"user_id": "admin-1", "db_name": "skb_ciel"},
            db=db,
            purpose_metadata={"purpose": "school_logo"},
            authorization_subject="school:logo:admin-1",
            scanner=scanner,
            storage=PrivateUploadStorage(local_root=tmp_path),
        )

    assert exc.value.status_code == 503
    assert db.rows[-1][1]["status"] == "scan_failed"


@pytest.mark.asyncio
async def test_secure_upload_requires_authorization_subject(tmp_path):
    with pytest.raises(HTTPException) as exc:
        await secure_upload(
            file=DummyUpload(make_png()),
            policy_id="school_logo",
            actor={"user_id": "admin-1", "db_name": "skb_ciel"},
            db=FakeDb(),
            purpose_metadata={"purpose": "school_logo"},
            authorization_subject="",
            scanner=FakeScanner(ScanResult.clean()),
            storage=PrivateUploadStorage(local_root=tmp_path),
        )

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_secure_upload_many_enforces_total_size_and_metadata_factory(tmp_path):
    data = make_png()
    files = [DummyUpload(data, filename=f"{index}.png") for index in range(2)]
    scanner = FakeScanner(ScanResult.clean())
    db = FakeDb()

    results = await secure_upload_many(
        files=files,
        policy_id="desktop_bug_image",
        actor={"user_id": "admin-1", "db_name": "skb_ciel"},
        db=db,
        purpose_metadata_factory=lambda file, index: {"purpose": "bug_image", "index": index},
        authorization_subject_factory=lambda file, index: f"bug:image:{index}",
        scanner=scanner,
        storage=PrivateUploadStorage(local_root=tmp_path),
    )

    assert len(results) == 2
    assert db.rows[0][1]["purpose_metadata"]["index"] == 0
    assert db.rows[1][1]["purpose_metadata"]["index"] == 1


@pytest.mark.asyncio
async def test_secure_upload_many_rejects_file_count(tmp_path):
    data = make_png()
    files = [DummyUpload(data, filename=f"{index}.png") for index in range(9)]

    with pytest.raises(HTTPException) as exc:
        await secure_upload_many(
            files=files,
            policy_id="desktop_bug_image",
            actor={"user_id": "admin-1", "db_name": "skb_ciel"},
            db=FakeDb(),
            purpose_metadata_factory=lambda file, index: {"purpose": "bug_image", "index": index},
            authorization_subject_factory=lambda file, index: f"bug:image:{index}",
            scanner=FakeScanner(ScanResult.clean()),
            storage=PrivateUploadStorage(local_root=tmp_path),
        )

    assert exc.value.status_code == 400
