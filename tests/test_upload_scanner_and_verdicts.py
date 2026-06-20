from datetime import datetime

import pytest

from core.upload_security.scanner import ClamAVScanner, ScanResult
from core.upload_security.verdicts import build_upload_verdict, persist_upload_verdict


class FakeDb:
    def __init__(self):
        self.rows = []

    async def mongo_insert_one(self, collection, document):
        self.rows.append((collection, document))
        return {"inserted_id": document["upload_id"]}


class FakeCollection:
    def __init__(self):
        self.rows = []
        self.indexes = []

    async def create_index(self, field):
        self.indexes.append(field)

    async def insert_one(self, document):
        self.rows.append(document)
        return {"inserted_id": document["upload_id"]}


class FakeMotorDb:
    def __init__(self):
        self.collection = FakeCollection()

    def __getitem__(self, name):
        assert name == "upload_security_verdicts"
        return self.collection


def test_scan_result_factories():
    assert ScanResult.clean(scanner_name="test").status == "clean"
    assert ScanResult.rejected("bad").status == "rejected"
    assert ScanResult.scan_failed("down").status == "scan_failed"


def test_clamdscan_command_uses_configured_socket(monkeypatch):
    monkeypatch.setattr("core.upload_security.scanner.settings.CLAMAV_SOCKET", "/var/run/clamav/clamd.ctl")

    scanner = ClamAVScanner()
    command, cleanup = scanner._build_clamdscan_command("/tmp/upload.pdf")

    try:
        assert command[:3] == ["clamdscan", "--fdpass", "--no-summary"]
        assert any(part.startswith("--config-file=") for part in command)
        assert command[-1] == "/tmp/upload.pdf"
    finally:
        cleanup()


@pytest.mark.asyncio
async def test_verdict_persistence_shape():
    db = FakeDb()
    verdict = build_upload_verdict(
        upload_id="upload-1",
        policy_id="school_logo",
        status="clean",
        sha256="a" * 64,
        size_bytes=3,
        original_filename="logo.png",
        declared_content_type="image/png",
        detected_magic_type="png",
        scanner_name="fake-av",
        scanner_version="1",
        scan_started_at=datetime.utcnow(),
        scan_finished_at=datetime.utcnow(),
        tenant_db="skb_ciel",
        user_id="admin-1",
        purpose_metadata={"purpose": "school_logo"},
        authorization_subject="school:logo:admin-1",
        quarantine_storage_path="/private/quarantine/file",
        released_storage_path="/private/released/file",
    )

    await persist_upload_verdict(db, verdict)

    assert db.rows[0][0] == "upload_security_verdicts"
    assert db.rows[0][1]["upload_id"] == "upload-1"
    assert db.rows[0][1]["status"] == "clean"


@pytest.mark.asyncio
async def test_verdict_persistence_creates_motor_indexes():
    db = FakeMotorDb()
    verdict = build_upload_verdict(
        upload_id="upload-2",
        policy_id="school_logo",
        status="clean",
        sha256="b" * 64,
        size_bytes=3,
        original_filename="logo.png",
        declared_content_type="image/png",
        detected_magic_type="png",
        scanner_name="fake-av",
        scanner_version="1",
        scan_started_at=datetime.utcnow(),
        scan_finished_at=datetime.utcnow(),
        tenant_db="skb_ciel",
        user_id="admin-1",
        purpose_metadata={"purpose": "school_logo"},
        authorization_subject="school:logo:admin-1",
        quarantine_storage_path="/private/quarantine/file",
        released_storage_path="/private/clean/file",
    )

    await persist_upload_verdict(db, verdict)

    assert db.collection.rows[0]["upload_id"] == "upload-2"
    assert set(db.collection.indexes) == {"upload_id", "sha256", "tenant_db", "status", "created_at"}
