from datetime import datetime, timedelta, timezone

import pytest

from core.upload_security.scanner import ClamAVScanner, ScanResult, clamav_signature_age_seconds
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

    def __call__(self, *args, **kwargs):
        # A real AsyncIOMotorCollection is callable too, despite not being a
        # DatabaseManager helper.  It raises when invoked; the persistence
        # helper must recognize it as a collection before reaching this path.
        raise TypeError("MotorCollection object is not callable")


class FakeMotorDb:
    def __init__(self):
        self.collection = FakeCollection()

    def __getitem__(self, name):
        assert name == "upload_security_verdicts"
        return self.collection

    def __getattr__(self, name):
        # Motor databases dynamically expose a collection for unknown
        # attributes, including a name that looks like a manager helper.
        if name == "mongo_insert_one":
            return self.collection
        raise AttributeError(name)


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


def test_clamav_signature_age_uses_newest_signature_file(tmp_path):
    now = datetime(2026, 6, 20, tzinfo=timezone.utc)
    old = tmp_path / "main.cvd"
    fresh = tmp_path / "daily.cld"
    old.write_bytes(b"old")
    fresh.write_bytes(b"fresh")
    old_ts = (now - timedelta(days=3)).timestamp()
    fresh_ts = (now - timedelta(hours=6)).timestamp()
    import os

    os.utime(old, (old_ts, old_ts))
    os.utime(fresh, (fresh_ts, fresh_ts))

    assert clamav_signature_age_seconds(tmp_path, now=now) == 6 * 60 * 60


@pytest.mark.asyncio
async def test_clamdscan_timeout_fails_closed(monkeypatch, tmp_path):
    class SlowProcess:
        returncode = 0
        killed = False

        async def communicate(self):
            import asyncio

            await asyncio.sleep(0.05)
            return b"", b""

        def kill(self):
            self.killed = True

        async def wait(self):
            return None

    process = SlowProcess()

    async def fake_create_subprocess_exec(*args, **kwargs):
        return process

    monkeypatch.setattr("core.upload_security.scanner.settings.UPLOAD_AV_ENABLED", True)
    monkeypatch.setattr("core.upload_security.scanner.settings.UPLOAD_SCANNER_TIMEOUT_SECONDS", 0.001, raising=False)
    async def fake_version():
        return "ClamAV test"

    monkeypatch.setattr("core.upload_security.scanner._get_clamdscan_version", fake_version)
    monkeypatch.setattr("core.upload_security.scanner.asyncio.create_subprocess_exec", fake_create_subprocess_exec)

    upload = tmp_path / "sample.pdf"
    upload.write_bytes(b"%PDF-1.7\n")

    result = await ClamAVScanner().scan_path(upload, filename="sample.pdf", policy_id="pdf_document")

    assert result.status == "scan_failed"
    assert "timed out" in (result.error or "").lower()
    assert process.killed is True


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
