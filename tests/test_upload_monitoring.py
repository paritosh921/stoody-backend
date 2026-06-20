import io
from dataclasses import dataclass

import pytest
from fastapi import HTTPException
from PIL import Image

from core.upload_security.cleanup import collect_upload_storage_usage
from core.upload_security.scanner import ScanResult
from core.upload_security.service import secure_upload
from core.upload_security.storage import PrivateUploadStorage


class DummyUpload:
    def __init__(self, data: bytes, filename: str, content_type: str):
        self._buffer = io.BytesIO(data)
        self.filename = filename
        self.content_type = content_type

    async def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)


class FakeDb:
    async def mongo_insert_one(self, collection, document):
        return {"inserted_id": document["upload_id"]}


@dataclass
class FakeScanner:
    result: ScanResult
    calls: int = 0

    async def scan_path(self, path, *, filename, policy_id):
        self.calls += 1
        return self.result


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_magic_mismatch_records_reason_before_scan(monkeypatch, tmp_path):
    rejections = []

    def fake_rejection(policy_id, reason):
        rejections.append((policy_id, reason))

    monkeypatch.setattr("core.upload_security.service.record_upload_security_rejection", fake_rejection)
    scanner = FakeScanner(ScanResult.clean())

    with pytest.raises(HTTPException):
        await secure_upload(
            file=DummyUpload(_png_bytes(), "paper.pdf", "application/pdf"),
            policy_id="pdf_document",
            actor={"user_id": "admin-1", "db_name": "skb_ciel"},
            db=FakeDb(),
            purpose_metadata={"purpose": "pdf_document"},
            authorization_subject="pdf:document:test",
            scanner=scanner,
            storage=PrivateUploadStorage(local_root=tmp_path),
        )

    assert rejections == [("pdf_document", "magic_mismatch")]
    assert scanner.calls == 0


@pytest.mark.asyncio
async def test_scanner_timeout_records_latency_and_alert(monkeypatch, tmp_path):
    alerts = []
    latencies = []

    def fake_alert(alert_type, active=True):
        alerts.append((alert_type, active))

    def fake_latency(policy_id, status, duration_seconds):
        latencies.append((policy_id, status, duration_seconds))

    monkeypatch.setattr("core.upload_security.service.set_upload_security_alert", fake_alert)
    monkeypatch.setattr("core.upload_security.service.observe_upload_scan_latency", fake_latency)

    with pytest.raises(HTTPException):
        await secure_upload(
            file=DummyUpload(_png_bytes(), "logo.png", "image/png"),
            policy_id="school_logo",
            actor={"user_id": "admin-1", "db_name": "skb_ciel"},
            db=FakeDb(),
            purpose_metadata={"purpose": "school_logo"},
            authorization_subject="school:logo:test",
            scanner=FakeScanner(ScanResult.scan_failed("clamdscan timed out after 30 seconds")),
            storage=PrivateUploadStorage(local_root=tmp_path),
        )

    assert alerts == [("scanner_timeout", True)]
    assert latencies
    assert latencies[0][0:2] == ("school_logo", "scan_failed")


def test_collect_upload_storage_usage_counts_private_prefixes(tmp_path):
    (tmp_path / "quarantine" / "tenant").mkdir(parents=True)
    (tmp_path / "rejected" / "tenant").mkdir(parents=True)
    (tmp_path / "quarantine" / "tenant" / "a.bin").write_bytes(b"abc")
    (tmp_path / "rejected" / "tenant" / "b.bin").write_bytes(b"de")

    usage = collect_upload_storage_usage(tmp_path)

    assert usage["quarantine"] == 3
    assert usage["rejected"] == 2
