import io
import zipfile
from dataclasses import dataclass

import pytest
from fastapi import HTTPException
from PIL import Image
from pypdf import PdfWriter

from core.upload_security.policies import DEFAULT_UPLOAD_POLICIES, get_upload_policy
from core.upload_security.routes import UPLOAD_ROUTE_POLICY_MAP
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


def _pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("xl/workbook.xml", b"<workbook/>")
    return buffer.getvalue()


def _csv_bytes() -> bytes:
    return b"name,email\nA,a@example.test\n"


def _sample_for_policy(policy_id: str) -> tuple[bytes, str, str]:
    policy = get_upload_policy(policy_id)
    extensions = set(policy.allowed_extensions)
    magic = set(policy.allowed_magic_types)
    if "pdf" in extensions and "pdf" in magic:
        return _pdf_bytes(), "sample.pdf", "application/pdf"
    if "png" in extensions and "png" in magic:
        return _png_bytes(), "sample.png", "image/png"
    if "csv" in extensions and "csv" in magic:
        return _csv_bytes(), "sample.csv", "text/csv"
    if "zip" in extensions and "zip" in magic:
        return _zip_bytes(), "sample.zip", "application/zip"
    if "xlsx" in extensions and "zip" in magic:
        return _zip_bytes(), "sample.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    raise AssertionError(f"No test sample for policy {policy_id}")


def _fake_extension_sample(policy_id: str) -> tuple[bytes, str, str]:
    policy = get_upload_policy(policy_id)
    extensions = set(policy.allowed_extensions)
    if "pdf" in extensions:
        return _png_bytes(), "renamed.pdf", "application/pdf"
    if "png" in extensions:
        return _pdf_bytes(), "renamed.png", "image/png"
    if "zip" in extensions:
        return _pdf_bytes(), "renamed.zip", "application/zip"
    if "xlsx" in extensions:
        return _pdf_bytes(), "renamed.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if "csv" in extensions:
        return _pdf_bytes(), "renamed.csv", "text/csv"
    first_ext = sorted(extensions)[0]
    return _pdf_bytes(), f"renamed.{first_ext}", "application/octet-stream"


BINARY_POLICIES = tuple(
    policy_id
    for policy_id, policy in sorted(DEFAULT_UPLOAD_POLICIES.items())
    if policy.policy_kind == "binary"
)


@dataclass(frozen=True)
class RoutePolicyCase:
    route_id: str
    policy_id: str


def _route_policy_cases() -> tuple[RoutePolicyCase, ...]:
    cases: list[RoutePolicyCase] = []
    for route in UPLOAD_ROUTE_POLICY_MAP:
        policy_ids = [route.policy_id, *route.field_policies.values()]
        for policy_id in dict.fromkeys(policy_ids):
            policy = get_upload_policy(policy_id)
            if policy.policy_kind != "binary":
                continue
            field_suffix = "" if policy_id == route.policy_id else f":{policy_id}"
            cases.append(RoutePolicyCase(f"{route.method} {route.path_template}{field_suffix}", policy_id))
    return tuple(cases)


BINARY_ROUTE_POLICY_CASES = _route_policy_cases()


@pytest.mark.asyncio
@pytest.mark.parametrize("policy_id", BINARY_POLICIES)
async def test_binary_upload_policy_accepts_clean_upload(policy_id, tmp_path):
    data, filename, content_type = _sample_for_policy(policy_id)
    scanner = FakeScanner(ScanResult.clean(scanner_name="fake-av", scanner_version="1"))

    result = await secure_upload(
        file=DummyUpload(data, filename, content_type),
        policy_id=policy_id,
        actor={"user_id": "admin-1", "db_name": "skb_ciel"},
        db=FakeDb(),
        purpose_metadata={"purpose": policy_id},
        authorization_subject=f"upload:{policy_id}:clean",
        scanner=scanner,
        storage=PrivateUploadStorage(local_root=tmp_path),
        include_bytes=False,
    )

    assert result.detected_magic_type in get_upload_policy(policy_id).allowed_magic_types
    assert scanner.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("policy_id", BINARY_POLICIES)
async def test_binary_upload_policy_rejects_renamed_fake_extension_before_scan(policy_id, tmp_path):
    data, filename, content_type = _fake_extension_sample(policy_id)
    scanner = FakeScanner(ScanResult.clean(scanner_name="fake-av", scanner_version="1"))

    with pytest.raises(HTTPException) as exc:
        await secure_upload(
            file=DummyUpload(data, filename, content_type),
            policy_id=policy_id,
            actor={"user_id": "admin-1", "db_name": "skb_ciel"},
            db=FakeDb(),
            purpose_metadata={"purpose": policy_id},
            authorization_subject=f"upload:{policy_id}:fake-extension",
            scanner=scanner,
            storage=PrivateUploadStorage(local_root=tmp_path),
        )

    assert exc.value.status_code == 400
    assert scanner.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("policy_id", BINARY_POLICIES)
async def test_binary_upload_policy_rejects_mime_mismatch_before_scan(policy_id, tmp_path):
    data, filename, _ = _sample_for_policy(policy_id)
    scanner = FakeScanner(ScanResult.clean(scanner_name="fake-av", scanner_version="1"))

    with pytest.raises(HTTPException) as exc:
        await secure_upload(
            file=DummyUpload(data, filename, "application/x-msdownload"),
            policy_id=policy_id,
            actor={"user_id": "admin-1", "db_name": "skb_ciel"},
            db=FakeDb(),
            purpose_metadata={"purpose": policy_id},
            authorization_subject=f"upload:{policy_id}:mime-mismatch",
            scanner=scanner,
            storage=PrivateUploadStorage(local_root=tmp_path),
        )

    assert exc.value.status_code == 400
    assert scanner.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("policy_id", BINARY_POLICIES)
async def test_binary_upload_policy_rejects_oversize_before_scan(monkeypatch, policy_id, tmp_path):
    policy = get_upload_policy(policy_id).model_copy(update={"max_size_bytes": 3})
    monkeypatch.setattr("core.upload_security.service.get_upload_policy", lambda requested: policy)
    _, filename, content_type = _sample_for_policy(policy_id)
    scanner = FakeScanner(ScanResult.clean(scanner_name="fake-av", scanner_version="1"))

    with pytest.raises(HTTPException) as exc:
        await secure_upload(
            file=DummyUpload(b"abcd", filename, content_type),
            policy_id=policy_id,
            actor={"user_id": "admin-1", "db_name": "skb_ciel"},
            db=FakeDb(),
            purpose_metadata={"purpose": policy_id},
            authorization_subject=f"upload:{policy_id}:oversize",
            scanner=scanner,
            storage=PrivateUploadStorage(local_root=tmp_path),
        )

    assert exc.value.status_code == 413
    assert scanner.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("scan_result,status_code", [
    (ScanResult.rejected("Eicar-Test-Signature", scanner_name="fake-av"), 400),
    (ScanResult.scan_failed("clamd unavailable", scanner_name="fake-av"), 503),
])
async def test_pdf_upload_policy_fails_closed_before_parser_on_scanner_failure(
    monkeypatch,
    scan_result,
    status_code,
    tmp_path,
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("parser guard should not run unless malware scan is clean")

    monkeypatch.setattr("core.upload_security.service.run_post_scan_parser_guards", fail_if_called)
    data, filename, content_type = _sample_for_policy("pdf_document")
    scanner = FakeScanner(scan_result)

    with pytest.raises(HTTPException) as exc:
        await secure_upload(
            file=DummyUpload(data, filename, content_type),
            policy_id="pdf_document",
            actor={"user_id": "admin-1", "db_name": "skb_ciel"},
            db=FakeDb(),
            purpose_metadata={"purpose": "pdf_document"},
            authorization_subject="upload:pdf_document:scanner-failure",
            scanner=scanner,
            storage=PrivateUploadStorage(local_root=tmp_path),
        )

    assert exc.value.status_code == status_code
    assert scanner.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    BINARY_ROUTE_POLICY_CASES,
    ids=lambda case: case.route_id,
)
async def test_each_binary_upload_route_accepts_clean_upload(case, tmp_path):
    data, filename, content_type = _sample_for_policy(case.policy_id)
    scanner = FakeScanner(ScanResult.clean(scanner_name="fake-av", scanner_version="1"))

    result = await secure_upload(
        file=DummyUpload(data, filename, content_type),
        policy_id=case.policy_id,
        actor={"user_id": "admin-1", "db_name": "skb_ciel"},
        db=FakeDb(),
        purpose_metadata={"purpose": case.route_id, "route_policy": case.policy_id},
        authorization_subject=f"route:{case.route_id}:clean",
        scanner=scanner,
        storage=PrivateUploadStorage(local_root=tmp_path),
        include_bytes=False,
    )

    assert result.detected_magic_type in get_upload_policy(case.policy_id).allowed_magic_types
    assert scanner.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    BINARY_ROUTE_POLICY_CASES,
    ids=lambda case: case.route_id,
)
async def test_each_binary_upload_route_rejects_renamed_fake_extension_before_scan(case, tmp_path):
    data, filename, content_type = _fake_extension_sample(case.policy_id)
    scanner = FakeScanner(ScanResult.clean(scanner_name="fake-av", scanner_version="1"))

    with pytest.raises(HTTPException) as exc:
        await secure_upload(
            file=DummyUpload(data, filename, content_type),
            policy_id=case.policy_id,
            actor={"user_id": "admin-1", "db_name": "skb_ciel"},
            db=FakeDb(),
            purpose_metadata={"purpose": case.route_id, "route_policy": case.policy_id},
            authorization_subject=f"route:{case.route_id}:fake-extension",
            scanner=scanner,
            storage=PrivateUploadStorage(local_root=tmp_path),
        )

    assert exc.value.status_code == 400
    assert scanner.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    BINARY_ROUTE_POLICY_CASES,
    ids=lambda case: case.route_id,
)
async def test_each_binary_upload_route_rejects_mime_mismatch_before_scan(case, tmp_path):
    data, filename, _ = _sample_for_policy(case.policy_id)
    scanner = FakeScanner(ScanResult.clean(scanner_name="fake-av", scanner_version="1"))

    with pytest.raises(HTTPException) as exc:
        await secure_upload(
            file=DummyUpload(data, filename, "application/x-msdownload"),
            policy_id=case.policy_id,
            actor={"user_id": "admin-1", "db_name": "skb_ciel"},
            db=FakeDb(),
            purpose_metadata={"purpose": case.route_id, "route_policy": case.policy_id},
            authorization_subject=f"route:{case.route_id}:mime-mismatch",
            scanner=scanner,
            storage=PrivateUploadStorage(local_root=tmp_path),
        )

    assert exc.value.status_code == 400
    assert scanner.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    BINARY_ROUTE_POLICY_CASES,
    ids=lambda case: case.route_id,
)
async def test_each_binary_upload_route_rejects_oversize_before_scan(monkeypatch, case, tmp_path):
    policy = get_upload_policy(case.policy_id).model_copy(update={"max_size_bytes": 3})
    monkeypatch.setattr("core.upload_security.service.get_upload_policy", lambda requested: policy)
    _, filename, content_type = _sample_for_policy(case.policy_id)
    scanner = FakeScanner(ScanResult.clean(scanner_name="fake-av", scanner_version="1"))

    with pytest.raises(HTTPException) as exc:
        await secure_upload(
            file=DummyUpload(b"abcd", filename, content_type),
            policy_id=case.policy_id,
            actor={"user_id": "admin-1", "db_name": "skb_ciel"},
            db=FakeDb(),
            purpose_metadata={"purpose": case.route_id, "route_policy": case.policy_id},
            authorization_subject=f"route:{case.route_id}:oversize",
            scanner=scanner,
            storage=PrivateUploadStorage(local_root=tmp_path),
        )

    assert exc.value.status_code == 413
    assert scanner.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,scan_result,status_code",
    [
        (case, ScanResult.rejected("Eicar-Test-Signature", scanner_name="fake-av"), 400)
        for case in BINARY_ROUTE_POLICY_CASES
    ]
    + [
        (case, ScanResult.scan_failed("clamd timeout", scanner_name="fake-av"), 503)
        for case in BINARY_ROUTE_POLICY_CASES
    ],
    ids=lambda value: value.route_id if isinstance(value, RoutePolicyCase) else str(value),
)
async def test_each_binary_upload_route_fails_closed_before_parser_on_scanner_failure(
    monkeypatch,
    case,
    scan_result,
    status_code,
    tmp_path,
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("parser guard should not run unless malware scan is clean")

    monkeypatch.setattr("core.upload_security.service.run_post_scan_parser_guards", fail_if_called)
    data, filename, content_type = _sample_for_policy(case.policy_id)
    scanner = FakeScanner(scan_result)

    with pytest.raises(HTTPException) as exc:
        await secure_upload(
            file=DummyUpload(data, filename, content_type),
            policy_id=case.policy_id,
            actor={"user_id": "admin-1", "db_name": "skb_ciel"},
            db=FakeDb(),
            purpose_metadata={"purpose": case.route_id, "route_policy": case.policy_id},
            authorization_subject=f"route:{case.route_id}:scanner-failure",
            scanner=scanner,
            storage=PrivateUploadStorage(local_root=tmp_path),
        )

    assert exc.value.status_code == status_code
    assert scanner.calls == 1
