import io
import zipfile

import pytest
from fastapi import HTTPException
from PIL import Image
from pypdf import PdfWriter

from core.upload_security.policies import get_upload_policy


class DummyUpload:
    def __init__(self, data: bytes, filename: str = "file.bin", content_type: str = "application/octet-stream"):
        self._buffer = io.BytesIO(data)
        self.filename = filename
        self.content_type = content_type

    async def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)


def make_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def make_png(size=(2, 2)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    return buffer.getvalue()


def make_zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_read_upload_file_limited_rejects_oversize():
    from core.upload_security.validation import read_upload_file_limited

    policy = get_upload_policy("school_logo").model_copy(update={"max_size_bytes": 3})

    with pytest.raises(HTTPException) as exc:
        await read_upload_file_limited(DummyUpload(b"abcd"), policy)

    assert exc.value.status_code == 413


def test_detect_file_type_accepts_valid_pdf():
    from core.upload_security.detection import detect_file_type

    detected = detect_file_type(make_pdf(), "paper.pdf", "application/pdf", get_upload_policy("pdf_document"))

    assert detected.extension == "pdf"
    assert detected.magic_type == "pdf"
    assert detected.declared_mime_type == "application/pdf"


def test_detect_file_type_rejects_mismatched_magic():
    from core.upload_security.detection import detect_file_type

    with pytest.raises(HTTPException) as exc:
        detect_file_type(make_png(), "paper.pdf", "application/pdf", get_upload_policy("pdf_document"))

    assert exc.value.status_code == 400
    assert "magic" in exc.value.detail.lower()


def test_post_scan_pdf_guard_counts_pages():
    from core.upload_security.detection import detect_file_type
    from core.upload_security.validation import run_post_scan_parser_guards

    data = make_pdf()
    policy = get_upload_policy("pdf_document").model_copy(update={"max_pdf_pages": 0})
    detected = detect_file_type(data, "paper.pdf", "application/pdf", get_upload_policy("pdf_document"))

    with pytest.raises(HTTPException) as exc:
        run_post_scan_parser_guards(data, detected, policy)

    assert exc.value.status_code == 400
    assert "page" in exc.value.detail.lower()


def test_post_scan_zip_guard_rejects_path_traversal():
    from core.upload_security.detection import detect_file_type
    from core.upload_security.validation import run_post_scan_parser_guards

    data = make_zip({"../evil.txt": b"x"})
    policy = get_upload_policy("debugger_document")
    detected = detect_file_type(
        data,
        "notes.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        policy,
    )

    with pytest.raises(HTTPException) as exc:
        run_post_scan_parser_guards(data, detected, policy)

    assert exc.value.status_code == 400
    assert "archive path" in exc.value.detail.lower()


def test_post_scan_zip_guard_rejects_uncompressed_overflow():
    from core.upload_security.detection import detect_file_type
    from core.upload_security.validation import run_post_scan_parser_guards

    data = make_zip({"large.txt": b"12345"})
    policy = get_upload_policy("debugger_document").model_copy(update={"max_archive_uncompressed_bytes": 4})
    detected = detect_file_type(
        data,
        "notes.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        get_upload_policy("debugger_document"),
    )

    with pytest.raises(HTTPException) as exc:
        run_post_scan_parser_guards(data, detected, policy)

    assert exc.value.status_code == 400
    assert "uncompressed" in exc.value.detail.lower()


def test_post_scan_image_guard_rejects_pixel_overflow():
    from core.upload_security.detection import detect_file_type
    from core.upload_security.validation import run_post_scan_parser_guards

    data = make_png(size=(2, 2))
    policy = get_upload_policy("school_logo").model_copy(update={"max_image_pixels": 3})
    detected = detect_file_type(data, "logo.png", "image/png", get_upload_policy("school_logo"))

    with pytest.raises(HTTPException) as exc:
        run_post_scan_parser_guards(data, detected, policy)

    assert exc.value.status_code == 400
    assert "pixel" in exc.value.detail.lower()
