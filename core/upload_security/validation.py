"""Upload byte limits and post-scan parser guards."""

from __future__ import annotations

import csv
import io
import posixpath
import warnings
import zipfile
from pathlib import PurePosixPath
from typing import Any

from fastapi import HTTPException, status

from .detection import DetectedFileType
from .policies import UploadPolicy


READ_CHUNK_SIZE = 1024 * 1024


def _reject(status_code: int, detail: str) -> None:
    raise HTTPException(status_code=status_code, detail=detail)


async def read_upload_file_limited(file: Any, policy: UploadPolicy) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(READ_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > policy.max_size_bytes:
            _reject(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"Upload exceeds {policy.max_size_bytes} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def run_post_scan_parser_guards(data: bytes, detected: DetectedFileType, policy: UploadPolicy) -> None:
    if detected.magic_type == "pdf" and policy.max_pdf_pages is not None:
        _guard_pdf(data, policy)
    if detected.magic_type in {"png", "jpeg", "gif", "bmp", "webp"} and policy.max_image_pixels is not None:
        _guard_image(data, policy)
    if detected.magic_type == "zip":
        _guard_zip(data, policy)
    if detected.magic_type == "csv" and (policy.max_rows is not None or policy.max_columns is not None):
        _guard_csv(data, policy)


def _guard_pdf(data: bytes, policy: UploadPolicy) -> None:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            _reject(status.HTTP_400_BAD_REQUEST, "Encrypted PDFs are not allowed")
        page_count = len(reader.pages)
    except HTTPException:
        raise
    except Exception as exc:
        _reject(status.HTTP_400_BAD_REQUEST, f"Unreadable PDF: {exc}")

    if policy.max_pdf_pages is not None and page_count > policy.max_pdf_pages:
        _reject(status.HTTP_400_BAD_REQUEST, f"PDF page count exceeds {policy.max_pdf_pages}")


def _guard_image(data: bytes, policy: UploadPolicy) -> None:
    try:
        from PIL import Image

        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(data))
            width, height = image.size
            pixels = width * height
            if policy.max_image_pixels is not None and pixels > policy.max_image_pixels:
                _reject(status.HTTP_400_BAD_REQUEST, f"Image pixel count exceeds {policy.max_image_pixels}")
            image.verify()
    except HTTPException:
        raise
    except Exception as exc:
        _reject(status.HTTP_400_BAD_REQUEST, f"Unreadable image: {exc}")


def _is_unsafe_archive_name(name: str) -> bool:
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("../"):
        return True
    if posixpath.isabs(normalized):
        return True
    path = PurePosixPath(normalized)
    return any(part == ".." for part in path.parts)


def _guard_zip(data: bytes, policy: UploadPolicy) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
    except zipfile.BadZipFile as exc:
        _reject(status.HTTP_400_BAD_REQUEST, f"Unreadable ZIP/Office container: {exc}")

    if policy.max_archive_entries is not None and len(infos) > policy.max_archive_entries:
        _reject(status.HTTP_400_BAD_REQUEST, f"Archive entry count exceeds {policy.max_archive_entries}")

    total_uncompressed = 0
    for info in infos:
        name = info.filename
        if _is_unsafe_archive_name(name):
            _reject(status.HTTP_400_BAD_REQUEST, "Archive path traversal is not allowed")
        if policy.max_archive_depth is not None:
            depth = len([part for part in PurePosixPath(name.replace("\\", "/")).parts if part and part != "."])
            if depth > policy.max_archive_depth:
                _reject(status.HTTP_400_BAD_REQUEST, f"Archive depth exceeds {policy.max_archive_depth}")
        total_uncompressed += info.file_size
        if (
            policy.max_archive_uncompressed_bytes is not None
            and total_uncompressed > policy.max_archive_uncompressed_bytes
        ):
            _reject(
                status.HTTP_400_BAD_REQUEST,
                f"Archive uncompressed size exceeds {policy.max_archive_uncompressed_bytes}",
            )


def _guard_csv(data: bytes, policy: UploadPolicy) -> None:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        _reject(status.HTTP_400_BAD_REQUEST, f"Unreadable CSV: {exc}")

    reader = csv.reader(io.StringIO(text))
    rows = 0
    for row in reader:
        rows += 1
        if policy.max_rows is not None and rows > policy.max_rows:
            _reject(status.HTTP_400_BAD_REQUEST, f"CSV row count exceeds {policy.max_rows}")
        if policy.max_columns is not None and len(row) > policy.max_columns:
            _reject(status.HTTP_400_BAD_REQUEST, f"CSV column count exceeds {policy.max_columns}")
