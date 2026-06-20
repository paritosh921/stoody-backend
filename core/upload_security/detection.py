"""Cheap file type detection before malware scanning."""

from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass

from fastapi import HTTPException, status

from .policies import UploadPolicy


@dataclass(frozen=True)
class DetectedFileType:
    extension: str
    declared_mime_type: str
    magic_type: str
    original_filename: str


_EXTENSION_MAGIC_ALIASES = {
    "jpg": "jpeg",
    "jpeg": "jpeg",
    "docx": "zip",
    "xlsx": "zip",
    "pptx": "zip",
    "doc": "ole",
    "xls": "ole",
    "ppt": "ole",
    "csv": "csv",
    "json": "json",
    "webm": "ebml",
    "mkv": "ebml",
}


def _normalized_extension(filename: str | None) -> str:
    if not filename:
        return ""
    return os.path.splitext(filename)[1].lower().lstrip(".")


def _normalized_mime(content_type: str | None, filename: str | None) -> str:
    if content_type:
        return content_type.split(";", 1)[0].strip().lower()
    guessed, _ = mimetypes.guess_type(filename or "")
    return guessed or ""


def detect_magic_type(data: bytes) -> str:
    prefix = data[:64]
    if prefix.startswith(b"%PDF-"):
        return "pdf"
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if prefix.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if prefix.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if prefix.startswith(b"BM"):
        return "bmp"
    if prefix.startswith(b"RIFF") and len(prefix) >= 12 and prefix[8:12] == b"WEBP":
        return "webp"
    if prefix.startswith(b"PK\x03\x04") or prefix.startswith(b"PK\x05\x06") or prefix.startswith(b"PK\x07\x08"):
        return "zip"
    if prefix.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return "ole"
    if len(prefix) >= 12 and prefix[4:8] == b"ftyp":
        major_brand = prefix[8:12].lower()
        if major_brand in {b"qt  "}:
            return "mov"
        return "mp4"
    if prefix.startswith(b"\x1a\x45\xdf\xa3"):
        return "ebml"

    stripped = data[:4096].lstrip()
    if stripped.startswith((b"{", b"[")):
        return "json"

    try:
        sample = data[:4096].decode("utf-8")
    except UnicodeDecodeError:
        return "unknown"
    if "\x00" in sample:
        return "unknown"
    if "," in sample or "\n" in sample:
        return "csv"
    return "text"


def _reject(detail: str) -> None:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def detect_file_type(data: bytes, filename: str | None, content_type: str | None, policy: UploadPolicy) -> DetectedFileType:
    extension = _normalized_extension(filename)
    declared_mime_type = _normalized_mime(content_type, filename)
    magic_type = detect_magic_type(data)

    if policy.allowed_extensions and extension not in policy.allowed_extensions:
        _reject(f"File extension not allowed for policy {policy.policy_id}")

    if policy.allowed_mime_types and declared_mime_type:
        if declared_mime_type == "application/octet-stream" and policy.allow_octet_stream:
            pass
        elif declared_mime_type not in policy.allowed_mime_types:
            _reject(f"Declared MIME type not allowed for policy {policy.policy_id}")

    if policy.allowed_magic_types and magic_type not in policy.allowed_magic_types:
        _reject(f"File magic type not allowed for policy {policy.policy_id}")

    expected_magic = _EXTENSION_MAGIC_ALIASES.get(extension, extension)
    if expected_magic and magic_type != "unknown" and expected_magic not in {magic_type, "txt", "text"}:
        _reject("File extension does not match detected magic type")

    return DetectedFileType(
        extension=extension,
        declared_mime_type=declared_mime_type,
        magic_type=magic_type,
        original_filename=filename or "",
    )
