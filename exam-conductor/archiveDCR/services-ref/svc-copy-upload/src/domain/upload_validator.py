"""Pure validation logic for copy image uploads.

ZERO I/O — this module must never import asyncio, aiohttp, sqlalchemy,
nats, boto3, or any I/O library.
"""

from __future__ import annotations

from dataclasses import dataclass

# JPEG: FF D8 FF, PNG: 89 50 4E 47 0D 0A 1A 0A
_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of an upload validation check."""

    valid: bool
    error: str | None = None


def validate_upload(
    exam_id: str | None,
    student_id: str | None,
    page_number: int | None,
    file_size: int,
    content_type: str | None,
) -> ValidationResult:
    """Validate upload metadata and file constraints.

    Returns a :class:`ValidationResult` with ``valid=True`` when all
    checks pass, or ``valid=False`` with a human-readable error.
    """
    if not exam_id:
        return ValidationResult(valid=False, error="exam_id is required")
    if not student_id:
        return ValidationResult(valid=False, error="student_id is required")
    if page_number is None:
        return ValidationResult(valid=False, error="page_number is required")
    if page_number < 1:
        return ValidationResult(
            valid=False, error="page_number must be >= 1"
        )
    if file_size <= 0:
        return ValidationResult(valid=False, error="file is empty")
    if file_size > _MAX_FILE_SIZE:
        return ValidationResult(
            valid=False,
            error=f"file exceeds {_MAX_FILE_SIZE // (1024 * 1024)} MB limit",
        )
    if content_type not in _ALLOWED_CONTENT_TYPES:
        return ValidationResult(
            valid=False,
            error=f"unsupported content type: {content_type}",
        )
    return ValidationResult(valid=True)


def validate_magic_bytes(header: bytes) -> ValidationResult:
    """Check the leading bytes of the file against JPEG/PNG signatures.

    Parameters
    ----------
    header:
        The first 8 bytes of the uploaded file.
    """
    if len(header) < 3:
        return ValidationResult(
            valid=False, error="file too short to identify format"
        )
    if header[:3] == _JPEG_MAGIC:
        return ValidationResult(valid=True)
    if len(header) >= 8 and header[:8] == _PNG_MAGIC:
        return ValidationResult(valid=True)
    return ValidationResult(
        valid=False,
        error="file magic bytes do not match JPEG or PNG",
    )


def extension_for_content_type(content_type: str) -> str:
    """Return file extension (without dot) for a supported content type."""
    mapping = {"image/jpeg": "jpg", "image/png": "png"}
    return mapping.get(content_type, "bin")
