"""Unit tests for domain upload validation logic (ZERO I/O).

Test IDs: U-COPY-01 through U-COPY-12
"""

import pytest

from src.domain.upload_validator import (
    ValidationResult,
    extension_for_content_type,
    validate_magic_bytes,
    validate_upload,
)

# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------


class TestValidateUpload:
    """U-COPY-01..08 — metadata and file constraint checks."""

    def test_valid_jpeg_upload(self) -> None:
        """U-COPY-01: Accept valid JPEG metadata."""
        result = validate_upload(
            exam_id="abc-123",
            student_id="stu-456",
            page_number=1,
            file_size=500_000,
            content_type="image/jpeg",
        )
        assert result.valid is True
        assert result.error is None

    def test_valid_png_upload(self) -> None:
        """U-COPY-02: Accept valid PNG metadata."""
        result = validate_upload(
            exam_id="abc-123",
            student_id="stu-456",
            page_number=3,
            file_size=1_000_000,
            content_type="image/png",
        )
        assert result.valid is True

    def test_missing_exam_id(self) -> None:
        """U-COPY-03: Reject when exam_id is empty."""
        result = validate_upload(
            exam_id="",
            student_id="stu-456",
            page_number=1,
            file_size=100,
            content_type="image/jpeg",
        )
        assert result.valid is False
        assert "exam_id" in (result.error or "")

    def test_missing_student_id(self) -> None:
        """U-COPY-04: Reject when student_id is empty."""
        result = validate_upload(
            exam_id="abc-123",
            student_id="",
            page_number=1,
            file_size=100,
            content_type="image/jpeg",
        )
        assert result.valid is False
        assert "student_id" in (result.error or "")

    def test_missing_page_number(self) -> None:
        """U-COPY-05: Reject when page_number is None."""
        result = validate_upload(
            exam_id="abc-123",
            student_id="stu-456",
            page_number=None,
            file_size=100,
            content_type="image/jpeg",
        )
        assert result.valid is False
        assert "page_number" in (result.error or "")

    def test_negative_page_number(self) -> None:
        """U-COPY-05b: Reject when page_number < 1."""
        result = validate_upload(
            exam_id="abc-123",
            student_id="stu-456",
            page_number=0,
            file_size=100,
            content_type="image/jpeg",
        )
        assert result.valid is False
        assert "page_number" in (result.error or "")

    def test_empty_file(self) -> None:
        """U-COPY-06: Reject zero-byte uploads."""
        result = validate_upload(
            exam_id="abc-123",
            student_id="stu-456",
            page_number=1,
            file_size=0,
            content_type="image/jpeg",
        )
        assert result.valid is False
        assert "empty" in (result.error or "")

    def test_oversized_file(self) -> None:
        """U-COPY-07: Reject files over 10 MB."""
        result = validate_upload(
            exam_id="abc-123",
            student_id="stu-456",
            page_number=1,
            file_size=11 * 1024 * 1024,
            content_type="image/jpeg",
        )
        assert result.valid is False
        assert "10 MB" in (result.error or "")

    def test_unsupported_content_type(self) -> None:
        """U-COPY-08: Reject non-JPEG/PNG content types."""
        result = validate_upload(
            exam_id="abc-123",
            student_id="stu-456",
            page_number=1,
            file_size=100,
            content_type="image/gif",
        )
        assert result.valid is False
        assert "unsupported" in (result.error or "")


# ---------------------------------------------------------------------------
# Magic byte validation
# ---------------------------------------------------------------------------


class TestValidateMagicBytes:
    """U-COPY-09..11 — file signature checks."""

    def test_jpeg_magic(self) -> None:
        """U-COPY-09: Accept JPEG magic bytes."""
        header = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        assert validate_magic_bytes(header).valid is True

    def test_png_magic(self) -> None:
        """U-COPY-10: Accept PNG magic bytes."""
        header = b"\x89PNG\r\n\x1a\n"
        assert validate_magic_bytes(header).valid is True

    def test_invalid_magic(self) -> None:
        """U-COPY-11: Reject unrecognised magic bytes."""
        header = b"GIF89a\x00\x00"
        result = validate_magic_bytes(header)
        assert result.valid is False
        assert "magic bytes" in (result.error or "")

    def test_too_short(self) -> None:
        """U-COPY-11b: Reject file too short to identify."""
        result = validate_magic_bytes(b"\xff\xd8")
        assert result.valid is False
        assert "too short" in (result.error or "")


# ---------------------------------------------------------------------------
# Extension mapping
# ---------------------------------------------------------------------------


class TestExtensionForContentType:
    """U-COPY-12 — content-type to extension mapping."""

    @pytest.mark.parametrize(
        "ct,expected",
        [
            ("image/jpeg", "jpg"),
            ("image/png", "png"),
            ("application/octet-stream", "bin"),
        ],
    )
    def test_mapping(self, ct: str, expected: str) -> None:
        """U-COPY-12: Map content type to file extension."""
        assert extension_for_content_type(ct) == expected
