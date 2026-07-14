"""Single source of truth for upload policy defaults and env overrides."""

from __future__ import annotations

import os
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field


MB = 1024 * 1024


class UploadPolicyConfigError(RuntimeError):
    """Raised when upload policy configuration is missing or invalid."""


class UploadPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    policy_id: str
    policy_kind: Literal["binary", "structured"] = "binary"
    max_size_bytes: int = Field(gt=0)
    max_files: int | None = None
    max_total_size_bytes: int | None = None
    allowed_extensions: tuple[str, ...] = ()
    allowed_mime_types: tuple[str, ...] = ()
    allowed_magic_types: tuple[str, ...] = ()
    allow_octet_stream: bool = False
    max_pdf_pages: int | None = None
    max_image_pixels: int | None = None
    max_archive_entries: int | None = None
    max_archive_uncompressed_bytes: int | None = None
    max_archive_depth: int | None = None
    max_rows: int | None = None
    max_columns: int | None = None
    max_sessions: int | None = None
    max_frames_per_session: int | None = None
    max_frames_per_batch: int | None = None
    max_frame_json_bytes: int | None = None
    max_payload_base64_bytes: int | None = None
    max_decoded_payload_bytes: int | None = None
    max_total_chunks: int | None = None
    max_pages: int | None = None
    max_strokes_per_page: int | None = None


class BinaryUploadPolicy(UploadPolicy):
    policy_kind: Literal["binary"] = "binary"


class StructuredUploadPolicy(UploadPolicy):
    policy_kind: Literal["structured"] = "structured"


def _types(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(value.lower().lstrip(".") for value in values)


DEFAULT_UPLOAD_POLICIES: dict[str, UploadPolicy] = {
    "registration_document": BinaryUploadPolicy(
        policy_id="registration_document",
        max_size_bytes=20 * MB,
        max_files=8,
        max_total_size_bytes=160 * MB,
        allowed_extensions=_types(["pdf", "png", "jpg", "jpeg"]),
        allowed_mime_types=_types(["application/pdf", "image/png", "image/jpeg"]),
        allowed_magic_types=_types(["pdf", "png", "jpeg"]),
        allow_octet_stream=True,
    ),
    "registration_reply_attachment": BinaryUploadPolicy(
        policy_id="registration_reply_attachment",
        max_size_bytes=20 * MB,
        max_files=10,
        max_total_size_bytes=100 * MB,
        allowed_extensions=_types(["pdf", "png", "jpg", "jpeg"]),
        allowed_mime_types=_types(["application/pdf", "image/png", "image/jpeg"]),
        allowed_magic_types=_types(["pdf", "png", "jpeg"]),
        allow_octet_stream=True,
    ),
    "support_message_attachment": BinaryUploadPolicy(
        policy_id="support_message_attachment",
        max_size_bytes=20 * MB,
        max_files=10,
        max_total_size_bytes=100 * MB,
        allowed_extensions=_types(["pdf", "png", "jpg", "jpeg"]),
        allowed_mime_types=_types(["application/pdf", "image/png", "image/jpeg"]),
        allowed_magic_types=_types(["pdf", "png", "jpeg"]),
        allow_octet_stream=True,
    ),
    "debugger_document": BinaryUploadPolicy(
        policy_id="debugger_document",
        max_size_bytes=10 * MB,
        allowed_extensions=_types(["pdf", "docx", "doc", "png", "jpg", "jpeg", "webp"]),
        allowed_mime_types=_types(
            [
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword",
                "image/png",
                "image/jpeg",
                "image/webp",
            ]
        ),
        allowed_magic_types=_types(["pdf", "zip", "ole", "png", "jpeg", "webp"]),
        allow_octet_stream=True,
        max_pdf_pages=100,
        max_image_pixels=25_000_000,
        max_archive_entries=500,
        max_archive_uncompressed_bytes=50 * MB,
        max_archive_depth=4,
    ),
    "stoody_book_pdf": BinaryUploadPolicy(
        policy_id="stoody_book_pdf",
        max_size_bytes=10 * MB,
        allowed_extensions=_types(["pdf"]),
        allowed_mime_types=_types(["application/pdf"]),
        allowed_magic_types=_types(["pdf"]),
        allow_octet_stream=True,
        max_pdf_pages=100,
    ),
    "pdf_document": BinaryUploadPolicy(
        policy_id="pdf_document",
        max_size_bytes=50 * MB,
        allowed_extensions=_types(["pdf"]),
        allowed_mime_types=_types(["application/pdf"]),
        allowed_magic_types=_types(["pdf"]),
        allow_octet_stream=True,
        max_pdf_pages=250,
    ),
    "answer_sheet_pdf": BinaryUploadPolicy(
        policy_id="answer_sheet_pdf",
        max_size_bytes=25 * MB,
        allowed_extensions=_types(["pdf"]),
        allowed_mime_types=_types(["application/pdf"]),
        allowed_magic_types=_types(["pdf"]),
        allow_octet_stream=True,
        max_pdf_pages=150,
    ),
    "exam_template_file": BinaryUploadPolicy(
        policy_id="exam_template_file",
        max_size_bytes=25 * MB,
        allowed_extensions=_types(["pdf", "png", "jpg", "jpeg"]),
        allowed_mime_types=_types(["application/pdf", "image/png", "image/jpeg"]),
        allowed_magic_types=_types(["pdf", "png", "jpeg"]),
        allow_octet_stream=True,
        max_pdf_pages=50,
        max_image_pixels=25_000_000,
    ),
    "direct_ocr_pdf": BinaryUploadPolicy(
        policy_id="direct_ocr_pdf",
        max_size_bytes=25 * MB,
        allowed_extensions=_types(["pdf"]),
        allowed_mime_types=_types(["application/pdf"]),
        allowed_magic_types=_types(["pdf"]),
        allow_octet_stream=True,
        max_pdf_pages=100,
    ),
    "tally_question_source_pdf": BinaryUploadPolicy(
        policy_id="tally_question_source_pdf",
        max_size_bytes=25 * MB,
        allowed_extensions=_types(["pdf"]),
        allowed_mime_types=_types(["application/pdf"]),
        allowed_magic_types=_types(["pdf"]),
        allow_octet_stream=True,
        max_pdf_pages=100,
    ),
    "manual_question_image": BinaryUploadPolicy(
        policy_id="manual_question_image",
        max_size_bytes=8 * MB,
        max_files=11,
        max_total_size_bytes=80 * MB,
        allowed_extensions=_types(["png", "jpg", "jpeg", "webp"]),
        allowed_mime_types=_types(["image/png", "image/jpeg", "image/webp"]),
        allowed_magic_types=_types(["png", "jpeg", "webp"]),
        max_image_pixels=25_000_000,
    ),
    "generic_image_upload": BinaryUploadPolicy(
        policy_id="generic_image_upload",
        max_size_bytes=10 * MB,
        allowed_extensions=_types(["png", "jpg", "jpeg", "gif", "bmp", "webp"]),
        allowed_mime_types=_types(["image/png", "image/jpeg", "image/gif", "image/bmp", "image/webp"]),
        allowed_magic_types=_types(["png", "jpeg", "gif", "bmp", "webp"]),
        max_image_pixels=25_000_000,
    ),
    "school_logo": BinaryUploadPolicy(
        policy_id="school_logo",
        max_size_bytes=3 * MB,
        allowed_extensions=_types(["png", "jpg", "jpeg", "webp"]),
        allowed_mime_types=_types(["image/png", "image/jpeg", "image/webp"]),
        allowed_magic_types=_types(["png", "jpeg", "webp"]),
        max_image_pixels=8_000_000,
    ),
    "bulk_students": BinaryUploadPolicy(
        policy_id="bulk_students",
        max_size_bytes=5 * MB,
        allowed_extensions=_types(["csv", "xlsx", "xls"]),
        allowed_mime_types=_types(
            [
                "text/csv",
                "application/csv",
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ]
        ),
        allowed_magic_types=_types(["csv", "zip", "ole"]),
        allow_octet_stream=True,
        max_rows=10_000,
        max_columns=80,
        max_archive_entries=500,
        max_archive_uncompressed_bytes=50 * MB,
        max_archive_depth=4,
    ),
    "bulk_tutors": BinaryUploadPolicy(
        policy_id="bulk_tutors",
        max_size_bytes=5 * MB,
        allowed_extensions=_types(["csv", "xlsx", "xls"]),
        allowed_mime_types=_types(
            [
                "text/csv",
                "application/csv",
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ]
        ),
        allowed_magic_types=_types(["csv", "zip", "ole"]),
        allow_octet_stream=True,
        max_rows=10_000,
        max_columns=80,
        max_archive_entries=500,
        max_archive_uncompressed_bytes=50 * MB,
        max_archive_depth=4,
    ),
    "bulk_timetable": BinaryUploadPolicy(
        policy_id="bulk_timetable",
        max_size_bytes=10 * MB,
        allowed_extensions=_types(["csv", "xlsx", "xls"]),
        allowed_mime_types=_types(
            [
                "text/csv",
                "application/csv",
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ]
        ),
        allowed_magic_types=_types(["csv", "zip", "ole"]),
        allow_octet_stream=True,
        max_rows=20_000,
        max_columns=80,
        max_archive_entries=500,
        max_archive_uncompressed_bytes=50 * MB,
        max_archive_depth=4,
    ),
    "teaching_material": BinaryUploadPolicy(
        policy_id="teaching_material",
        max_size_bytes=50 * MB,
        allowed_extensions=_types(
            ["pdf", "doc", "docx", "ppt", "pptx", "png", "jpg", "jpeg", "gif", "webp", "mp4", "mov", "webm", "mkv"]
        ),
        allowed_mime_types=_types(
            [
                "application/pdf",
                "application/msword",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/vnd.ms-powerpoint",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "image/png",
                "image/jpeg",
                "image/gif",
                "image/webp",
                "video/mp4",
                "video/quicktime",
                "video/webm",
                "video/x-matroska",
            ]
        ),
        allowed_magic_types=_types(["pdf", "zip", "ole", "png", "jpeg", "gif", "webp", "mp4", "mov", "ebml"]),
        allow_octet_stream=True,
        max_pdf_pages=250,
        max_image_pixels=25_000_000,
        max_archive_entries=1000,
        max_archive_uncompressed_bytes=100 * MB,
        max_archive_depth=4,
    ),
    "desktop_diagnostics_zip": BinaryUploadPolicy(
        policy_id="desktop_diagnostics_zip",
        max_size_bytes=25 * MB,
        allowed_extensions=_types(["zip"]),
        allowed_mime_types=_types(["application/zip", "application/x-zip-compressed"]),
        allowed_magic_types=_types(["zip"]),
        allow_octet_stream=True,
        max_archive_entries=500,
        max_archive_uncompressed_bytes=100 * MB,
        max_archive_depth=4,
    ),
    "desktop_bug_image": BinaryUploadPolicy(
        policy_id="desktop_bug_image",
        max_size_bytes=5 * MB,
        max_files=8,
        max_total_size_bytes=20 * MB,
        allowed_extensions=_types(["png", "jpg", "jpeg", "webp", "bmp"]),
        allowed_mime_types=_types(["image/png", "image/jpeg", "image/webp", "image/bmp"]),
        allowed_magic_types=_types(["png", "jpeg", "webp", "bmp"]),
        max_image_pixels=25_000_000,
    ),
    "camera_answer_image": BinaryUploadPolicy(
        policy_id="camera_answer_image",
        max_size_bytes=12 * MB,
        allowed_extensions=_types(["jpg", "jpeg", "png"]),
        allowed_mime_types=_types(["image/jpeg", "image/png"]),
        allowed_magic_types=_types(["jpeg", "png"]),
        max_image_pixels=25_000_000,
    ),
    # Student answer copies use the same safe raster limits as the
    # invigilator camera path, with a separate policy/audit identity.  The
    # route-level policy provides an aggregate request limit; individual
    # image/PDF fields are checked again with their stricter policies below.
    "student_answer_copy_upload": BinaryUploadPolicy(
        policy_id="student_answer_copy_upload",
        max_size_bytes=80 * MB,
        max_files=50,
        max_total_size_bytes=100 * MB,
        allowed_extensions=_types(["jpg", "jpeg", "png", "pdf"]),
        allowed_mime_types=_types(["image/jpeg", "image/png", "application/pdf"]),
        allowed_magic_types=_types(["jpeg", "png", "pdf"]),
        allow_octet_stream=True,
        max_pdf_pages=50,
        max_image_pixels=25_000_000,
    ),
    "student_answer_copy_image": BinaryUploadPolicy(
        policy_id="student_answer_copy_image",
        max_size_bytes=12 * MB,
        max_files=50,
        max_total_size_bytes=100 * MB,
        allowed_extensions=_types(["jpg", "jpeg", "png"]),
        allowed_mime_types=_types(["image/jpeg", "image/png"]),
        allowed_magic_types=_types(["jpeg", "png"]),
        max_image_pixels=25_000_000,
    ),
    "student_answer_copy_pdf": BinaryUploadPolicy(
        policy_id="student_answer_copy_pdf",
        max_size_bytes=40 * MB,
        allowed_extensions=_types(["pdf"]),
        allowed_mime_types=_types(["application/pdf"]),
        allowed_magic_types=_types(["pdf"]),
        allow_octet_stream=True,
        max_pdf_pages=50,
    ),
    "hub_raw_data_batch": StructuredUploadPolicy(
        policy_id="hub_raw_data_batch",
        max_size_bytes=50 * MB,
        allowed_extensions=_types(["json"]),
        allowed_mime_types=_types(["application/json"]),
        allowed_magic_types=_types(["json"]),
        max_sessions=20,
        max_frames_per_session=50_000,
        max_frames_per_batch=100_000,
        max_frame_json_bytes=8 * 1024,
    ),
    "hub_stroke_chunk": StructuredUploadPolicy(
        policy_id="hub_stroke_chunk",
        max_size_bytes=768 * 1024,
        allowed_extensions=_types(["json"]),
        allowed_mime_types=_types(["application/json"]),
        allowed_magic_types=_types(["json"]),
        max_payload_base64_bytes=512 * 1024,
        max_decoded_payload_bytes=384 * 1024,
        max_total_chunks=5000,
    ),
    "hub_stroke_finalize": StructuredUploadPolicy(
        policy_id="hub_stroke_finalize",
        max_size_bytes=10 * MB,
        allowed_extensions=_types(["json"]),
        allowed_mime_types=_types(["application/json"]),
        allowed_magic_types=_types(["json"]),
        max_total_chunks=5000,
        max_pages=500,
        max_strokes_per_page=20_000,
    ),
}


_BYTE_FIELD_MB_ENV_SUFFIXES = {
    "max_size_bytes": "MAX_SIZE_MB",
    "max_total_size_bytes": "MAX_TOTAL_SIZE_MB",
    "max_archive_uncompressed_bytes": "MAX_ARCHIVE_UNCOMPRESSED_MB",
}


def _override_env_name(policy_id: str, field_name: str) -> str:
    return f"UPLOAD_POLICY_{policy_id.upper()}_{field_name.upper()}"


def _coerce_override(raw_value: str, current_value: Any, field_name: str) -> Any:
    if isinstance(current_value, bool):
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current_value, int) or current_value is None and field_name.startswith("max_"):
        return int(raw_value)
    if isinstance(current_value, tuple):
        return _types(part.strip() for part in raw_value.split(",") if part.strip())
    return raw_value


def _apply_env_overrides(policy: UploadPolicy) -> UploadPolicy:
    data = policy.model_dump()
    for field_name, current_value in data.items():
        env_name = _override_env_name(policy.policy_id, field_name)
        raw_value = os.getenv(env_name)
        if raw_value is None and field_name in _BYTE_FIELD_MB_ENV_SUFFIXES:
            raw_mb = os.getenv(f"UPLOAD_POLICY_{policy.policy_id.upper()}_{_BYTE_FIELD_MB_ENV_SUFFIXES[field_name]}")
            if raw_mb is not None:
                raw_value = str(int(raw_mb) * MB)
        if raw_value is None:
            continue
        try:
            data[field_name] = _coerce_override(raw_value, current_value, field_name)
        except (TypeError, ValueError) as exc:
            raise UploadPolicyConfigError(f"Invalid {env_name} override for {policy.policy_id}") from exc
    model_type = StructuredUploadPolicy if policy.policy_kind == "structured" else BinaryUploadPolicy
    return model_type(**data)


def get_upload_policy(policy_id: str) -> UploadPolicy:
    try:
        policy = DEFAULT_UPLOAD_POLICIES[policy_id]
    except KeyError as exc:
        raise UploadPolicyConfigError(f"Unknown upload policy: {policy_id}") from exc
    return _apply_env_overrides(policy)


def all_public_upload_policies() -> dict[str, dict[str, Any]]:
    public: dict[str, dict[str, Any]] = {}
    for policy_id in sorted(DEFAULT_UPLOAD_POLICIES):
        policy = get_upload_policy(policy_id)
        data = policy.model_dump(
            include={
                "policy_id",
                "policy_kind",
                "max_size_bytes",
                "max_files",
                "max_total_size_bytes",
                "allowed_extensions",
                "allowed_mime_types",
                "max_pdf_pages",
                "max_image_pixels",
                "max_rows",
                "max_columns",
                "max_sessions",
                "max_frames_per_session",
                "max_frames_per_batch",
                "max_total_chunks",
                "max_pages",
                "max_strokes_per_page",
            }
        )
        data["max_size_mb"] = round(policy.max_size_bytes / MB, 3)
        if policy.max_total_size_bytes is not None:
            data["max_total_size_mb"] = round(policy.max_total_size_bytes / MB, 3)
        public[policy_id] = data
    return public
