"""Pure-logic validation for stroke chunk uploads.

ZERO I/O -- this module must never import asyncio, aiohttp, sqlalchemy,
nats, redis, or any I/O library.
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from typing import Any

# MAC address pattern: six hex-pair groups separated by colons
_MAC_RE = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")

# UUID v4 pattern (loose: accepts any hex in the version nibble)
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

VALID_UPLOAD_PATHS = frozenset({"wifi", "mobile"})

VALID_BINDING_STATUSES = frozenset({
    "unknown", "provisional", "confirmed", "rejected",
})


@dataclass(frozen=True, slots=True)
class ValidationError:
    """Single validation failure."""

    field: str
    message: str


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of chunk validation -- either OK or a list of errors."""

    errors: list[ValidationError]

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0


def make_idempotency_key(
    exam_id: str,
    pen_mac: str,
    chunk_index: int,
) -> str:
    """Deterministic idempotency key for a single chunk."""
    return f"{exam_id}:{pen_mac}:{chunk_index}"


def validate_chunk(payload: dict[str, Any]) -> ValidationResult:
    """Validate a stroke chunk upload request.

    Checks required fields, value ranges, MAC format, UUID format,
    upload_path enum, and CRC-32 checksum integrity.
    """
    errors: list[ValidationError] = []

    # -- required fields ------------------------------------------------
    required = [
        "exam_id", "pen_mac", "chunk_index", "total_chunks",
        "payload_base64", "checksum_crc32", "upload_path",
        "idempotency_key",
    ]
    for field in required:
        if field not in payload or payload[field] is None:
            errors.append(ValidationError(field, "required field missing"))

    # Bail out early if required fields are absent
    if errors:
        return ValidationResult(errors)

    # -- format checks --------------------------------------------------
    exam_id = payload["exam_id"]
    if not isinstance(exam_id, str) or not _UUID_RE.match(exam_id):
        errors.append(ValidationError("exam_id", "must be a valid UUID"))

    pen_mac = payload["pen_mac"]
    if not isinstance(pen_mac, str) or not _MAC_RE.match(pen_mac):
        errors.append(ValidationError("pen_mac", "must be a valid MAC address"))

    chunk_index = payload["chunk_index"]
    if not isinstance(chunk_index, int) or chunk_index < 0:
        errors.append(ValidationError("chunk_index", "must be a non-negative integer"))

    total_chunks = payload["total_chunks"]
    if not isinstance(total_chunks, int) or total_chunks < 1:
        errors.append(ValidationError("total_chunks", "must be a positive integer"))

    if (
        isinstance(chunk_index, int)
        and isinstance(total_chunks, int)
        and chunk_index >= total_chunks
    ):
        errors.append(ValidationError(
            "chunk_index",
            "must be less than total_chunks",
        ))

    upload_path = payload["upload_path"]
    if upload_path not in VALID_UPLOAD_PATHS:
        errors.append(ValidationError("upload_path", "must be 'wifi' or 'mobile'"))

    binding_status = payload.get("binding_status")
    if binding_status is not None and binding_status not in VALID_BINDING_STATUSES:
        errors.append(ValidationError(
            "binding_status",
            "must be one of: unknown, provisional, confirmed, rejected",
        ))

    # -- CRC-32 verification --------------------------------------------
    payload_b64 = payload["payload_base64"]
    checksum_crc32 = payload["checksum_crc32"]
    crc_err = verify_crc32(payload_b64, checksum_crc32)
    if crc_err is not None:
        errors.append(crc_err)

    return ValidationResult(errors)


def verify_crc32(
    payload_base64: str,
    expected_crc32: str,
) -> ValidationError | None:
    """Verify CRC-32 of decoded base64 payload against the expected hex.

    Returns ``None`` on success, or a ``ValidationError`` on mismatch /
    decode failure.
    """
    try:
        raw = base64.b64decode(payload_base64, validate=True)
    except (binascii.Error, ValueError):
        return ValidationError("payload_base64", "invalid base64 encoding")

    computed = binascii.crc32(raw) & 0xFFFFFFFF
    expected_str = expected_crc32.lower().lstrip("0x")
    computed_str = f"{computed:08x}"

    if computed_str != expected_str.zfill(8):
        return ValidationError(
            "checksum_crc32",
            f"CRC-32 mismatch: computed {computed_str}, expected {expected_str.zfill(8)}",
        )
    return None
