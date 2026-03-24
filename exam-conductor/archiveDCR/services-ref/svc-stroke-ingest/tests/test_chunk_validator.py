"""Unit tests for domain/chunk_validator.py — CRC, field, idempotency key.

Test IDs: U-SINGEST-01 through U-SINGEST-12
Markers: unit (ZERO I/O)
"""

from __future__ import annotations

import base64
import binascii

import pytest

from src.domain.chunk_validator import (
    ValidationError,
    make_idempotency_key,
    validate_chunk,
    verify_crc32,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXAM_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
_PEN_MAC = "AA:BB:CC:DD:EE:FF"
_PAYLOAD = b"hello stroke data"
_PAYLOAD_B64 = base64.b64encode(_PAYLOAD).decode()
_CRC32 = f"{binascii.crc32(_PAYLOAD) & 0xFFFFFFFF:08x}"


def _valid_chunk(**overrides) -> dict:
    base = {
        "exam_id": _EXAM_ID,
        "pen_mac": _PEN_MAC,
        "chunk_index": 0,
        "total_chunks": 5,
        "payload_base64": _PAYLOAD_B64,
        "checksum_crc32": _CRC32,
        "upload_path": "wifi",
        "idempotency_key": f"{_EXAM_ID}:{_PEN_MAC}:0",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# U-SINGEST-01: Valid chunk passes validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_valid_chunk_passes():
    result = validate_chunk(_valid_chunk())
    assert result.valid
    assert result.errors == []


# ---------------------------------------------------------------------------
# U-SINGEST-02: Missing required fields rejected
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("field", [
    "exam_id", "pen_mac", "chunk_index", "total_chunks",
    "payload_base64", "checksum_crc32", "upload_path", "idempotency_key",
])
def test_missing_required_field(field: str):
    chunk = _valid_chunk()
    del chunk[field]
    result = validate_chunk(chunk)
    assert not result.valid
    assert any(e.field == field for e in result.errors)


# ---------------------------------------------------------------------------
# U-SINGEST-03: Invalid exam_id format rejected
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_invalid_exam_id():
    result = validate_chunk(_valid_chunk(exam_id="not-a-uuid"))
    assert not result.valid
    assert any(e.field == "exam_id" for e in result.errors)


# ---------------------------------------------------------------------------
# U-SINGEST-04: Invalid pen_mac format rejected
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_invalid_pen_mac():
    result = validate_chunk(_valid_chunk(pen_mac="invalid"))
    assert not result.valid
    assert any(e.field == "pen_mac" for e in result.errors)


# ---------------------------------------------------------------------------
# U-SINGEST-05: Negative chunk_index rejected
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_negative_chunk_index():
    result = validate_chunk(_valid_chunk(chunk_index=-1))
    assert not result.valid
    assert any(e.field == "chunk_index" for e in result.errors)


# ---------------------------------------------------------------------------
# U-SINGEST-06: chunk_index >= total_chunks rejected
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_chunk_index_exceeds_total():
    result = validate_chunk(_valid_chunk(chunk_index=5, total_chunks=5))
    assert not result.valid
    assert any(
        e.field == "chunk_index" and "less than" in e.message
        for e in result.errors
    )


# ---------------------------------------------------------------------------
# U-SINGEST-07: Invalid upload_path rejected
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_invalid_upload_path():
    result = validate_chunk(_valid_chunk(upload_path="bluetooth"))
    assert not result.valid
    assert any(e.field == "upload_path" for e in result.errors)


# ---------------------------------------------------------------------------
# U-SINGEST-08: CRC-32 mismatch detected
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_crc32_mismatch():
    result = validate_chunk(_valid_chunk(checksum_crc32="deadbeef"))
    assert not result.valid
    assert any(
        e.field == "checksum_crc32" and "mismatch" in e.message
        for e in result.errors
    )


# ---------------------------------------------------------------------------
# U-SINGEST-09: Invalid base64 payload detected
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_invalid_base64():
    result = validate_chunk(_valid_chunk(payload_base64="not!valid!b64!"))
    assert not result.valid
    assert any(e.field == "payload_base64" for e in result.errors)


# ---------------------------------------------------------------------------
# U-SINGEST-10: CRC-32 verification passes for correct data
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_verify_crc32_passes():
    err = verify_crc32(_PAYLOAD_B64, _CRC32)
    assert err is None


# ---------------------------------------------------------------------------
# U-SINGEST-11: Idempotency key generation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_idempotency_key_format():
    key = make_idempotency_key("exam-1", "AA:BB:CC:DD:EE:FF", 3)
    assert key == "exam-1:AA:BB:CC:DD:EE:FF:3"


# ---------------------------------------------------------------------------
# U-SINGEST-12: Valid binding_status accepted; invalid rejected
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_valid_binding_status():
    result = validate_chunk(_valid_chunk(binding_status="confirmed"))
    assert result.valid


@pytest.mark.unit
def test_invalid_binding_status():
    result = validate_chunk(_valid_chunk(binding_status="bogus"))
    assert not result.valid
    assert any(e.field == "binding_status" for e in result.errors)
