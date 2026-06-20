import base64

import pytest
from pydantic import ValidationError

from api.v1.stroke_ingest_async import FinalizeRequest, StrokeChunkUpload
from core.upload_security.routes import resolve_upload_policy_for_route


def test_stroke_routes_use_specific_chunk_and_finalize_policies():
    assert (
        resolve_upload_policy_for_route("POST", "/api/v1/ingest/strokes/exam-1/AA:BB").policy_id
        == "hub_stroke_chunk"
    )
    assert (
        resolve_upload_policy_for_route("POST", "/api/v1/ingest/strokes/exam-1/AA:BB/complete").policy_id
        == "hub_stroke_finalize"
    )


def test_stroke_chunk_rejects_invalid_base64():
    with pytest.raises(ValidationError):
        StrokeChunkUpload(
            exam_type="pcr",
            student_id="student-1",
            chunk_index=0,
            total_chunks=1,
            payload_base64="not-valid-base64!",
        )


def test_stroke_chunk_rejects_decoded_payload_over_limit():
    payload = base64.b64encode(b"x" * (384 * 1024 + 1)).decode("ascii")

    with pytest.raises(ValidationError):
        StrokeChunkUpload(
            exam_type="pcr",
            student_id="student-1",
            chunk_index=0,
            total_chunks=1,
            payload_base64=payload,
        )


def test_stroke_finalize_rejects_invalid_checksum_and_excess_pages():
    with pytest.raises(ValidationError):
        FinalizeRequest(student_id="student-1", expected_checksum="not-hex", total_chunks=1)

    with pytest.raises(ValidationError):
        FinalizeRequest(
            student_id="student-1",
            expected_checksum="a" * 64,
            total_chunks=1,
            pages=[{"page_number": 1, "raw_strokes": []} for _ in range(501)],
        )
