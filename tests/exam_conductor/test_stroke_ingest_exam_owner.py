from __future__ import annotations

import base64
import hashlib
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


def _fresh_db():
    from mongomock_motor import AsyncMongoMockClient

    return AsyncMongoMockClient()["skb_test"]


def _hub_user(hub_id: str = "HUB-1") -> Dict[str, Any]:
    return {
        "user_id": hub_id,
        "user_type": "hub",
        "hub_id": hub_id,
        "db_name": "skb_test",
        "scopes": ["hub:data:upload"],
    }


def _admin_user(admin_id: str = "ADMIN-CALLER") -> Dict[str, Any]:
    return {"user_id": admin_id, "user_type": "admin", "db_name": "skb_test"}


def _tutor_user(tutor_id: str = "TUT-1") -> Dict[str, Any]:
    return {
        "user_id": f"user-{tutor_id}",
        "user_type": "tutor",
        "tutor_id": tutor_id,
        "admin_id": "ADMIN-1",
        "db_name": "skb_test",
    }


async def _seed_exam(db, **overrides) -> None:
    doc = {
        "exam_id": "EXAM-1",
        "exam_type": "pcr",
        "admin_id": "ADMIN-OWNER",
        "teacher_ids": ["TUT-1"],
        "created_by_tutor_id": None,
        "lifecycle_state": "uploading",
        "hub_assignments": [{"hub_id": "HUB-1", "status": "active"}],
        "pen_bindings": {"AA:BB:CC:DD:EE:FF": "STU-1"},
    }
    doc.update(overrides)
    await db["exampen_exams"].insert_one(doc)


async def _seed_chunk(db, *, student_id: str = "STU-1", pen_mac: str = "AA:BB:CC:DD:EE:FF"):
    payload_b64 = base64.b64encode(b"stroke-payload").decode()
    checksum = hashlib.sha256(payload_b64.encode()).hexdigest()
    await db["exampen_stroke_chunks"].insert_one(
        {
            "artifact_id": "artifact-1",
            "exam_id": "EXAM-1",
            "exam_type": "pcr",
            "hub_id": "HUB-1",
            "pen_mac": pen_mac,
            "student_id": student_id,
            "chunk_index": 0,
            "total_chunks": 1,
            "payload_base64": payload_b64,
            "payload_hash": checksum,
            "dedup_hash": "dedup-1",
            "finalized": False,
        }
    )
    return checksum


def _fake_ingest(captured: Dict[str, Any]):
    class _Result:
        submission_id = "SUB-1"
        content_hash = "c" * 64
        page_count = 1
        segmentation_status = "pending"
        already_existed = False

    service = MagicMock()
    service.initialize = AsyncMock(return_value=None)
    service.ingest_submission = AsyncMock(
        side_effect=lambda **kwargs: captured.update(kwargs) or _Result()
    )
    module = MagicMock()
    module.IngestService = MagicMock(return_value=service)
    return patch("api.v1._exampen_imports.load_exampen", return_value=module)


async def _finalize(db, current_user, *, hub_id: Optional[str] = "HUB-1", student_id: str = "STU-1"):
    from api.v1.stroke_ingest_async import FinalizeRequest, finalize_pen_upload

    checksum = await _seed_chunk(db, student_id=student_id)
    body = FinalizeRequest(
        student_id=student_id,
        expected_checksum=checksum,
        total_chunks=1,
        hub_id=hub_id,
    )
    with patch("api.v1.stroke_ingest_async._get_tenant_db", return_value=db):
        return await finalize_pen_upload(
            exam_id="EXAM-1",
            pen_mac="AA:BB:CC:DD:EE:FF",
            body=body,
            current_user=current_user,
            db=None,
        )


async def _upload_chunk(db, current_user, **overrides):
    from api.v1.stroke_ingest_async import StrokeChunkUpload, upload_stroke_chunk

    payload_b64 = base64.b64encode(b"stroke-payload").decode()
    body = StrokeChunkUpload(
        exam_type=overrides.pop("exam_type", "pcr"),
        student_id=overrides.pop("student_id", "STU-1"),
        chunk_index=0,
        total_chunks=1,
        payload_base64=payload_b64,
        payload_hash=overrides.pop(
            "payload_hash",
            hashlib.sha256(payload_b64.encode()).hexdigest(),
        ),
        hub_id=overrides.pop("hub_id", "HUB-1"),
        metadata=overrides.pop("metadata", {}),
        **overrides,
    )
    with patch("api.v1.stroke_ingest_async._get_tenant_db", return_value=db):
        return await upload_stroke_chunk(
            exam_id="EXAM-1",
            pen_mac="AA:BB:CC:DD:EE:FF",
            body=body,
            current_user=current_user,
            db=None,
        )


@pytest.mark.asyncio
async def test_hub_finalize_uses_exam_owner_admin_id():
    db = _fresh_db()
    await _seed_exam(db, admin_id="ADMIN-OWNER")
    captured: Dict[str, Any] = {}

    with _fake_ingest(captured):
        await _finalize(db, _hub_user("HUB-1"))

    assert captured["admin_id"] == "ADMIN-OWNER"


@pytest.mark.asyncio
async def test_hub_finalize_requires_assigned_hub_and_matching_body_hub_id():
    db = _fresh_db()
    await _seed_exam(db, hub_assignments=[{"hub_id": "HUB-1", "status": "active"}])

    with _fake_ingest({}):
        with pytest.raises(HTTPException) as exc:
            await _finalize(db, _hub_user("HUB-2"), hub_id="HUB-2")
    assert exc.value.status_code == 403

    with _fake_ingest({}):
        with pytest.raises(HTTPException) as exc:
            await _finalize(db, _hub_user("HUB-1"), hub_id="HUB-OTHER")
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_tutor_finalize_respects_exam_visibility():
    db = _fresh_db()
    await _seed_exam(db, teacher_ids=["TUT-2"])

    with _fake_ingest({}):
        with pytest.raises(HTTPException) as exc:
            await _finalize(db, _tutor_user("TUT-1"), hub_id=None)

    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_upload_chunk_uses_canonical_exam_type_and_records_hub_id():
    db = _fresh_db()
    await _seed_exam(db, exam_type="dcr")

    result = await _upload_chunk(db, _hub_user("HUB-1"), exam_type="dcr")
    stored = await db["exampen_stroke_chunks"].find_one({"artifact_id": result.artifact_id})

    assert stored["exam_type"] == "dcr"
    assert stored["hub_id"] == "HUB-1"


@pytest.mark.asyncio
async def test_upload_chunk_rejects_payload_hash_mismatch():
    db = _fresh_db()
    await _seed_exam(db, exam_type="pcr")

    with pytest.raises(HTTPException) as exc:
        await _upload_chunk(db, _hub_user("HUB-1"), payload_hash="bad-hash")

    assert exc.value.status_code == 400
    assert "payload_hash mismatch" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_upload_chunk_rejects_exam_type_and_student_binding_mismatch():
    db = _fresh_db()
    await _seed_exam(db, exam_type="pcr", pen_bindings={"AA:BB:CC:DD:EE:FF": "STU-1"})

    with pytest.raises(HTTPException) as exc:
        await _upload_chunk(db, _hub_user("HUB-1"), exam_type="dcr")
    assert exc.value.status_code == 400

    with pytest.raises(HTTPException) as exc:
        await _upload_chunk(db, _hub_user("HUB-1"), student_id="STU-OTHER")
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_missing_exam_owner_admin_id_rejects_before_ingest():
    db = _fresh_db()
    await _seed_exam(db, admin_id="")
    captured: Dict[str, Any] = {}

    with _fake_ingest(captured):
        with pytest.raises(HTTPException) as exc:
            await _finalize(db, _admin_user(), hub_id=None)

    assert exc.value.status_code == 400
    assert captured == {}
