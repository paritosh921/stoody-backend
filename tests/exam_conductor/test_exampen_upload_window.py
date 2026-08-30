from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from services.exampen_upload_window import (
    answer_copy_upload_is_open,
    answer_copy_upload_state,
    close_answer_copy_uploads,
    release_answer_copy_ingest,
    reopen_answer_copy_uploads,
    reserve_answer_copy_ingest,
)


def _fresh_db():
    from mongomock_motor import AsyncMongoMockClient

    return AsyncMongoMockClient()["skb_test"]


def _supported_economy_exam(exam_id: str) -> dict:
    return {
        "exam_id": exam_id,
        "exam_type": "pcr",
        "lifecycle_state": "collection_closed",
        "checking_mode": "economy",
        "answer_copy_upload_state": "open",
        "pcr_grading_contract": {
            "prompt_version": "pcr-full-document-visual-v16",
            "pipeline_version": 7,
            "mapping_pipeline_version": "whole-copy-rubric-v7",
            "required_processing_path": "full_document_visual",
        },
        "created_by": "teacher-1",
        "roster": ["student-1"],
        "hub_assignments": [],
    }


def _live_economy_exam(exam_id: str) -> dict:
    exam = _supported_economy_exam(exam_id)
    exam["lifecycle_state"] = "in_progress"
    return exam


@pytest.mark.asyncio
async def test_close_waits_for_a_copy_that_is_becoming_canonical():
    db = _fresh_db()
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "exam-window-race",
            "exam_type": "pcr",
            "lifecycle_state": "collection_closed",
            "answer_copy_upload_state": "open",
        }
    )
    token = await reserve_answer_copy_ingest(
        db,
        exam_id="exam-window-race",
        actor_id="student-1",
    )

    with pytest.raises(HTTPException) as exc:
        await close_answer_copy_uploads(
            db,
            exam_id="exam-window-race",
            actor_id="teacher-1",
        )

    assert exc.value.status_code == 409
    assert "still being finalized" in str(exc.value.detail)
    await release_answer_copy_ingest(
        db,
        exam_id="exam-window-race",
        reservation_token=token,
    )

    closed = await close_answer_copy_uploads(
        db,
        exam_id="exam-window-race",
        actor_id="teacher-1",
    )
    assert closed["answer_copy_upload_state"] == "closed"
    assert closed["lifecycle_state"] == "uploading"


@pytest.mark.asyncio
async def test_closed_window_rejects_new_copy_reservations_until_reopened():
    db = _fresh_db()
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "exam-window-closed",
            "exam_type": "pcr",
            "lifecycle_state": "uploading",
            "answer_copy_upload_state": "closed",
        }
    )

    with pytest.raises(HTTPException) as exc:
        await reserve_answer_copy_ingest(
            db,
            exam_id="exam-window-closed",
            actor_id="student-1",
        )
    assert exc.value.status_code == 409

    reopened = await reopen_answer_copy_uploads(
        db,
        exam_id="exam-window-closed",
        actor_id="teacher-1",
    )
    assert reopened["answer_copy_upload_state"] == "open"
    token = await reserve_answer_copy_ingest(
        db,
        exam_id="exam-window-closed",
        actor_id="student-1",
    )
    assert token


@pytest.mark.asyncio
async def test_close_rejects_unfinished_legacy_camera_pages():
    db = _fresh_db()
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "exam-camera-draft",
            "exam_type": "pcr",
            "lifecycle_state": "collection_closed",
            "answer_copy_upload_state": "open",
        }
    )
    await db["exampen_camera_uploads"].insert_one(
        {
            "exam_id": "exam-camera-draft",
            "student_id": "student-7",
            "page_number": 1,
        }
    )

    with pytest.raises(HTTPException) as exc:
        await close_answer_copy_uploads(
            db,
            exam_id="exam-camera-draft",
            actor_id="teacher-1",
        )

    assert exc.value.status_code == 409
    assert "student-7" in str(exc.value.detail)
    await db["exampen_camera_uploads"].update_one(
        {"exam_id": "exam-camera-draft"},
        {"$set": {"submission_id": "submission-7"}},
    )
    closed = await close_answer_copy_uploads(
        db,
        exam_id="exam-camera-draft",
        actor_id="teacher-1",
    )
    assert closed["answer_copy_upload_state"] == "closed"


@pytest.mark.asyncio
async def test_start_economy_batch_closes_uploads_before_snapshot():
    from api.v1.exam_orch_async import start_economy_batch

    db = _fresh_db()
    await db["exampen_exams"].insert_one(_supported_economy_exam("exam-start"))
    create_group = AsyncMock(
        return_value={
            "batch_group_id": "econ-start",
            "exam_id": "exam-start",
            "status": "provider_processing",
            "requested_count": 1,
        }
    )
    with (
        patch("api.v1.exam_orch_async._get_tenant_db", return_value=db),
        patch("services.exampen_openai_batch.create_economy_batch_group", new=create_group),
    ):
        result = await start_economy_batch(
            exam_id="exam-start",
            body=None,
            current_user={"user_id": "teacher-1", "user_type": "admin", "db_name": "skb_test"},
            db=None,
        )

    stored = await db["exampen_exams"].find_one({"exam_id": "exam-start"})
    assert stored["answer_copy_upload_state"] == "closed"
    assert stored["lifecycle_state"] == "uploading"
    assert result.batch_group_id == "econ-start"
    create_group.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_economy_batch_can_confirm_and_close_live_collection():
    from api.v1.exam_orch_async import EconomyBatchStartRequest, start_economy_batch

    db = _fresh_db()
    await db["exampen_exams"].insert_one(_live_economy_exam("exam-live-start"))
    create_group = AsyncMock(
        return_value={
            "batch_group_id": "econ-live-start",
            "exam_id": "exam-live-start",
            "status": "provider_processing",
            "requested_count": 2,
        }
    )
    with (
        patch("api.v1.exam_orch_async._get_tenant_db", return_value=db),
        patch("services.exampen_openai_batch.create_economy_batch_group", new=create_group),
    ):
        result = await start_economy_batch(
            exam_id="exam-live-start",
            body=EconomyBatchStartRequest(close_collection=True),
            current_user={"user_id": "teacher-1", "user_type": "admin", "db_name": "skb_test"},
            db=None,
        )

    stored = await db["exampen_exams"].find_one({"exam_id": "exam-live-start"})
    assert stored["answer_copy_upload_state"] == "closed"
    assert stored["lifecycle_state"] == "uploading"
    assert stored["collection_closed_by"] == "teacher-1"
    assert result.batch_group_id == "econ-live-start"
    create_group.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_economy_batch_requires_explicit_live_collection_confirmation():
    from api.v1.exam_orch_async import start_economy_batch

    db = _fresh_db()
    await db["exampen_exams"].insert_one(_live_economy_exam("exam-live-unconfirmed"))
    with patch("api.v1.exam_orch_async._get_tenant_db", return_value=db):
        with pytest.raises(HTTPException) as exc:
            await start_economy_batch(
                exam_id="exam-live-unconfirmed",
                body=None,
                current_user={"user_id": "teacher-1", "user_type": "admin", "db_name": "skb_test"},
                db=None,
            )

    assert exc.value.status_code == 409
    assert "Confirm" in str(exc.value.detail)
    stored = await db["exampen_exams"].find_one({"exam_id": "exam-live-unconfirmed"})
    assert stored["lifecycle_state"] == "in_progress"
    assert stored["answer_copy_upload_state"] == "open"


@pytest.mark.asyncio
async def test_start_without_waiting_jobs_reopens_a_previously_open_window():
    from api.v1.exam_orch_async import start_economy_batch

    db = _fresh_db()
    await db["exampen_exams"].insert_one(_supported_economy_exam("exam-empty"))
    with patch("api.v1.exam_orch_async._get_tenant_db", return_value=db):
        with pytest.raises(HTTPException) as exc:
            await start_economy_batch(
                exam_id="exam-empty",
                body=None,
                current_user={"user_id": "teacher-1", "user_type": "admin", "db_name": "skb_test"},
                db=None,
            )

    assert exc.value.status_code == 409
    stored = await db["exampen_exams"].find_one({"exam_id": "exam-empty"})
    assert stored["answer_copy_upload_state"] == "open"


@pytest.mark.asyncio
async def test_completed_review_can_reopen_for_a_late_copy_and_followup_batch():
    db = _fresh_db()
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "exam-late-copy",
            "exam_type": "pcr",
            "lifecycle_state": "ready_for_eval",
            "answer_copy_upload_state": "closed",
        }
    )

    reopened = await reopen_answer_copy_uploads(
        db,
        exam_id="exam-late-copy",
        actor_id="teacher-1",
    )

    assert reopened["answer_copy_upload_state"] == "open"
    assert reopened["lifecycle_state"] == "uploading"
    assert answer_copy_upload_is_open(reopened) is True


def test_legacy_upload_window_state_is_derived_from_lifecycle():
    active = {"lifecycle_state": "uploading"}
    draft = {"lifecycle_state": "draft"}
    explicitly_closed = {
        "lifecycle_state": "uploading",
        "answer_copy_upload_state": "closed",
    }

    assert answer_copy_upload_state(active) == "open"
    assert answer_copy_upload_is_open(active) is True
    assert answer_copy_upload_state(draft) == "closed"
    assert answer_copy_upload_is_open(draft) is False
    assert answer_copy_upload_is_open(explicitly_closed) is False


def test_student_availability_reports_teacher_closed_window():
    from api.v1.evalpen_student_submission_async import _student_upload_availability

    allowed, reason = _student_upload_availability(
        {
            "exam_type": "pcr",
            "capture_mode": "camera",
            "student_self_submission_enabled": True,
            "roster": ["student-1"],
            "absent_student_ids": [],
            "lifecycle_state": "uploading",
            "answer_copy_upload_state": "closed",
        },
        "student-1",
    )

    assert allowed is False
    assert reason == "Answer-copy uploads have been closed by your teacher"
