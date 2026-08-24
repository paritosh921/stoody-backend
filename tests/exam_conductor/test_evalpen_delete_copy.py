from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException


def _fresh_db():
    from mongomock_motor import AsyncMongoMockClient

    return AsyncMongoMockClient()["skb_test"]


def _submission(*, publication_status: str = "draft"):
    return {
        "submission_id": "SUB-WRONG-STUDENT",
        "exam_id": "EXAM-1",
        "student_id": "STUDENT-A",
        "admin_id": "ADMIN-1",
        "source": "camera",
        "publication_status": publication_status,
        "segmentation_status": "complete",
        "total_score": 17.0,
        "total_max_score": 25.0,
    }


@pytest.mark.asyncio
async def test_delete_copy_clears_complete_active_lifecycle_and_keeps_exam_paper():
    from services.exampen_submission_deletion import delete_submission_copy

    db = _fresh_db()
    submission = _submission()
    await db["evalpen_submissions"].insert_one(submission)
    await db["evalpen_answer_pages"].insert_one(
        {
            "page_id": "PAGE-1",
            "submission_id": submission["submission_id"],
            "page_number": 1,
            "raw_image_ref": "s3://test/private/exampen/student-answer-copies/page-1.png",
        }
    )
    await db["evalpen_detected_responses"].insert_one(
        {"response_id": "RESP-1", "submission_id": submission["submission_id"]}
    )
    await db["evalpen_evaluations"].insert_one(
        {
            "evaluation_id": "EVAL-1",
            # Legacy response evaluations were linked through response_id and
            # did not persist submission_id.
            "response_id": "RESP-1",
            "question_id": "Q-1",
            "marks_awarded": 1.0,
            "max_marks": 1.0,
        }
    )
    await db["evalpen_document_grading_runs"].insert_one(
        {"run_id": "RUN-1", "submission_id": submission["submission_id"]}
    )
    await db["evalpen_objective_grading_runs"].insert_one(
        {"run_id": "OBJ-1", "submission_id": submission["submission_id"]}
    )
    await db["exampen_dcr_results"].insert_one(
        {
            "result_id": "DCR-1",
            "exam_id": submission["exam_id"],
            "student_id": submission["student_id"],
            "question_id": "Q-1",
            "score": 1.0,
        }
    )
    await db["evalpen_recheck_requests"].insert_one(
        {"request_id": "RECHECK-1", "submission_id": submission["submission_id"]}
    )
    await db["exampen_processing_jobs"].insert_one(
        {
            "job_id": "JOB-1",
            "submission_id": submission["submission_id"],
            "status": "completed",
        }
    )
    await db["exampen_student_copy_uploads"].insert_one(
        {
            "attempt_id": "ATTEMPT-1",
            "exam_id": submission["exam_id"],
            "student_id": submission["student_id"],
            "status": "submitted",
        }
    )
    await db["exampen_camera_uploads"].insert_one(
        {
            "artifact_id": "CAM-1",
            "submission_id": submission["submission_id"],
            "exam_id": submission["exam_id"],
            "student_id": submission["student_id"],
        }
    )
    await db["exampen_exams"].insert_one(
        {"exam_id": submission["exam_id"], "title": "Vectors"}
    )
    await db["evalpen_questions"].insert_one(
        {"exam_id": submission["exam_id"], "question_id": "Q-1"}
    )

    with patch(
        "services.exampen_submission_deletion._cleanup_storage_paths",
        new=AsyncMock(return_value={"deleted": ["page-1"], "failed": [], "skipped": []}),
    ):
        result = await delete_submission_copy(
            db,
            submission,
            actor_id="TUTOR-1",
            actor_role="tutor",
            reason_code="wrong_student",
            reason_note="Uploaded under Student A instead of Student B",
        )

    assert result["status"] == "deleted"
    for collection_name in (
        "evalpen_submissions",
        "evalpen_answer_pages",
        "evalpen_detected_responses",
        "evalpen_evaluations",
        "evalpen_document_grading_runs",
        "evalpen_objective_grading_runs",
        "exampen_dcr_results",
        "evalpen_recheck_requests",
        "exampen_processing_jobs",
        "exampen_student_copy_uploads",
        "exampen_camera_uploads",
    ):
        assert await db[collection_name].count_documents({}) == 0

    assert await db["exampen_exams"].count_documents({"exam_id": "EXAM-1"}) == 1
    assert await db["evalpen_questions"].count_documents({"question_id": "Q-1"}) == 1

    audit = await db["evalpen_submission_deletion_audit"].find_one(
        {"deletion_id": result["deletion_id"]}
    )
    assert audit is not None
    assert audit["status"] == "completed"
    assert audit["reason_code"] == "wrong_student"
    assert audit["submission_snapshot"]["total_score"] == 17.0
    assert audit["evaluation_snapshot"][0]["marks_awarded"] == 1.0


@pytest.mark.asyncio
async def test_delete_copy_cancels_queued_job_before_removing_submission():
    from services.exampen_submission_deletion import delete_submission_copy

    db = _fresh_db()
    submission = _submission()
    await db["evalpen_submissions"].insert_one(submission)
    await db["exampen_processing_jobs"].insert_one(
        {
            "job_id": "JOB-QUEUED",
            "submission_id": submission["submission_id"],
            "status": "queued_pipeline_v7",
        }
    )

    with patch(
        "services.exampen_submission_deletion._cleanup_storage_paths",
        new=AsyncMock(return_value={"deleted": [], "failed": [], "skipped": []}),
    ):
        result = await delete_submission_copy(
            db,
            submission,
            actor_id="ADMIN-1",
            actor_role="admin",
            reason_code="wrong_file",
        )

    audit = await db["evalpen_submission_deletion_audit"].find_one(
        {"deletion_id": result["deletion_id"]}
    )
    assert audit["processing_job_snapshot"]["status"] == "queued_pipeline_v7"
    assert await db["exampen_processing_jobs"].count_documents({}) == 0
    assert await db["evalpen_submissions"].count_documents({}) == 0


@pytest.mark.asyncio
async def test_delete_copy_refuses_active_processing_and_preserves_copy():
    from services.exampen_submission_deletion import (
        SubmissionCopyBusyError,
        delete_submission_copy,
    )

    db = _fresh_db()
    submission = _submission()
    await db["evalpen_submissions"].insert_one(submission)
    await db["evalpen_evaluations"].insert_one(
        {"evaluation_id": "EVAL-1", "submission_id": submission["submission_id"]}
    )
    await db["exampen_processing_jobs"].insert_one(
        {
            "job_id": "JOB-ACTIVE",
            "submission_id": submission["submission_id"],
            "status": "processing",
            "lease_token": "worker-token",
        }
    )

    with pytest.raises(SubmissionCopyBusyError, match="currently being checked"):
        await delete_submission_copy(
            db,
            submission,
            actor_id="TUTOR-1",
            actor_role="tutor",
            reason_code="wrong_student",
        )

    assert await db["evalpen_submissions"].count_documents({}) == 1
    assert await db["evalpen_evaluations"].count_documents({}) == 1
    assert await db["exampen_processing_jobs"].count_documents({}) == 1
    assert await db["evalpen_submission_deletion_audit"].count_documents({}) == 0


@pytest.mark.asyncio
async def test_published_copy_requires_explicit_withdrawal_confirmation():
    from api.v1.evalpen_submissions_async import DeleteCopyRequest, delete_copy

    db = _fresh_db()
    submission = _submission(publication_status="published")
    await db["evalpen_submissions"].insert_one(submission)
    await db["exampen_exams"].insert_one(
        {
            "exam_id": submission["exam_id"],
            "teacher_ids": ["TUTOR-1"],
            "created_by_tutor_id": "TUTOR-1",
        }
    )
    database_manager = SimpleNamespace(get_tenant_db=AsyncMock(return_value=db))

    with pytest.raises(HTTPException) as exc_info:
        await delete_copy(
            submission_id=submission["submission_id"],
            body=DeleteCopyRequest(reason="wrong_student"),
            current_user={
                "user_id": "USER-TUTOR-1",
                "user_type": "tutor",
                "tutor_id": "TUTOR-1",
                "db_name": "skb_test",
            },
            db=database_manager,
        )

    assert exc_info.value.status_code == 409
    assert "published" in str(exc_info.value.detail).lower()
    assert await db["evalpen_submissions"].count_documents({}) == 1


@pytest.mark.asyncio
async def test_delete_copy_route_applies_lease_and_returns_student_to_missing():
    from api.v1.evalpen_submissions_async import DeleteCopyRequest, delete_copy

    db = _fresh_db()
    submission = _submission()
    await db["evalpen_submissions"].insert_one(submission)
    await db["exampen_processing_jobs"].insert_one(
        {
            "job_id": "JOB-COMPLETE",
            "submission_id": submission["submission_id"],
            "status": "completed",
        }
    )
    database_manager = SimpleNamespace(get_tenant_db=AsyncMock(return_value=db))

    with patch(
        "services.exampen_submission_deletion._cleanup_storage_paths",
        new=AsyncMock(return_value={"deleted": [], "failed": [], "skipped": []}),
    ):
        result = await delete_copy(
            submission_id=submission["submission_id"],
            body=DeleteCopyRequest(
                reason="wrong_file",
                note="The pages belong to another student",
                confirm_published=True,
            ),
            current_user={
                "user_id": "ADMIN-1",
                "user_type": "admin",
                "db_name": "skb_test",
            },
            db=database_manager,
        )

    assert result.status == "deleted"
    assert result.student_id == submission["student_id"]
    assert await db["evalpen_submissions"].count_documents({}) == 0
    assert await db["exampen_processing_jobs"].count_documents({}) == 0
    audit = await db["evalpen_submission_deletion_audit"].find_one(
        {"deletion_id": result.deletion_id}
    )
    assert audit["actor_id"] == "ADMIN-1"
    assert audit["reason_code"] == "wrong_file"
