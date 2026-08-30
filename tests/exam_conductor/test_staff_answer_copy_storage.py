from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from starlette.datastructures import Headers, UploadFile


@pytest.mark.asyncio
async def test_staff_pdf_upload_records_s3_original_and_pages_as_one_owned_attempt():
    from api.v1.camera_upload_async import upload_complete_answer_copy
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["skb_test"]
    exam = {
        "exam_id": "exam-staff-pdf",
        "exam_type": "pcr",
        "admin_id": "admin-1",
        "student_submission_max_pages": 10,
    }
    await db["exampen_exams"].insert_one(exam)
    pdf = UploadFile(
        file=io.BytesIO(b"mock-pdf"),
        filename="student-copy.pdf",
        headers=Headers({"content-type": "application/pdf"}),
    )
    original_path = (
        "s3://stoody-test/private/exampen/student-answer-copies/"
        "skb_test/exam-staff-pdf/staff-attempt/original.pdf"
    )
    page_paths = [
        "s3://stoody-test/private/exampen/student-answer-copies/"
        f"skb_test/exam-staff-pdf/staff-attempt/page-{number}.png"
        for number in (1, 2)
    ]
    secure_result = (
        [
            {
                "page_number": number,
                "raw_image_ref": path,
                "storage_path": path,
                "content_hash": str(number) * 64,
                "content_type": "image/png",
                "file_size_bytes": 100 + number,
            }
            for number, path in enumerate(page_paths, start=1)
        ],
        "pdf",
        {
            "storage_path": original_path,
            "filename": "student-copy.pdf",
            "content_type": "application/pdf",
            "size_bytes": 800,
            "sha256": "a" * 64,
        },
        ["C:/temporary-scanner-release/student-copy.pdf"],
        [original_path, *page_paths],
        [],
    )
    ingest_result = SimpleNamespace(
        submission_id="submission-staff-pdf",
        already_existed=False,
    )
    queue_result = {"job_id": "job-staff-pdf", "status": "waiting_for_batch"}

    with (
        patch("api.v1.camera_upload_async._get_tenant_db", new=AsyncMock(return_value=db)),
        patch(
            "api.v1.camera_upload_async._require_camera_upload_context",
            new=AsyncMock(return_value=exam),
        ),
        patch(
            "api.v1.camera_upload_async.reserve_answer_copy_ingest",
            new=AsyncMock(return_value="staff-reservation"),
        ),
        patch(
            "api.v1.camera_upload_async.release_answer_copy_ingest",
            new=AsyncMock(),
        ),
        patch(
            "api.v1.evalpen_student_submission_async._secure_student_copy_pages",
            new=AsyncMock(return_value=secure_result),
        ),
        patch(
            "api.v1.evalpen_student_submission_async._canonical_ingest",
            new=AsyncMock(return_value=ingest_result),
        ),
        patch(
            "api.v1.evalpen_student_submission_async._queue_pcr_processing",
            new=AsyncMock(return_value=queue_result),
        ),
        patch(
            "api.v1.evalpen_student_submission_async._cleanup_released_student_copy_paths",
            new=AsyncMock(return_value=[]),
        ),
    ):
        result = await upload_complete_answer_copy(
            exam_id="exam-staff-pdf",
            student_id="student-1",
            pages=None,
            answer_pdf=pdf,
            confirm_submission=True,
            current_user={
                "user_id": "teacher-1",
                "user_type": "tutor",
                "db_name": "skb_test",
            },
            db=None,
        )

    attempt = await db["exampen_student_copy_uploads"].find_one(
        {"exam_id": "exam-staff-pdf", "student_id": "student-1"}
    )
    assert result.submission_id == "submission-staff-pdf"
    assert attempt["submission_channel"] == "staff_web"
    assert attempt["submitted_by"] == "teacher-1"
    assert attempt["storage_backend"] == "s3"
    assert attempt["storage_handoff_status"] == "complete"
    assert attempt["original_asset"]["storage_path"] == original_path
    assert [page["storage_path"] for page in attempt["pages"]] == page_paths
    assert attempt["local_scan_cleanup_status"] == "complete"
    assert attempt["status"] == "queued"
    assert attempt["processing_job_id"] == "job-staff-pdf"
