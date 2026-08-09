from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest
from bson import ObjectId
from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile


def _fresh_db():
    from mongomock_motor import AsyncMongoMockClient

    return AsyncMongoMockClient()["skb_test"]


def _student_user(student_id: str = "student-1") -> Dict[str, Any]:
    return {"user_id": student_id, "user_type": "student", "db_name": "skb_test"}


async def _seed_self_submission_exam(db, **overrides: Any) -> None:
    exam = {
        "exam_id": "PCR-SELF-1",
        "title": "Chemistry PCR",
        "exam_type": "pcr",
        "capture_mode": "camera",
        "student_self_submission_enabled": True,
        "student_submission_max_pages": 5,
        "lifecycle_state": "in_progress",
        "roster": ["student-1"],
        "absent_student_ids": [],
        "admin_id": "admin-1",
    }
    exam.update(overrides)
    await db["exampen_exams"].insert_one(exam)


def _upload_file() -> UploadFile:
    return UploadFile(
        file=io.BytesIO(b"not-used-by-mocked-upload-gateway"),
        filename="answer-page-1.jpg",
        headers=Headers({"content-type": "image/jpeg"}),
    )


@pytest.mark.asyncio
async def test_student_submission_options_only_show_enabled_rostered_pcr_sessions():
    from api.v1.evalpen_student_submission_async import list_answer_copy_options

    db = _fresh_db()
    await _seed_self_submission_exam(db)
    await _seed_self_submission_exam(
        db,
        exam_id="PCR-DISABLED",
        student_self_submission_enabled=False,
    )
    await _seed_self_submission_exam(
        db,
        exam_id="PCR-OTHER-STUDENT",
        roster=["student-2"],
    )

    with patch(
        "api.v1.evalpen_student_submission_async._get_tenant_db",
        return_value=db,
    ):
        result = await list_answer_copy_options(current_user=_student_user(), db=None)

    assert [item.exam_id for item in result.items] == ["PCR-SELF-1"]
    assert result.items[0].can_submit is True
    assert result.items[0].max_pages == 5
    assert result.items[0].submission.status == "not_submitted"


@pytest.mark.asyncio
async def test_student_submission_options_resolve_canonical_prepared_paper_metadata():
    """Legacy sessions must remain discoverable through their finalized paper."""
    from api.v1.evalpen_student_submission_async import list_answer_copy_options

    db = _fresh_db()
    created_at = datetime(2026, 8, 9, 16, 38, tzinfo=timezone.utc)
    updated_at = created_at + timedelta(minutes=2)
    await db["documents"].insert_one(
        {
            "document_id": "Phy1",
            "title": "Phy 1",
            "subject": "Mathematics",
            "content_category_id": "weekly-1",
            "content_category_name": "Weekly 1",
            "file_path": "uploads/phy1.pdf",
            "exam_finalized_at": created_at,
        }
    )
    await _seed_self_submission_exam(
        db,
        exam_id="PHY1-SESSION",
        title="Phy 1 session",
        prepared_document_id="Phy1",
        # Sessions created before the metadata contract was introduced do not
        # have these fields.  The prepared document is their source of truth.
        subject=None,
        code=None,
        content_category_id="stale-category",
        content_category_name="Stale category",
        created_at=created_at,
        updated_at=updated_at,
    )

    with patch(
        "api.v1.evalpen_student_submission_async._get_tenant_db",
        return_value=db,
    ):
        result = await list_answer_copy_options(current_user=_student_user(), db=None)

    assert len(result.items) == 1
    option = result.items[0]
    assert option.exam_id == "PHY1-SESSION"
    assert option.title == "Phy 1 session"
    assert option.paper_title == "Phy 1"
    assert option.subject == "Mathematics"
    assert option.code == "Phy1"
    assert option.content_category_id == "weekly-1"
    assert option.content_category_name == "Weekly 1"
    assert option.question_paper_available is True
    assert option.answer_copy_available is False
    assert option.created_at == created_at.isoformat()
    assert option.updated_at == updated_at.isoformat()
    assert option.can_submit is True


@pytest.mark.asyncio
async def test_student_submission_options_resolve_student_db_roster_id_for_existing_login_session():
    """A token's account id must resolve to the Student DB id used by rosters."""
    from api.v1.evalpen_student_submission_async import list_answer_copy_options

    db = _fresh_db()
    account_id = ObjectId()
    await db["students"].insert_one(
        {
            "_id": account_id,
            "student_id": "STU-harsh-1",
            "username": "harsh",
            "username_lower": "harsh",
        }
    )
    await _seed_self_submission_exam(db, roster=["STU-harsh-1"])
    legacy_login_session = {
        "user_id": str(account_id),
        "user_type": "student",
        "username": "harsh",
        "db_name": "skb_test",
    }

    with patch(
        "api.v1.evalpen_student_submission_async._get_tenant_db",
        return_value=db,
    ):
        result = await list_answer_copy_options(current_user=legacy_login_session, db=None)

    assert [item.exam_id for item in result.items] == ["PCR-SELF-1"]
    assert result.items[0].can_submit is True


@pytest.mark.asyncio
async def test_student_copy_submission_derives_identity_and_queues_existing_pcr_pipeline():
    from api.v1.evalpen_student_submission_async import submit_answer_copy

    db = _fresh_db()
    await _seed_self_submission_exam(db)
    clean_page = SimpleNamespace(
        released_storage_path="/private/answer-page-1.jpg",
        original_filename="answer-page-1.jpg",
        upload_id="upload-1",
        sha256="a" * 64,
        content_type="image/jpeg",
        size_bytes=1200,
        bytes=b"clean-jpeg-page",
    )
    ingest_result = SimpleNamespace(submission_id="submission-1")
    queue_result = {"job_id": "job-1", "status": "queued"}

    with (
        patch("api.v1.evalpen_student_submission_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_student_submission_async.secure_upload_many",
            new=AsyncMock(return_value=[clean_page]),
        ) as upload_many,
        patch(
            "api.v1.evalpen_student_submission_async.upload_private_object",
            new=AsyncMock(return_value="s3://stoody-test/private/exampen/student-answer-copies/page-1.jpg"),
        ) as upload_private,
        patch(
            "api.v1.evalpen_student_submission_async._canonical_ingest",
            new=AsyncMock(return_value=ingest_result),
        ) as canonical_ingest,
        patch(
            "api.v1.evalpen_student_submission_async._queue_pcr_processing",
            new=AsyncMock(return_value=queue_result),
        ) as queue_processing,
    ):
        ack = await submit_answer_copy(
            exam_id="PCR-SELF-1",
            pages=[_upload_file()],
            answer_pdf=None,
            confirm_submission=True,
            current_user=_student_user(),
            db=None,
        )

    assert ack.submission_id == "submission-1"
    assert ack.page_count == 1
    assert ack.processing_status == "queued"
    assert upload_many.await_args.kwargs["policy_id"] == "student_answer_copy_image"
    assert upload_many.await_args.kwargs["include_bytes"] is True
    assert upload_private.await_count == 1
    assert canonical_ingest.await_args.kwargs["student_id"] == "student-1"
    assert canonical_ingest.await_args.kwargs["admin_id"] == "admin-1"
    assert queue_processing.await_args.kwargs["student_id"] == "student-1"

    attempt = await db["exampen_student_copy_uploads"].find_one({"exam_id": "PCR-SELF-1"})
    assert attempt["submission_channel"] == "student_web"
    assert attempt["status"] == "queued"
    assert attempt["submission_id"] == "submission-1"
    assert attempt["storage_backend"] == "s3"
    assert attempt["pages"][0]["storage_path"].startswith("s3://")


@pytest.mark.asyncio
async def test_student_copy_submission_writes_the_student_db_id_not_the_login_account_id():
    from api.v1.evalpen_student_submission_async import submit_answer_copy

    db = _fresh_db()
    account_id = ObjectId()
    await db["students"].insert_one(
        {
            "_id": account_id,
            "student_id": "STU-harsh-1",
            "username": "harsh",
            "username_lower": "harsh",
        }
    )
    await _seed_self_submission_exam(db, roster=["STU-harsh-1"])
    clean_page = SimpleNamespace(
        released_storage_path="/private/answer-page-1.jpg",
        original_filename="answer-page-1.jpg",
        upload_id="upload-1",
        sha256="a" * 64,
        content_type="image/jpeg",
        size_bytes=1200,
        bytes=b"clean-jpeg-page",
    )
    ingest_result = SimpleNamespace(submission_id="submission-1")
    legacy_login_session = {
        "user_id": str(account_id),
        "user_type": "student",
        "username": "harsh",
        "db_name": "skb_test",
    }

    with (
        patch("api.v1.evalpen_student_submission_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_student_submission_async.secure_upload_many",
            new=AsyncMock(return_value=[clean_page]),
        ),
        patch(
            "api.v1.evalpen_student_submission_async.upload_private_object",
            new=AsyncMock(return_value="s3://stoody-test/private/exampen/student-answer-copies/page-1.jpg"),
        ),
        patch(
            "api.v1.evalpen_student_submission_async._canonical_ingest",
            new=AsyncMock(return_value=ingest_result),
        ) as canonical_ingest,
        patch(
            "api.v1.evalpen_student_submission_async._queue_pcr_processing",
            new=AsyncMock(return_value={"job_id": "job-1", "status": "queued"}),
        ),
    ):
        await submit_answer_copy(
            exam_id="PCR-SELF-1",
            pages=[_upload_file()],
            answer_pdf=None,
            confirm_submission=True,
            current_user=legacy_login_session,
            db=None,
        )

    assert canonical_ingest.await_args.kwargs["student_id"] == "STU-harsh-1"
    attempt = await db["exampen_student_copy_uploads"].find_one({"exam_id": "PCR-SELF-1"})
    assert attempt["student_id"] == "STU-harsh-1"


@pytest.mark.asyncio
async def test_student_cannot_submit_for_an_exam_outside_their_roster():
    from api.v1.evalpen_student_submission_async import submit_answer_copy

    db = _fresh_db()
    await _seed_self_submission_exam(db, roster=["student-2"])

    with patch(
        "api.v1.evalpen_student_submission_async._get_tenant_db",
        return_value=db,
    ):
        with pytest.raises(HTTPException) as exc:
            await submit_answer_copy(
                exam_id="PCR-SELF-1",
                pages=[_upload_file()],
                answer_pdf=None,
                confirm_submission=True,
                current_user=_student_user("student-1"),
                db=None,
            )

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_student_copy_status_is_not_exposed_for_sessions_without_the_opt_in_channel():
    from api.v1.evalpen_student_submission_async import get_answer_copy_status

    db = _fresh_db()
    await _seed_self_submission_exam(db, student_self_submission_enabled=False)

    with patch(
        "api.v1.evalpen_student_submission_async._get_tenant_db",
        return_value=db,
    ):
        with pytest.raises(HTTPException) as exc:
            await get_answer_copy_status(
                exam_id="PCR-SELF-1",
                current_user=_student_user(),
                db=None,
            )

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_existing_final_submission_blocks_another_student_copy():
    from api.v1.evalpen_student_submission_async import list_answer_copy_options, submit_answer_copy

    db = _fresh_db()
    await _seed_self_submission_exam(db)
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "already-submitted",
            "exam_id": "PCR-SELF-1",
            "student_id": "student-1",
            "page_count": 2,
            "segmentation_status": "pending",
        }
    )
    await db["evalpen_answer_pages"].insert_one(
        {
            "submission_id": "already-submitted",
            "exam_id": "PCR-SELF-1",
            "student_id": "student-1",
            "page_number": 1,
            "raw_image_ref": "/private/already-submitted-page.jpg",
        }
    )

    with patch(
        "api.v1.evalpen_student_submission_async._get_tenant_db",
        return_value=db,
    ):
        options = await list_answer_copy_options(current_user=_student_user(), db=None)
        with pytest.raises(HTTPException) as exc:
            await submit_answer_copy(
                exam_id="PCR-SELF-1",
                pages=[_upload_file()],
                answer_pdf=None,
                confirm_submission=True,
                current_user=_student_user(),
                db=None,
            )

    assert options.items[0].can_submit is False
    assert options.items[0].submission.submission_id == "already-submitted"
    assert options.items[0].answer_copy_available is True
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_answer_copy_download_reads_canonical_pages_uploaded_by_staff(tmp_path):
    from api.v1.evalpen_student_submission_async import download_answer_copy

    db = _fresh_db()
    await _seed_self_submission_exam(db, student_self_submission_enabled=False)
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "staff-submission-1",
            "exam_id": "PCR-SELF-1",
            "student_id": "student-1",
        }
    )
    page_path = tmp_path / "staff-answer-page.png"
    page_path.write_bytes(b"canonical-staff-answer-page")
    await db["evalpen_answer_pages"].insert_one(
        {
            "submission_id": "staff-submission-1",
            "exam_id": "PCR-SELF-1",
            "student_id": "student-1",
            "page_number": 1,
            "raw_image_ref": str(page_path),
            "content_type": "image/png",
        }
    )

    with patch(
        "api.v1.evalpen_student_submission_async._get_tenant_db",
        return_value=db,
    ):
        response = await download_answer_copy(
            exam_id="PCR-SELF-1",
            current_user=_student_user(),
            db=None,
        )

    assert response.body == b"canonical-staff-answer-page"
    assert response.media_type == "image/png"
    assert "PCR-SELF-1-answer-copy.png" in response.headers["content-disposition"]


@pytest.mark.asyncio
async def test_single_final_copy_reservation_blocks_parallel_browser_tabs_but_allows_safe_upload_retry():
    from api.v1.evalpen_student_submission_async import (
        _ensure_student_copy_indexes,
        _mark_attempt_upload_failed,
        _reserve_student_copy_attempt,
    )

    db = _fresh_db()
    collection = db["exampen_student_copy_uploads"]
    await _ensure_student_copy_indexes(collection)

    first_attempt = await _reserve_student_copy_attempt(
        collection,
        attempt_id="attempt-first",
        exam_id="PCR-SELF-1",
        student_id="student-1",
        admin_id="admin-1",
    )
    assert first_attempt == "attempt-first"

    with pytest.raises(HTTPException) as exc:
        await _reserve_student_copy_attempt(
            collection,
            attempt_id="attempt-second",
            exam_id="PCR-SELF-1",
            student_id="student-1",
            admin_id="admin-1",
        )
    assert exc.value.status_code == 409

    await _mark_attempt_upload_failed(
        collection,
        attempt_id="attempt-first",
        reason="temporary scanner failure",
    )
    retry_attempt = await _reserve_student_copy_attempt(
        collection,
        attempt_id="attempt-retry",
        exam_id="PCR-SELF-1",
        student_id="student-1",
        admin_id="admin-1",
    )
    assert retry_attempt == "attempt-first"
    reservation = await collection.find_one({"attempt_id": "attempt-first"})
    assert reservation["status"] == "receiving"
    assert reservation["upload_attempt_count"] == 2


@pytest.mark.asyncio
async def test_crashed_receiving_reservation_expires_and_reuses_the_same_attempt():
    from api.v1.evalpen_student_submission_async import (
        _ensure_student_copy_indexes,
        _reserve_student_copy_attempt,
        list_answer_copy_options,
    )

    db = _fresh_db()
    await _seed_self_submission_exam(db)
    collection = db["exampen_student_copy_uploads"]
    await _ensure_student_copy_indexes(collection)
    await _reserve_student_copy_attempt(
        collection,
        attempt_id="attempt-crashed",
        exam_id="PCR-SELF-1",
        student_id="student-1",
        admin_id="admin-1",
    )
    expired_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    await collection.update_one(
        {"attempt_id": "attempt-crashed"},
        {"$set": {"lease_expires_at": expired_at}},
    )

    with patch(
        "api.v1.evalpen_student_submission_async._get_tenant_db",
        return_value=db,
    ):
        options = await list_answer_copy_options(current_user=_student_user(), db=None)
    assert options.items[0].can_submit is True

    retry_attempt = await _reserve_student_copy_attempt(
        collection,
        attempt_id="attempt-new-client-request",
        exam_id="PCR-SELF-1",
        student_id="student-1",
        admin_id="admin-1",
    )
    assert retry_attempt == "attempt-crashed"
    reservation = await collection.find_one({"attempt_id": "attempt-crashed"})
    assert reservation["status"] == "receiving"
    assert reservation["upload_attempt_count"] == 2
    assert reservation["lease_expires_at"] > datetime.now(timezone.utc).replace(tzinfo=None)


@pytest.mark.asyncio
async def test_student_copy_does_not_claim_a_canonical_submission_created_concurrently():
    from api.v1.evalpen_student_submission_async import submit_answer_copy

    db = _fresh_db()
    await _seed_self_submission_exam(db)
    clean_page = SimpleNamespace(
        released_storage_path="/private/answer-page-1.jpg",
        original_filename="answer-page-1.jpg",
        upload_id="upload-1",
        sha256="a" * 64,
        content_type="image/jpeg",
        size_bytes=1200,
        bytes=b"clean-jpeg-page",
    )
    concurrent_result = SimpleNamespace(submission_id="staff-submission-1", already_existed=True)

    with (
        patch("api.v1.evalpen_student_submission_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_student_submission_async.secure_upload_many",
            new=AsyncMock(return_value=[clean_page]),
        ),
        patch(
            "api.v1.evalpen_student_submission_async.upload_private_object",
            new=AsyncMock(return_value="s3://stoody-test/private/exampen/student-answer-copies/page-1.jpg"),
        ),
        patch(
            "api.v1.evalpen_student_submission_async._canonical_ingest",
            new=AsyncMock(return_value=concurrent_result),
        ),
    ):
        with pytest.raises(HTTPException) as exc:
            await submit_answer_copy(
                exam_id="PCR-SELF-1",
                pages=[_upload_file()],
                answer_pdf=None,
                confirm_submission=True,
                current_user=_student_user(),
                db=None,
            )

    assert exc.value.status_code == 409
    attempt = await db["exampen_student_copy_uploads"].find_one({"exam_id": "PCR-SELF-1"})
    assert attempt["status"] == "superseded"
    assert attempt["submission_id"] == "staff-submission-1"


def test_student_self_submission_requires_pcr_camera_or_hybrid_capture():
    from api.v1.exam_orch_async import _normalize_student_self_submission_config

    assert _normalize_student_self_submission_config(
        enabled=True,
        max_pages=8,
        exam_type="pcr",
        capture_mode="hybrid",
    ) == (True, 8)

    with pytest.raises(HTTPException) as pen_exc:
        _normalize_student_self_submission_config(
            enabled=True,
            max_pages=8,
            exam_type="pcr",
            capture_mode="pen",
        )
    assert pen_exc.value.status_code == 422

    with pytest.raises(HTTPException) as dcr_exc:
        _normalize_student_self_submission_config(
            enabled=True,
            max_pages=8,
            exam_type="dcr",
            capture_mode="camera",
        )
    assert dcr_exc.value.status_code == 422


@pytest.mark.asyncio
async def test_student_copy_uses_the_existing_canonical_pcr_ingest_records():
    from api.v1.evalpen_student_submission_async import _canonical_ingest

    db = _fresh_db()
    result = await _canonical_ingest(
        db,
        exam_id="PCR-CANONICAL-1",
        student_id="student-1",
        admin_id="admin-1",
        pages=[
            {
                "page_number": 1,
                "raw_strokes": None,
                "raw_image_ref": "/private/student-answer-page-1.png",
            }
        ],
    )

    submission = await db["evalpen_submissions"].find_one({"submission_id": result.submission_id})
    answer_page = await db["evalpen_answer_pages"].find_one({"submission_id": result.submission_id})
    assert submission["exam_id"] == "PCR-CANONICAL-1"
    assert submission["student_id"] == "student-1"
    assert submission["source"] == "camera"
    assert answer_page["raw_image_ref"] == "/private/student-answer-page-1.png"
