from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, patch


def _fresh_db():
    from mongomock_motor import AsyncMongoMockClient

    return AsyncMongoMockClient()["skb_test"]


class _RecordedTask:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def delay(self, db_name: str, job_id: str) -> None:
        self.calls.append((db_name, job_id))


@pytest.mark.asyncio
async def test_processing_job_uses_full_document_result_without_running_ocr():
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        process_pcr_processing_job,
    )

    db = _fresh_db()
    await db[PROCESSING_JOBS_COLLECTION].insert_one(
        {
            "job_id": "pcr-job-SUB-DOC",
            "submission_id": "SUB-DOC",
            "exam_id": "EXAM-DOC",
            "student_id": "STU-DOC",
            "status": "queued",
            "attempts": 0,
        }
    )
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-DOC",
            "exam_id": "EXAM-DOC",
            "student_id": "STU-DOC",
            "source": "camera",
        }
    )
    await db["exampen_exams"].insert_one(
        {"exam_id": "EXAM-DOC", "exam_type": "pcr", "lifecycle_state": "in_progress"}
    )

    class _Gate:
        def __init__(self, _db):
            pass

        async def initialize(self):
            return None

    class _DocumentGrader:
        def __init__(self, _db, _gate):
            pass

        async def grade_submission(self, submission_id: str):
            assert submission_id == "SUB-DOC"
            return SimpleNamespace(
                handled=True,
                status="completed",
                page_count=4,
                response_count=11,
                evaluated_count=11,
                blocked_count=0,
                warning_count=0,
                run_id="DOCGR-1",
                errors=[],
            )

    def _load(name: str):
        if name == "pcr.services":
            return SimpleNamespace(FullDocumentGradingService=_DocumentGrader)
        if name == "llm_gate":
            return SimpleNamespace(LLMGate=_Gate)
        raise AssertionError(f"Unexpected module load: {name}")

    with (
        patch("api.v1._exampen_imports.load_exampen", side_effect=_load),
        patch(
            "api.v1.evalpen_submissions_async._build_submission_service",
            new=AsyncMock(side_effect=AssertionError("OCR fallback must not run")),
        ),
    ):
        result = await process_pcr_processing_job(db, "pcr-job-SUB-DOC")

    assert result["status"] == "completed"
    assert result["processing_path"] == "full_document_visual"
    stored = await db[PROCESSING_JOBS_COLLECTION].find_one(
        {"job_id": "pcr-job-SUB-DOC"}
    )
    assert stored["processing_path"] == "full_document_visual"
    assert stored["evaluation"]["evaluated_count"] == 11
    assert "lease_token" not in stored


@pytest.mark.asyncio
async def test_reprocess_resets_terminal_copy_with_audit_and_requeues(monkeypatch):
    """A teacher retry must be a fresh, auditable mapping run, not a mutation race."""
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        reprocess_processing_job,
    )

    db = _fresh_db()
    jobs = db[PROCESSING_JOBS_COLLECTION]
    await jobs.insert_one(
        {
            "job_id": "pcr-job-SUB-1",
            "submission_id": "SUB-1",
            "exam_id": "EXAM-1",
            "student_id": "STU-1",
            "status": "blocked_for_review",
            "attempts": 2,
            "last_error": "OCR produced a collapsed full-page response",
            "segmentation": {"status": "blocked"},
            "evaluation": {"status": "blocked"},
            "finished_at": datetime.now(timezone.utc),
            "pipeline_version": 1,
        }
    )
    task = _RecordedTask()
    monkeypatch.setitem(
        sys.modules,
        "celery_app",
        SimpleNamespace(process_exampen_pcr_submission=task),
    )
    monkeypatch.setattr(
        "services.exampen_workflow._celery_broker_available",
        lambda: True,
    )

    result = await reprocess_processing_job(
        db,
        db_name="skb_test",
        job_id="pcr-job-SUB-1",
        requested_by="TUT-1",
        reason="Run the full-document answer mapper again",
    )

    assert result["status"] == "queued"
    assert task.calls == [("skb_test", "pcr-job-SUB-1")]

    stored = await jobs.find_one({"job_id": "pcr-job-SUB-1"})
    assert stored["last_error"] is None
    assert stored["segmentation"] == {}
    assert stored["evaluation"] == {}
    assert "finished_at" not in stored
    assert stored["mapping_pipeline_version"] == "full-document-visual-v1"
    assert stored["reprocess_count"] == 1
    assert stored["reprocess_requested_by"] == "TUT-1"
    assert stored["reprocess_history"] == [
        {
            "requested_at": stored["reprocess_requested_at"],
            "requested_by": "TUT-1",
            "reason": "Run the full-document answer mapper again",
            "previous_status": "blocked_for_review",
            "previous_attempts": 2,
            "previous_last_error": "OCR produced a collapsed full-page response",
            "previous_pipeline_version": 1,
            "force_reclaim": False,
        }
    ]


@pytest.mark.asyncio
async def test_teacher_reprocess_rejects_an_active_processing_lease(monkeypatch):
    """A teacher must not start a second mapper while a live worker owns it."""
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        ProcessingJobBusyError,
        reprocess_processing_job,
    )

    db = _fresh_db()
    jobs = db[PROCESSING_JOBS_COLLECTION]
    await jobs.insert_one(
        {
            "job_id": "pcr-job-SUB-2",
            "submission_id": "SUB-2",
            "status": "processing",
            "attempts": 1,
            "updated_at": datetime.now(timezone.utc),
            "lease_token": "worker-one",
            "lease_expires_at": datetime.now(timezone.utc) + timedelta(minutes=20),
        }
    )
    monkeypatch.setattr(
        "services.exampen_workflow._celery_broker_available",
        lambda: True,
    )
    task = _RecordedTask()
    monkeypatch.setitem(
        sys.modules,
        "celery_app",
        SimpleNamespace(process_exampen_pcr_submission=task),
    )

    with pytest.raises(ProcessingJobBusyError, match="active worker"):
        await reprocess_processing_job(
            db,
            db_name="skb_test",
            job_id="pcr-job-SUB-2",
            requested_by="TUT-1",
            reason="Teacher reprocess while worker is live",
        )
    assert task.calls == []
    stored = await jobs.find_one({"job_id": "pcr-job-SUB-2"})
    assert stored["status"] == "processing"
    assert stored["lease_token"] == "worker-one"
    assert "reprocess_count" not in stored


@pytest.mark.asyncio
async def test_teacher_reprocess_reclaims_only_an_expired_processing_lease(monkeypatch):
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        reprocess_processing_job,
    )

    db = _fresh_db()
    jobs = db[PROCESSING_JOBS_COLLECTION]
    await jobs.insert_one(
        {
            "job_id": "pcr-job-SUB-expired",
            "submission_id": "SUB-expired",
            "status": "processing",
            "attempts": 1,
            "updated_at": datetime.now(timezone.utc) - timedelta(minutes=40),
            "lease_token": "dead-worker",
            "lease_expires_at": datetime.now(timezone.utc) - timedelta(minutes=10),
        }
    )
    monkeypatch.setattr(
        "services.exampen_workflow._celery_broker_available",
        lambda: True,
    )
    task = _RecordedTask()
    monkeypatch.setitem(
        sys.modules,
        "celery_app",
        SimpleNamespace(process_exampen_pcr_submission=task),
    )

    result = await reprocess_processing_job(
        db,
        db_name="skb_test",
        job_id="pcr-job-SUB-expired",
        requested_by="TUT-1",
        reason="Recover expired worker",
    )

    assert result["status"] == "queued"
    assert task.calls == [("skb_test", "pcr-job-SUB-expired")]
    stored = await jobs.find_one({"job_id": "pcr-job-SUB-expired"})
    assert "lease_token" not in stored
    assert "lease_expires_at" not in stored
    assert stored["reprocess_history"][0]["force_reclaim"] is True


@pytest.mark.asyncio
async def test_force_dispatch_keeps_a_worker_claimed_job_untouched():
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        dispatch_processing_job,
    )

    db = _fresh_db()
    jobs = db[PROCESSING_JOBS_COLLECTION]
    job = {
        "job_id": "pcr-job-SUB-3",
        "submission_id": "SUB-3",
        "status": "processing",
    }
    await jobs.insert_one(job)

    result = await dispatch_processing_job(
        db,
        db_name="skb_test",
        job=job,
        force=True,
    )

    assert result["status"] == "processing"
    stored = await jobs.find_one({"job_id": "pcr-job-SUB-3"})
    assert stored["status"] == "processing"


@pytest.mark.asyncio
async def test_dispatch_without_redis_does_not_hang_and_schedules_inline(monkeypatch):
    """Reprocess must not sit on Celery Redis retries when Redis is down."""
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        dispatch_processing_job,
    )

    db = _fresh_db()
    jobs = db[PROCESSING_JOBS_COLLECTION]
    job = {
        "job_id": "pcr-job-SUB-4",
        "submission_id": "SUB-4",
        "status": "queued",
    }
    await jobs.insert_one(job)

    monkeypatch.setattr(
        "services.exampen_workflow._celery_broker_available",
        lambda: False,
    )
    scheduled: list[str] = []
    monkeypatch.setattr(
        "services.exampen_workflow._schedule_inline_processing",
        lambda tenant_db, job_id: scheduled.append(job_id),
    )
    # Ensure delay is never called (would hang in real Redis-down environments).
    def _boom(*_args, **_kwargs):
        raise AssertionError("Celery delay must not be called when broker is down")

    monkeypatch.setitem(
        sys.modules,
        "celery_app",
        SimpleNamespace(process_exampen_pcr_submission=SimpleNamespace(delay=_boom)),
    )

    result = await dispatch_processing_job(
        db,
        db_name="skb_test",
        job=job,
        force=True,
    )

    assert result["status"] == "queued"
    assert "broker unavailable" in str(result.get("last_error") or "").lower()
    assert scheduled == ["pcr-job-SUB-4"]


@pytest.mark.asyncio
async def test_worker_error_write_cannot_overwrite_a_newer_lease_owner():
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        mark_processing_job_retryable_error,
    )

    db = _fresh_db()
    jobs = db[PROCESSING_JOBS_COLLECTION]
    await jobs.insert_one(
        {
            "job_id": "pcr-job-fenced",
            "submission_id": "SUB-fenced",
            "status": "processing",
            "lease_token": "new-worker",
            "lease_expires_at": datetime.now(timezone.utc) + timedelta(minutes=20),
        }
    )

    stale_write = await mark_processing_job_retryable_error(
        db,
        "pcr-job-fenced",
        RuntimeError("old worker failed late"),
        expected_lease_token="old-worker",
    )

    assert stale_write is False
    stored = await jobs.find_one({"job_id": "pcr-job-fenced"})
    assert stored["status"] == "processing"
    assert stored["lease_token"] == "new-worker"

    owner_write = await mark_processing_job_retryable_error(
        db,
        "pcr-job-fenced",
        RuntimeError("current worker failed"),
        expected_lease_token="new-worker",
    )

    assert owner_write is True
    stored = await jobs.find_one({"job_id": "pcr-job-fenced"})
    assert stored["status"] == "retryable_error"
    assert "lease_token" not in stored
    assert "lease_expires_at" not in stored
