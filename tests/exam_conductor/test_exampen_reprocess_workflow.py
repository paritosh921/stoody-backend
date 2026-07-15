from __future__ import annotations

import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest


def _fresh_db():
    from mongomock_motor import AsyncMongoMockClient

    return AsyncMongoMockClient()["skb_test"]


class _RecordedTask:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def delay(self, db_name: str, job_id: str) -> None:
        self.calls.append((db_name, job_id))


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
    assert stored["mapping_pipeline_version"] == "document-answer-mapping-v1"
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
async def test_teacher_reprocess_always_reclaims_processing_job(monkeypatch):
    """Teacher Reprocess must never 409 on a stuck processing lease."""
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
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
        job_id="pcr-job-SUB-2",
        requested_by="TUT-1",
        reason="Teacher reprocess after stuck job",
    )
    assert result["status"] == "queued"
    assert task.calls == [("skb_test", "pcr-job-SUB-2")]
    stored = await jobs.find_one({"job_id": "pcr-job-SUB-2"})
    assert stored["reprocess_count"] == 1
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
