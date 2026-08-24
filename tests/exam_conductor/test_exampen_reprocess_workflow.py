from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, patch


def _fresh_db():
    from mongomock_motor import AsyncMongoMockClient

    return AsyncMongoMockClient()["skb_test"]


async def _seed_supported_subjective_exam(db, exam_id: str) -> None:
    await db["exampen_exams"].insert_one(
        {
            "exam_id": exam_id,
            "exam_type": "pcr",
            "pcr_grading_contract": {
                "prompt_version": "pcr-full-document-visual-v16",
                "pipeline_version": 7,
                "mapping_pipeline_version": "whole-copy-rubric-v7",
                "required_processing_path": "full_document_visual",
            },
        }
    )


async def _seed_submission(db, submission_id: str, exam_id: str, **extra) -> None:
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": submission_id,
            "exam_id": exam_id,
            "student_id": f"STU-{submission_id}",
            "source": "camera",
            "publication_status": "pending",
            **extra,
        }
    )


class _RecordedTask:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def delay(self, db_name: str, job_id: str, pipeline_version: int) -> None:
        self.calls.append((db_name, job_id, pipeline_version))


@pytest.mark.asyncio
async def test_processing_job_uses_objective_lane_without_subjective_grader():
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        process_pcr_processing_job,
    )

    db = _fresh_db()
    await db[PROCESSING_JOBS_COLLECTION].insert_one(
        {
            "job_id": "pcr-job-SUB-OBJ",
            "submission_id": "SUB-OBJ",
            "exam_id": "EXAM-OBJ",
            "student_id": "STU-OBJ",
            "status": "queued",
            "attempts": 0,
        }
    )
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-OBJ",
            "exam_id": "EXAM-OBJ",
            "student_id": "STU-OBJ",
            "source": "camera",
        }
    )
    await db["exampen_exams"].insert_one(
        {"exam_id": "EXAM-OBJ", "exam_type": "pcr", "lifecycle_state": "in_progress"}
    )

    class _Gate:
        def __init__(self, _db):
            pass

        async def initialize(self):
            return None

    class _ObjectiveGrader:
        def __init__(self, _db, _gate):
            pass

        async def grade_submission(self, submission_id: str):
            assert submission_id == "SUB-OBJ"
            return SimpleNamespace(
                handled=True,
                status="completed",
                page_count=1,
                response_count=75,
                evaluated_count=75,
                blocked_count=0,
                warning_count=0,
                run_id="OBJGR-1",
                errors=[],
                review_state="ready",
                document_review_required=False,
                review_reasons=[],
                processing_path="objective_answer_sheet",
            )

    class _SubjectiveGrader:
        def __init__(self, _db, _gate):
            raise AssertionError(
                "A handled Objective paper must not construct the Subjective grader"
            )

    def _load(name: str):
        if name == "pcr.services":
            return SimpleNamespace(
                ObjectiveAnswerSheetGradingService=_ObjectiveGrader,
                FullDocumentGradingService=_SubjectiveGrader,
            )
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
        result = await process_pcr_processing_job(db, "pcr-job-SUB-OBJ")

    assert result["status"] == "completed"
    assert result["processing_path"] == "objective_answer_sheet"
    stored = await db[PROCESSING_JOBS_COLLECTION].find_one(
        {"job_id": "pcr-job-SUB-OBJ"}
    )
    assert stored["processing_path"] == "objective_answer_sheet"
    assert stored["evaluation"]["evaluated_count"] == 75


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
                review_state="ready",
                document_review_required=False,
                review_reasons=[],
            )

    class _ObjectiveGrader:
        def __init__(self, _db, _gate):
            pass

        async def grade_submission(self, _submission_id: str):
            return SimpleNamespace(
                handled=False,
                skipped_reason="paper is subjective",
            )

    def _load(name: str):
        if name == "pcr.services":
            return SimpleNamespace(
                ObjectiveAnswerSheetGradingService=_ObjectiveGrader,
                FullDocumentGradingService=_DocumentGrader,
            )
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
    assert stored["review"]["state"] == "ready"
    assert "lease_token" not in stored


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "visual_contract_version",
    [
        "canonical-full-document-visual-v1",
        "canonical-full-document-visual-v2",
    ],
)
async def test_visual_contract_cannot_silently_fall_back_to_ocr_mapping(
    visual_contract_version,
):
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        process_pcr_processing_job,
    )

    db = _fresh_db()
    await db[PROCESSING_JOBS_COLLECTION].insert_one(
        {
            "job_id": "pcr-job-SUB-VISUAL",
            "submission_id": "SUB-VISUAL",
            "exam_id": "EXAM-VISUAL",
            "student_id": "STU-VISUAL",
            "status": "queued",
            "attempts": 0,
        }
    )
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-VISUAL",
            "exam_id": "EXAM-VISUAL",
            "student_id": "STU-VISUAL",
            "source": "camera",
        }
    )
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "EXAM-VISUAL",
            "exam_type": "pcr",
            "paper_version_id": "PV-VISUAL",
        }
    )
    await db["exampen_paper_versions"].insert_one(
        {
            "paper_version_id": "PV-VISUAL",
            "paper_context": {
                "version": visual_contract_version,
                "mode": "full_document_visual",
                "ready": True,
            },
        }
    )

    class _Gate:
        def __init__(self, _db):
            pass

        async def initialize(self):
            return None

    class _FullDocError(RuntimeError):
        pass

    class _ObjectiveGrader:
        def __init__(self, _db, _gate):
            pass

        async def grade_submission(self, _submission_id: str):
            return SimpleNamespace(handled=False)

    class _DocumentGrader:
        def __init__(self, _db, _gate):
            pass

        async def grade_submission(self, _submission_id: str):
            return SimpleNamespace(
                handled=False,
                skipped_reason="canonical paper asset temporarily unavailable",
            )

    def _load(name: str):
        if name == "pcr.services":
            return SimpleNamespace(
                ObjectiveAnswerSheetGradingService=_ObjectiveGrader,
                FullDocumentGradingService=_DocumentGrader,
                FullDocumentGradingError=_FullDocError,
            )
        if name == "llm_gate":
            return SimpleNamespace(LLMGate=_Gate)
        raise AssertionError(f"Unexpected module load: {name}")

    with (
        patch("api.v1._exampen_imports.load_exampen", side_effect=_load),
        patch(
            "api.v1.evalpen_submissions_async._build_submission_service",
            new=AsyncMock(side_effect=AssertionError("OCR fallback must not run")),
        ),
        pytest.raises(_FullDocError, match="required"),
    ):
        await process_pcr_processing_job(db, "pcr-job-SUB-VISUAL")


@pytest.mark.asyncio
async def test_worker_pipeline_contract_rejects_an_old_job_revision():
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        process_pcr_processing_job,
    )

    db = _fresh_db()
    await db[PROCESSING_JOBS_COLLECTION].insert_one(
        {
            "job_id": "pcr-job-OLD",
            "submission_id": "SUB-OLD",
            "status": "queued",
            "pipeline_version": 1,
            "attempts": 0,
        }
    )

    result = await process_pcr_processing_job(
        db,
        "pcr-job-OLD",
        required_pipeline_version=2,
    )

    assert result["claimed"] is False
    stored = await db[PROCESSING_JOBS_COLLECTION].find_one(
        {"job_id": "pcr-job-OLD"}
    )
    assert stored["status"] == "queued"
    assert stored["attempts"] == 0


@pytest.mark.asyncio
async def test_v3_capability_queue_cannot_be_claimed_by_v2_worker():
    from services.exampen_workflow import (
        CAPABILITY_QUEUED_JOB_STATUS,
        PROCESSING_JOBS_COLLECTION,
        _claim_job,
    )

    db = _fresh_db()
    jobs = db[PROCESSING_JOBS_COLLECTION]
    await jobs.insert_one(
        {
            "job_id": "pcr-job-V3-FENCE",
            "submission_id": "SUB-V3-FENCE",
            "status": CAPABILITY_QUEUED_JOB_STATUS,
            "pipeline_version": 3,
            "attempts": 0,
        }
    )

    stale_claim = await _claim_job(
        db,
        "pcr-job-V3-FENCE",
        required_pipeline_version=2,
    )

    assert stale_claim is None
    still_queued = await jobs.find_one({"job_id": "pcr-job-V3-FENCE"})
    assert still_queued["status"] == CAPABILITY_QUEUED_JOB_STATUS
    assert still_queued["attempts"] == 0

    current_claim = await _claim_job(
        db,
        "pcr-job-V3-FENCE",
        execution_token="v3-worker",
        required_pipeline_version=3,
    )

    assert current_claim is not None
    assert current_claim["status"] == "processing"
    assert current_claim["lease_token"] == "v3-worker"
    assert current_claim["attempts"] == 1


@pytest.mark.asyncio
async def test_v5_capability_queue_cannot_be_claimed_by_v4_worker():
    from services.exampen_workflow import _claim_job

    db = _fresh_db()
    await db["exampen_processing_jobs"].insert_one(
        {
            "job_id": "JOB-V5-FENCE",
            "submission_id": "SUB-V5-FENCE",
            "exam_id": "EXAM-V5-FENCE",
            "pipeline_version": 5,
            "mapping_pipeline_version": "bounded-evidence-visual-v5",
            "status": "queued_pipeline_v5",
            "attempts": 0,
        }
    )

    stale_claim = await _claim_job(
        db,
        "JOB-V5-FENCE",
        execution_token="v4-worker",
        required_pipeline_version=4,
    )
    assert stale_claim is None

    current_claim = await _claim_job(
        db,
        "JOB-V5-FENCE",
        execution_token="v5-worker",
        required_pipeline_version=5,
    )
    assert current_claim is not None
    assert current_claim["pipeline_version"] == 5
    assert current_claim["lease_token"] == "v5-worker"


@pytest.mark.asyncio
async def test_reprocess_resets_terminal_copy_with_audit_and_requeues(monkeypatch):
    """A teacher retry must be a fresh, auditable mapping run, not a mutation race."""
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        reprocess_processing_job,
    )

    db = _fresh_db()
    await _seed_supported_subjective_exam(db, "EXAM-1")
    await _seed_submission(db, "SUB-1", "EXAM-1")
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

    assert result["status"] == "queued_pipeline_v7"
    assert task.calls == [("skb_test", "pcr-job-SUB-1", 7)]

    stored = await jobs.find_one({"job_id": "pcr-job-SUB-1"})
    assert stored["last_error"] is None
    assert stored["segmentation"] == {}
    assert stored["evaluation"] == {}
    assert "finished_at" not in stored
    assert stored["mapping_pipeline_version"] == "whole-copy-rubric-v7"
    assert stored["attempts"] == 0
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
            "contract_scope": "exam",
            "selected_copy_only": False,
            "source_prompt_version": "pcr-full-document-visual-v16",
            "target_prompt_version": "pcr-full-document-visual-v16",
            "contract_override_id": None,
        }
    ]


@pytest.mark.asyncio
async def test_reprocess_rejects_unsupported_exam_before_mutating_job(monkeypatch):
    from services.exampen_workflow import (
        GradingContractMigrationRequiredError,
        PROCESSING_JOBS_COLLECTION,
        reprocess_processing_job,
    )

    db = _fresh_db()
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "EXAM-LEGACY",
            "exam_type": "pcr",
            "pcr_grading_contract": {
                "prompt_version": "pcr-full-document-visual-v99"
            },
        }
    )
    await _seed_submission(db, "SUB-LEGACY", "EXAM-LEGACY")
    await db[PROCESSING_JOBS_COLLECTION].insert_one(
        {
            "job_id": "pcr-job-LEGACY",
            "submission_id": "SUB-LEGACY",
            "exam_id": "EXAM-LEGACY",
            "status": "failed",
            "pipeline_version": 2,
        }
    )
    task = _RecordedTask()
    monkeypatch.setitem(
        sys.modules,
        "celery_app",
        SimpleNamespace(process_exampen_pcr_submission=task),
    )

    with pytest.raises(
        GradingContractMigrationRequiredError, match="cannot be reprocessed"
    ):
        await reprocess_processing_job(
            db,
            db_name="skb_test",
            job_id="pcr-job-LEGACY",
            requested_by="TUT-1",
        )

    stored = await db[PROCESSING_JOBS_COLLECTION].find_one(
        {"job_id": "pcr-job-LEGACY"}
    )
    assert stored["status"] == "failed"
    assert "reprocess_count" not in stored
    assert task.calls == []


@pytest.mark.asyncio
async def test_new_job_reconciles_if_exam_contract_changes_during_insert(monkeypatch):
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        ensure_processing_job,
    )

    db = _fresh_db()
    contracts = iter(
        [
            (4, "evidence-first-visual-v4"),
            (7, "whole-copy-rubric-v7"),
        ]
    )

    async def pipeline_metadata(_db, _exam_id, *, job=None):
        return next(contracts)

    monkeypatch.setattr(
        "services.exampen_workflow._pipeline_metadata_for_exam",
        pipeline_metadata,
    )
    await ensure_processing_job(
        db,
        exam_id="EXAM-RACE",
        submission_id="SUB-RACE",
    )
    job = await db[PROCESSING_JOBS_COLLECTION].find_one(
        {"submission_id": "SUB-RACE"}
    )
    assert job["status"] == "queued_pipeline_v7"
    assert job["pipeline_version"] == 7
    assert job["mapping_pipeline_version"] == "whole-copy-rubric-v7"


@pytest.mark.asyncio
async def test_new_job_is_not_dispatched_while_whole_exam_migration_is_applying(
    monkeypatch,
):
    from services.exampen_workflow import (
        CONTRACT_MIGRATION_PENDING_JOB_STATUS,
        dispatch_processing_job,
        ensure_processing_job,
    )

    db = _fresh_db()
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "EXAM-MIGRATING",
            "pcr_grading_contract": {
                "prompt_version": "pcr-full-document-visual-v5"
            },
            "pcr_grading_contract_migration": {
                "migration_id": "PCR-MIG-TEST",
                "status": "applying",
            },
        }
    )
    job, created = await ensure_processing_job(
        db,
        exam_id="EXAM-MIGRATING",
        submission_id="SUB-MIGRATING",
    )
    assert created is True
    assert job["status"] == CONTRACT_MIGRATION_PENDING_JOB_STATUS
    assert job["migration_id"] == "PCR-MIG-TEST"

    task = _RecordedTask()
    monkeypatch.setitem(
        sys.modules,
        "celery_app",
        SimpleNamespace(process_exampen_pcr_submission=task),
    )
    result = await dispatch_processing_job(
        db,
        db_name="skb_test",
        job=job,
    )
    assert result["status"] == CONTRACT_MIGRATION_PENDING_JOB_STATUS
    assert task.calls == []


@pytest.mark.asyncio
async def test_teacher_reprocess_rejects_an_active_processing_lease(monkeypatch):
    """A teacher must not start a second mapper while a live worker owns it."""
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        ProcessingJobBusyError,
        reprocess_processing_job,
    )

    db = _fresh_db()
    await _seed_supported_subjective_exam(db, "EXAM-2")
    await _seed_submission(db, "SUB-2", "EXAM-2")
    jobs = db[PROCESSING_JOBS_COLLECTION]
    await jobs.insert_one(
        {
            "job_id": "pcr-job-SUB-2",
            "submission_id": "SUB-2",
            "exam_id": "EXAM-2",
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
    await _seed_supported_subjective_exam(db, "EXAM-expired")
    await _seed_submission(db, "SUB-expired", "EXAM-expired")
    jobs = db[PROCESSING_JOBS_COLLECTION]
    await jobs.insert_one(
        {
            "job_id": "pcr-job-SUB-expired",
            "submission_id": "SUB-expired",
            "exam_id": "EXAM-expired",
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

    assert result["status"] == "queued_pipeline_v7"
    assert task.calls == [("skb_test", "pcr-job-SUB-expired", 7)]
    stored = await jobs.find_one({"job_id": "pcr-job-SUB-expired"})
    assert "lease_token" not in stored
    assert "lease_expires_at" not in stored
    assert stored["reprocess_history"][0]["force_reclaim"] is True


@pytest.mark.asyncio
async def test_failure_record_schedules_one_bounded_durable_retry():
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        record_processing_job_failure,
    )

    db = _fresh_db()
    jobs = db[PROCESSING_JOBS_COLLECTION]
    now = datetime.now(timezone.utc)
    await jobs.insert_one(
        {
            "job_id": "pcr-job-retry",
            "submission_id": "SUB-retry",
            "status": "processing",
            "attempts": 1,
            "lease_token": "worker-one",
            "lease_expires_at": now + timedelta(minutes=20),
        }
    )

    result = await record_processing_job_failure(
        db,
        "pcr-job-retry",
        RuntimeError("temporary provider outage"),
        expected_lease_token="worker-one",
    )

    assert result["recorded"] is True
    assert result["terminal"] is False
    assert result["status"] == "retryable_error"
    stored = await jobs.find_one({"job_id": "pcr-job-retry"})
    assert stored["status"] == "retryable_error"
    assert stored["failure_code"] == "RuntimeError"
    assert "lease_token" not in stored
    retry_at = stored["next_retry_at"]
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    assert 55 <= (retry_at - now).total_seconds() <= 65


@pytest.mark.asyncio
async def test_failure_record_preserves_provider_failure_code():
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        record_processing_job_failure,
    )

    class _OCRProviderFailure(RuntimeError):
        retryable = False
        failure_code = "ProviderHTTPError"

    db = _fresh_db()
    jobs = db[PROCESSING_JOBS_COLLECTION]
    await jobs.insert_one(
        {
            "job_id": "pcr-job-provider-code",
            "submission_id": "SUB-provider-code",
            "status": "processing",
            "attempts": 1,
            "lease_token": "worker-one",
        }
    )

    result = await record_processing_job_failure(
        db,
        "pcr-job-provider-code",
        _OCRProviderFailure("model is unavailable"),
        expected_lease_token="worker-one",
    )

    assert result["terminal"] is True
    stored = await jobs.find_one({"job_id": "pcr-job-provider-code"})
    assert stored["failure_code"] == "ProviderHTTPError"


@pytest.mark.asyncio
async def test_failure_record_stops_after_global_attempt_budget():
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        PROCESSING_MAX_AUTOMATIC_ATTEMPTS,
        record_processing_job_failure,
    )

    db = _fresh_db()
    jobs = db[PROCESSING_JOBS_COLLECTION]
    await jobs.insert_one(
        {
            "job_id": "pcr-job-exhausted",
            "submission_id": "SUB-exhausted",
            "status": "processing",
            "attempts": PROCESSING_MAX_AUTOMATIC_ATTEMPTS,
            "lease_token": "worker-final",
        }
    )

    result = await record_processing_job_failure(
        db,
        "pcr-job-exhausted",
        RuntimeError("provider remains unavailable"),
        expected_lease_token="worker-final",
    )

    assert result["recorded"] is True
    assert result["terminal"] is True
    stored = await jobs.find_one({"job_id": "pcr-job-exhausted"})
    assert stored["status"] == "failed"
    assert "next_retry_at" not in stored
    assert "finished_at" in stored


@pytest.mark.asyncio
async def test_deterministic_failure_is_not_retried():
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        record_processing_job_failure,
    )

    class _DeterministicError(RuntimeError):
        retryable = False

    db = _fresh_db()
    jobs = db[PROCESSING_JOBS_COLLECTION]
    await jobs.insert_one(
        {
            "job_id": "pcr-job-permanent",
            "submission_id": "SUB-permanent",
            "status": "processing",
            "attempts": 1,
            "lease_token": "worker-one",
        }
    )

    result = await record_processing_job_failure(
        db,
        "pcr-job-permanent",
        _DeterministicError("identity conflict"),
        expected_lease_token="worker-one",
    )

    assert result["terminal"] is True
    stored = await jobs.find_one({"job_id": "pcr-job-permanent"})
    assert stored["status"] == "failed"
    assert stored["failure_code"] == "_DeterministicError"


@pytest.mark.asyncio
async def test_concurrent_dispatch_reserves_job_once(monkeypatch):
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        dispatch_processing_job,
    )

    db = _fresh_db()
    jobs = db[PROCESSING_JOBS_COLLECTION]
    job = {
        "job_id": "pcr-job-dispatch-once",
        "submission_id": "SUB-dispatch-once",
        "status": "queued",
    }
    await jobs.insert_one(job)
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

    await asyncio.gather(
        dispatch_processing_job(db, db_name="skb_test", job=dict(job)),
        dispatch_processing_job(db, db_name="skb_test", job=dict(job)),
    )

    assert task.calls == [("skb_test", "pcr-job-dispatch-once", 4)]


@pytest.mark.asyncio
async def test_reconciler_dispatches_only_due_retries(monkeypatch):
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        reconcile_processing_jobs,
    )

    db = _fresh_db()
    jobs = db[PROCESSING_JOBS_COLLECTION]
    now = datetime.now(timezone.utc)
    await jobs.insert_many(
        [
            {
                "job_id": "pcr-job-future",
                "submission_id": "SUB-future",
                "status": "retryable_error",
                "enqueue_attempted_at": now - timedelta(minutes=10),
                "next_retry_at": now + timedelta(minutes=5),
                "updated_at": now,
            },
            {
                "job_id": "pcr-job-due",
                "submission_id": "SUB-due",
                "status": "retryable_error",
                "enqueue_attempted_at": now - timedelta(minutes=10),
                "next_retry_at": now - timedelta(seconds=1),
                "updated_at": now,
            },
        ]
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

    result = await reconcile_processing_jobs(db, db_name="skb_test")

    assert result["pending"] == 1
    assert task.calls == [("skb_test", "pcr-job-due", 4)]
    future = await jobs.find_one({"job_id": "pcr-job-future"})
    due = await jobs.find_one({"job_id": "pcr-job-due"})
    assert future["status"] == "retryable_error"
    assert due["status"] == "queued_pipeline_v3"


@pytest.mark.asyncio
async def test_duplicate_worker_delivery_cannot_bypass_retry_schedule():
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        process_pcr_processing_job,
    )

    db = _fresh_db()
    jobs = db[PROCESSING_JOBS_COLLECTION]
    retry_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    await jobs.insert_one(
        {
            "job_id": "pcr-job-early-duplicate",
            "submission_id": "SUB-early-duplicate",
            "status": "retryable_error",
            "attempts": 1,
            "next_retry_at": retry_at,
        }
    )

    result = await process_pcr_processing_job(db, "pcr-job-early-duplicate")

    assert result == {
        "job_id": "pcr-job-early-duplicate",
        "status": "retryable_error",
        "claimed": False,
    }
    stored = await jobs.find_one({"job_id": "pcr-job-early-duplicate"})
    assert stored["status"] == "retryable_error"
    assert stored["attempts"] == 1
    saved_retry_at = stored["next_retry_at"]
    if saved_retry_at.tzinfo is None:
        saved_retry_at = saved_retry_at.replace(tzinfo=timezone.utc)
    assert abs((saved_retry_at - retry_at).total_seconds()) < 0.01
    assert "lease_token" not in stored


@pytest.mark.asyncio
async def test_automatic_recovery_uses_missing_heartbeats_without_reclaiming_live_worker():
    from services.exampen_workflow import (
        PROCESSING_HEARTBEAT_STALE_SECONDS,
        PROCESSING_JOBS_COLLECTION,
        recover_stale_processing_jobs,
    )

    db = _fresh_db()
    jobs = db[PROCESSING_JOBS_COLLECTION]
    now = datetime.now(timezone.utc)
    await jobs.insert_many(
        [
            {
                "job_id": "pcr-job-stalled",
                "submission_id": "SUB-stalled",
                "status": "processing",
                "lease_token": "dead-worker",
                # The long ownership lease has not expired, but its heartbeat
                # stopped. This is the production failure that left Checking
                # visible indefinitely.
                "lease_expires_at": now + timedelta(minutes=20),
                "updated_at": now
                - timedelta(seconds=PROCESSING_HEARTBEAT_STALE_SECONDS + 1),
            },
            {
                "job_id": "pcr-job-live",
                "submission_id": "SUB-live",
                "status": "processing",
                "lease_token": "live-worker",
                "lease_expires_at": now + timedelta(minutes=20),
                "updated_at": now - timedelta(seconds=10),
            },
        ]
    )

    recovered = await recover_stale_processing_jobs(db, now=now)

    assert recovered == 1
    stalled = await jobs.find_one({"job_id": "pcr-job-stalled"})
    assert stalled["status"] == "retryable_error"
    assert "heartbeat stopped" in stalled["last_error"].lower()
    recovered_at = stalled["stale_worker_recovered_at"]
    if recovered_at.tzinfo is None:
        recovered_at = recovered_at.replace(tzinfo=timezone.utc)
    assert abs((recovered_at - now).total_seconds()) < 0.01
    assert "lease_token" not in stalled
    assert "lease_expires_at" not in stalled
    retry_at = stalled["next_retry_at"]
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    assert 55 <= (retry_at - now).total_seconds() <= 65
    live = await jobs.find_one({"job_id": "pcr-job-live"})
    assert live["status"] == "processing"
    assert live["lease_token"] == "live-worker"


@pytest.mark.asyncio
async def test_stale_worker_recovery_honors_global_attempt_budget():
    from services.exampen_workflow import (
        PROCESSING_HEARTBEAT_STALE_SECONDS,
        PROCESSING_JOBS_COLLECTION,
        PROCESSING_MAX_AUTOMATIC_ATTEMPTS,
        recover_stale_processing_jobs,
    )

    db = _fresh_db()
    jobs = db[PROCESSING_JOBS_COLLECTION]
    now = datetime.now(timezone.utc)
    await jobs.insert_one(
        {
            "job_id": "pcr-job-stalled-exhausted",
            "submission_id": "SUB-stalled-exhausted",
            "status": "processing",
            "attempts": PROCESSING_MAX_AUTOMATIC_ATTEMPTS,
            "lease_token": "dead-final-worker",
            "lease_expires_at": now + timedelta(minutes=20),
            "updated_at": now
            - timedelta(seconds=PROCESSING_HEARTBEAT_STALE_SECONDS + 1),
        }
    )

    recovered = await recover_stale_processing_jobs(db, now=now)

    assert recovered == 1
    stored = await jobs.find_one({"job_id": "pcr-job-stalled-exhausted"})
    assert stored["status"] == "failed"
    assert stored["failure_code"] == "ProcessingWorkerHeartbeatExpired"
    assert "retry budget was exhausted" in stored["last_error"]
    assert "next_retry_at" not in stored
    assert "lease_token" not in stored


@pytest.mark.asyncio
async def test_stale_worker_recovery_does_not_downgrade_unknown_future_pipeline():
    from services.exampen_workflow import (
        PROCESSING_HEARTBEAT_STALE_SECONDS,
        PROCESSING_JOBS_COLLECTION,
        recover_stale_processing_jobs,
    )

    db = _fresh_db()
    jobs = db[PROCESSING_JOBS_COLLECTION]
    now = datetime.now(timezone.utc)
    await jobs.insert_one(
        {
            "job_id": "pcr-job-future-pipeline",
            "submission_id": "SUB-future-pipeline",
            "status": "processing",
            "pipeline_version": 8,
            "attempts": 1,
            "lease_token": "future-worker",
            "lease_expires_at": now + timedelta(minutes=20),
            "updated_at": now
            - timedelta(seconds=PROCESSING_HEARTBEAT_STALE_SECONDS + 1),
        }
    )

    assert await recover_stale_processing_jobs(db, now=now) == 1
    stored = await jobs.find_one({"job_id": "pcr-job-future-pipeline"})
    assert stored["pipeline_version"] == 8
    assert stored["status"] == "retryable_error"


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
async def test_dispatch_without_redis_leaves_job_for_bounded_inline_processor(monkeypatch):
    """Broker failure must not launch an untracked API-process task."""
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

    assert result["status"] == "queued_pipeline_v3"
    assert "broker unavailable" in str(result.get("last_error") or "").lower()


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
