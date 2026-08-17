from __future__ import annotations

import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from mongomock_motor import AsyncMongoMockClient


def _db():
    return AsyncMongoMockClient()["skb_selected_copy_test"]


class _RecordedTask:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def delay(self, db_name: str, job_id: str, pipeline_version: int) -> None:
        self.calls.append((db_name, job_id, pipeline_version))


async def _seed_legacy_exam(db, exam_id: str) -> None:
    paper_version_id = f"PV-{exam_id}"
    document_id = f"DOC-{exam_id}"
    await db["exampen_exams"].insert_one(
        {
            "exam_id": exam_id,
            "exam_type": "pcr",
            "paper_version_id": paper_version_id,
            "prepared_document_id": document_id,
            "pcr_grading_contract": {
                "prompt_version": "pcr-full-document-visual-v12",
                "pipeline_version": 3,
                "mapping_pipeline_version": "evidence-first-visual-v3",
                "required_processing_path": "full_document_visual",
                "model_id": "gpt-5.1-2025-11-13",
            },
        }
    )
    await db["exampen_paper_versions"].insert_one(
        {
            "paper_version_id": paper_version_id,
            "document_id": document_id,
            "paper_context": {
                "ready": True,
                "version": "canonical-full-document-visual-v2",
                "question_paper_asset_id": f"ASSET-{exam_id}",
                "has_teacher_solution_asset": False,
            },
            "paper_assets": {
                "question_paper": {
                    "asset_id": f"ASSET-{exam_id}",
                    "storage_uri": f"s3://test/{exam_id}/paper.pdf",
                }
            },
        }
    )
    await db["evalpen_questions"].insert_one(
        {
            "question_id": f"{exam_id}::Q1",
            "exam_id": exam_id,
            "question_number": 1,
            "question_type": "subjective",
            "grading_mode": "subjective",
        }
    )


async def _seed_submission(
    db,
    submission_id: str,
    exam_id: str,
    *,
    publication_status: str = "pending",
) -> None:
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": submission_id,
            "exam_id": exam_id,
            "student_id": f"STU-{submission_id}",
            "source": "camera",
            "publication_status": publication_status,
        }
    )
    await db["evalpen_answer_pages"].insert_one(
        {
            "page_id": f"PAGE-{submission_id}",
            "submission_id": submission_id,
            "page_number": 1,
            "raw_image_ref": f"s3://test/{submission_id}/page-1.jpg",
        }
    )


@pytest.mark.asyncio
async def test_legacy_reprocess_upgrades_only_selected_copy(monkeypatch):
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        reprocess_processing_job,
    )

    db = _db()
    await _seed_legacy_exam(db, "EXAM-ONE")
    await _seed_submission(db, "SUB-SELECTED", "EXAM-ONE")
    await _seed_submission(db, "SUB-SIBLING", "EXAM-ONE")
    await db[PROCESSING_JOBS_COLLECTION].insert_many(
        [
            {
                "job_id": "JOB-SELECTED",
                "submission_id": "SUB-SELECTED",
                "exam_id": "EXAM-ONE",
                "status": "completed",
                "pipeline_version": 3,
            },
            {
                "job_id": "JOB-SIBLING",
                "submission_id": "SUB-SIBLING",
                "exam_id": "EXAM-ONE",
                "status": "completed",
                "pipeline_version": 3,
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

    result = await reprocess_processing_job(
        db,
        db_name="skb_any_tenant",
        job_id="JOB-SELECTED",
        requested_by="TUT-ANY",
    )

    selected = await db[PROCESSING_JOBS_COLLECTION].find_one({"job_id": "JOB-SELECTED"})
    sibling = await db[PROCESSING_JOBS_COLLECTION].find_one({"job_id": "JOB-SIBLING"})
    exam = await db["exampen_exams"].find_one({"exam_id": "EXAM-ONE"})
    override = selected["grading_contract_override"]
    assert result["status"] == "queued_pipeline_v7"
    assert task.calls == [("skb_any_tenant", "JOB-SELECTED", 7)]
    assert override["scope"] == "selected_submission_reprocess"
    assert override["target_submission_id"] == "SUB-SELECTED"
    assert override["source_prompt_version"] == "pcr-full-document-visual-v12"
    assert override["prompt_version"] == "pcr-full-document-visual-v16"
    assert selected["reprocess_count"] == 1
    assert sibling["status"] == "completed"
    assert sibling["pipeline_version"] == 3
    assert "grading_contract_override" not in sibling
    assert exam["pcr_grading_contract"]["prompt_version"] == "pcr-full-document-visual-v12"


@pytest.mark.asyncio
async def test_second_click_is_rejected_before_second_queue(monkeypatch):
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        ProcessingJobBusyError,
        reprocess_processing_job,
    )

    db = _db()
    await _seed_legacy_exam(db, "EXAM-DUP")
    await _seed_submission(db, "SUB-DUP", "EXAM-DUP")
    await db[PROCESSING_JOBS_COLLECTION].insert_one(
        {
            "job_id": "JOB-DUP",
            "submission_id": "SUB-DUP",
            "exam_id": "EXAM-DUP",
            "status": "completed",
            "pipeline_version": 3,
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

    await reprocess_processing_job(
        db,
        db_name="skb_any_tenant",
        job_id="JOB-DUP",
        requested_by="TUT-ANY",
    )
    with pytest.raises(ProcessingJobBusyError, match="already queued"):
        await reprocess_processing_job(
            db,
            db_name="skb_any_tenant",
            job_id="JOB-DUP",
            requested_by="TUT-ANY",
        )

    stored = await db[PROCESSING_JOBS_COLLECTION].find_one({"job_id": "JOB-DUP"})
    assert stored["reprocess_count"] == 1
    assert len(stored["reprocess_history"]) == 1
    assert task.calls == [("skb_any_tenant", "JOB-DUP", 7)]


@pytest.mark.asyncio
async def test_published_copy_is_not_upgraded_or_queued(monkeypatch):
    from services.exampen_workflow import (
        GradingContractMigrationRequiredError,
        PROCESSING_JOBS_COLLECTION,
        reprocess_processing_job,
    )

    db = _db()
    await _seed_legacy_exam(db, "EXAM-PUBLISHED")
    await _seed_submission(
        db,
        "SUB-PUBLISHED",
        "EXAM-PUBLISHED",
        publication_status="published",
    )
    await db[PROCESSING_JOBS_COLLECTION].insert_one(
        {
            "job_id": "JOB-PUBLISHED",
            "submission_id": "SUB-PUBLISHED",
            "exam_id": "EXAM-PUBLISHED",
            "status": "completed",
            "pipeline_version": 3,
        }
    )
    task = _RecordedTask()
    monkeypatch.setitem(
        sys.modules,
        "celery_app",
        SimpleNamespace(process_exampen_pcr_submission=task),
    )

    with pytest.raises(GradingContractMigrationRequiredError, match="Published"):
        await reprocess_processing_job(
            db,
            db_name="skb_any_tenant",
            job_id="JOB-PUBLISHED",
            requested_by="TUT-ANY",
        )
    stored = await db[PROCESSING_JOBS_COLLECTION].find_one({"job_id": "JOB-PUBLISHED"})
    assert stored["status"] == "completed"
    assert "grading_contract_override" not in stored
    assert task.calls == []


@pytest.mark.asyncio
async def test_objective_paper_is_not_upgraded_into_subjective_ai_lane(monkeypatch):
    from services.exampen_workflow import (
        GradingContractMigrationRequiredError,
        PROCESSING_JOBS_COLLECTION,
        reprocess_processing_job,
    )

    db = _db()
    await _seed_legacy_exam(db, "EXAM-OBJECTIVE")
    await _seed_submission(db, "SUB-OBJECTIVE", "EXAM-OBJECTIVE")
    await db["evalpen_questions"].update_one(
        {"exam_id": "EXAM-OBJECTIVE"},
        {
            "$set": {
                "question_type": "mcq",
                "grading_mode": "objective",
            }
        },
    )
    await db[PROCESSING_JOBS_COLLECTION].insert_one(
        {
            "job_id": "JOB-OBJECTIVE",
            "submission_id": "SUB-OBJECTIVE",
            "exam_id": "EXAM-OBJECTIVE",
            "status": "completed",
            "pipeline_version": 3,
        }
    )
    task = _RecordedTask()
    monkeypatch.setitem(
        sys.modules,
        "celery_app",
        SimpleNamespace(process_exampen_pcr_submission=task),
    )

    with pytest.raises(GradingContractMigrationRequiredError, match="Objective"):
        await reprocess_processing_job(
            db,
            db_name="skb_any_tenant",
            job_id="JOB-OBJECTIVE",
            requested_by="TUT-ANY",
        )

    stored = await db[PROCESSING_JOBS_COLLECTION].find_one(
        {"job_id": "JOB-OBJECTIVE"}
    )
    assert stored["status"] == "completed"
    assert "grading_contract_override" not in stored
    assert task.calls == []


@pytest.mark.asyncio
async def test_v16_worker_accepts_selected_override_on_legacy_exam():
    from services.exampen_workflow import (
        PROCESSING_JOBS_COLLECTION,
        process_pcr_processing_job,
    )
    from services.pcr_grading_contract_policy import build_selected_copy_v16_override

    db = _db()
    await _seed_legacy_exam(db, "EXAM-WORKER")
    await _seed_submission(db, "SUB-WORKER", "EXAM-WORKER")
    exam = await db["exampen_exams"].find_one({"exam_id": "EXAM-WORKER"})
    override = build_selected_copy_v16_override(
        exam["pcr_grading_contract"],
        submission_id="SUB-WORKER",
        requested_by="TUT-WORKER",
        requested_at=datetime.now(timezone.utc),
    )
    await db[PROCESSING_JOBS_COLLECTION].insert_one(
        {
            "job_id": "JOB-WORKER",
            "submission_id": "SUB-WORKER",
            "exam_id": "EXAM-WORKER",
            "status": "queued_pipeline_v7",
            "pipeline_version": 7,
            "mapping_pipeline_version": "whole-copy-rubric-v7",
            "required_processing_path": "full_document_visual",
            "grading_contract_override": override,
            "attempts": 0,
        }
    )

    class _Gate:
        def __init__(self, _db):
            pass

        async def initialize(self):
            return None

    class _ObjectiveGrader:
        def __init__(self, _db, _gate):
            pass

        async def grade_submission(self, _submission_id: str):
            return SimpleNamespace(handled=False, skipped_reason="Subjective paper")

    class _DocumentGrader:
        def __init__(self, _db, _gate):
            pass

        async def grade_submission(self, submission_id: str):
            assert submission_id == "SUB-WORKER"
            return SimpleNamespace(
                handled=True,
                status="completed",
                page_count=4,
                response_count=9,
                evaluated_count=9,
                blocked_count=0,
                warning_count=0,
                run_id="DOCGR-WORKER",
                errors=[],
                review_state="ready",
                document_review_required=False,
                review_reasons=[],
                processing_path="full_document_visual",
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
        result = await process_pcr_processing_job(
            db,
            "JOB-WORKER",
            required_pipeline_version=7,
        )

    assert result["status"] == "completed"
    stored = await db[PROCESSING_JOBS_COLLECTION].find_one({"job_id": "JOB-WORKER"})
    assert stored["grading_contract_override"]["override_id"] == override["override_id"]
