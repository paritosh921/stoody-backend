from __future__ import annotations

import pytest
from mongomock_motor import AsyncMongoMockClient

from services.pcr_grading_contract_migration import (
    GradingContractMigrationError,
    V13_TO_V14_CONFIRMATION_TOKEN,
    inspect_v13_contracts,
    migrate_v13_exam_to_v14,
)


def _db():
    return AsyncMongoMockClient()["skb_v14_test"]


async def _seed(db, *, exam_status: str | None = None, published: bool = False):
    exam = {
        "exam_id": "EXAM-V14",
        "exam_name": "Physics",
        "prepared_document_id": "DOC-V14",
        "paper_version_id": "PAPER-V14",
        "pcr_grading_contract": {
            "prompt_version": "pcr-full-document-visual-v13",
            "model_id": "gpt-5.1-2025-11-13",
            "locked_at": "frozen",
        },
    }
    if exam_status:
        exam["status"] = exam_status
    await db["exampen_exams"].insert_one(exam)
    await db["exampen_paper_versions"].insert_one(
        {
            "paper_version_id": "PAPER-V14",
            "document_id": "DOC-V14",
            "paper_context": {
                "ready": True,
                "version": "canonical-full-document-visual-v2",
                "question_paper_asset_id": "Q-ASSET",
            },
            "paper_assets": {
                "question_paper": {
                    "asset_id": "Q-ASSET",
                    "storage_uri": "s3://papers/q.pdf",
                }
            },
        }
    )
    for index in range(2):
        submission_id = f"SUB-V14-{index}"
        submission = {"submission_id": submission_id, "exam_id": "EXAM-V14"}
        if published and index == 0:
            submission["publication_status"] = "published"
        await db["evalpen_submissions"].insert_one(submission)
        await db["exampen_processing_jobs"].insert_one(
            {
                "job_id": f"JOB-V14-{index}",
                "submission_id": submission_id,
                "exam_id": "EXAM-V14",
                "status": "failed",
                "attempts": 4,
            }
        )
    await db["evalpen_document_grading_runs"].insert_one(
        {
            "run_id": "RUN-V14",
            "exam_id": "EXAM-V14",
            "prompt_version": "pcr-full-document-visual-v13",
            "status": "failed",
        }
    )


@pytest.mark.asyncio
async def test_v14_inspection_is_dry_and_rejects_active_or_published_cohorts():
    db = _db()
    await _seed(db, exam_status="active")
    plans = await inspect_v13_contracts(db, db_name="skb_v14_test")
    assert plans[0]["eligible"] is False
    assert "active" in " ".join(plans[0]["blockers"])
    exam = await db["exampen_exams"].find_one({"exam_id": "EXAM-V14"})
    assert exam["pcr_grading_contract"]["prompt_version"].endswith("v13")

    db = _db()
    await _seed(db, published=True)
    plans = await inspect_v13_contracts(db, db_name="skb_v14_test")
    assert plans[0]["eligible"] is False
    assert "published submission" in " ".join(plans[0]["blockers"])


@pytest.mark.asyncio
async def test_v14_requires_confirmation_and_queues_one_job_per_submission():
    db = _db()
    await _seed(db)
    with pytest.raises(GradingContractMigrationError, match="confirmation token"):
        await migrate_v13_exam_to_v14(
            db,
            db_name="skb_v14_test",
            exam_id="EXAM-V14",
            requested_by="OPS",
            confirmation_token="wrong",
        )
    result = await migrate_v13_exam_to_v14(
        db,
        db_name="skb_v14_test",
        exam_id="EXAM-V14",
        requested_by="OPS",
        confirmation_token=V13_TO_V14_CONFIRMATION_TOKEN,
    )
    assert result["status"] == "migrated"
    assert result["queued_job_count"] == 2
    exam = await db["exampen_exams"].find_one({"exam_id": "EXAM-V14"})
    assert exam["pcr_grading_contract"]["prompt_version"].endswith("v14")
    assert exam["pcr_grading_contract"]["pipeline_version"] == 5
    assert exam["pcr_grading_contract"]["mapping_pipeline_version"] == "bounded-evidence-visual-v5"
    jobs = await db["exampen_processing_jobs"].find({}).to_list(length=None)
    assert len(jobs) == 2
    assert {job["pipeline_version"] for job in jobs} == {5}
    assert {job["status"] for job in jobs} == {"queued_pipeline_v5"}
    assert {job["mapping_pipeline_version"] for job in jobs} == {"bounded-evidence-visual-v5"}
    assert {job["required_processing_path"] for job in jobs} == {"full_document_visual"}
    assert await db["evalpen_document_grading_runs"].count_documents({"status": "superseded"}) == 1

    # Repeating the same operator action is a no-op, not another queue/reprocess.
    again = await migrate_v13_exam_to_v14(
        db,
        db_name="skb_v14_test",
        exam_id="EXAM-V14",
        requested_by="OPS",
        confirmation_token=V13_TO_V14_CONFIRMATION_TOKEN,
    )
    assert again["status"] == "already_migrated"
    jobs = await db["exampen_processing_jobs"].find({}).to_list(length=None)
    assert {job["reprocess_count"] for job in jobs} == {1}


@pytest.mark.asyncio
async def test_existing_job_cannot_be_reused_with_a_stale_pipeline_contract():
    from services.exampen_workflow import ensure_processing_job

    db = _db()
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "EXAM-STALE",
            "pcr_grading_contract": {
                "prompt_version": "pcr-full-document-visual-v14",
                "pipeline_version": 5,
                "mapping_pipeline_version": "bounded-evidence-visual-v5",
            },
        }
    )
    await db["exampen_processing_jobs"].insert_one(
        {
            "job_id": "pcr-job-SUB-STALE",
            "submission_id": "SUB-STALE",
            "exam_id": "EXAM-STALE",
            "pipeline_version": 4,
            "mapping_pipeline_version": "evidence-first-visual-v4",
            "status": "failed",
        }
    )
    with pytest.raises(ValueError, match="requires pipeline 5"):
        await ensure_processing_job(
            db,
            exam_id="EXAM-STALE",
            submission_id="SUB-STALE",
        )
