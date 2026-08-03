from __future__ import annotations

import pytest
from mongomock_motor import AsyncMongoMockClient

from services.pcr_grading_contract_migration import (
    GradingContractMigrationError,
    inspect_v5_contracts,
    migrate_v5_exam_to_v6,
)


def _db():
    return AsyncMongoMockClient()["skb_test"]


async def _seed(db, *, published: bool = False) -> None:
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "EXAM-1",
            "exam_name": "Physics",
            "paper_context": {
                "ready": True,
                "version": "canonical-full-document-visual-v2",
            },
            "pcr_grading_contract": {
                "prompt_version": "pcr-full-document-visual-v5",
                "model_id": "gpt-5.1-2025-11-13",
                "reasoning_effort": "medium",
                "temperature": 0.1,
                "locked_at": "original-lock",
            },
        }
    )
    for index in range(2):
        submission_id = f"SUB-{index}"
        submission = {
            "submission_id": submission_id,
            "exam_id": "EXAM-1",
            "review_state": "blocked",
        }
        if published and index == 0:
            submission["publication_status"] = "published"
        await db["evalpen_submissions"].insert_one(submission)
        await db["exampen_processing_jobs"].insert_one(
            {
                "job_id": f"JOB-{index}",
                "submission_id": submission_id,
                "exam_id": "EXAM-1",
                "status": "failed",
                "attempts": 4,
                "last_error": "unsupported contract",
                "failure_code": "UnsupportedGradingContractError",
                "retry_at": "later",
            }
        )
    await db["evalpen_document_grading_runs"].insert_one(
        {
            "run_id": "RUN-1",
            "exam_id": "EXAM-1",
            "prompt_version": "pcr-full-document-visual-v5",
            "status": "failed",
        }
    )


@pytest.mark.asyncio
async def test_migration_updates_the_contract_and_requeues_the_complete_cohort():
    db = _db()
    await _seed(db)

    plans = await inspect_v5_contracts(db, db_name="skb_test", exam_id="EXAM-1")
    assert plans[0]["eligible"] is True

    result = await migrate_v5_exam_to_v6(
        db, db_name="skb_test", exam_id="EXAM-1", requested_by="OPS-1"
    )

    assert result["status"] == "migrated"
    assert result["queued_job_count"] == 2
    exam = await db["exampen_exams"].find_one({"exam_id": "EXAM-1"})
    contract = exam["pcr_grading_contract"]
    assert contract["prompt_version"] == "pcr-full-document-visual-v6"
    assert contract["model_id"] == "gpt-5.1-2025-11-13"
    assert contract["locked_at"] == "original-lock"
    assert exam["pcr_grading_contract_migration"]["status"] == "complete"
    jobs = await db["exampen_processing_jobs"].find({}).to_list(length=None)
    assert {job["status"] for job in jobs} == {"queued"}
    assert {job["attempts"] for job in jobs} == {0}
    assert all("last_error" not in job for job in jobs)
    assert all(job["reprocess_count"] == 1 for job in jobs)
    run = await db["evalpen_document_grading_runs"].find_one({"run_id": "RUN-1"})
    assert run["status"] == "superseded"
    audit = await db["exampen_grading_contract_migrations"].find_one(
        {"migration_id": result["migration_id"]}
    )
    assert audit["status"] == "complete"


@pytest.mark.asyncio
async def test_migration_refuses_to_mix_with_published_results():
    db = _db()
    await _seed(db, published=True)

    plans = await inspect_v5_contracts(db, db_name="skb_test", exam_id="EXAM-1")
    assert plans[0]["eligible"] is False
    assert "published submission" in " ".join(plans[0]["blockers"])

    with pytest.raises(GradingContractMigrationError, match="published submission"):
        await migrate_v5_exam_to_v6(
            db, db_name="skb_test", exam_id="EXAM-1", requested_by="OPS-1"
        )

    exam = await db["exampen_exams"].find_one({"exam_id": "EXAM-1"})
    assert exam["pcr_grading_contract"]["prompt_version"] == (
        "pcr-full-document-visual-v5"
    )
