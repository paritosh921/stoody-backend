from __future__ import annotations

import pytest
from mongomock_motor import AsyncMongoMockClient

from services.exampen_workflow import (
    DISPATCHABLE_JOB_STATUSES,
    PCR_V16_MAPPING_VERSION,
    PCR_V16_PIPELINE_VERSION,
    V16_CAPABILITY_QUEUED_JOB_STATUS,
    _pipeline_metadata_for_exam,
    _queued_status_for_pipeline,
    is_supported_pcr_grading_contract,
)
from services.pcr_grading_contract_migration import (
    GradingContractMigrationError,
    V15_TO_V16_CONFIRMATION_TOKEN,
    inspect_v15_contracts,
    migrate_v15_exam_to_v16,
)


def _db():
    return AsyncMongoMockClient()["skb_v16_test"]


async def _seed(db, *, published: bool = False):
    await db["exampen_exams"].insert_one({
        "exam_id": "EXAM-V16",
        "exam_name": "Hindi",
        "prepared_document_id": "DOC-V16",
        "paper_version_id": "PAPER-V16",
        "pcr_grading_contract": {
            "prompt_version": "pcr-full-document-visual-v15",
            "pipeline_version": 6,
            "mapping_pipeline_version": "bounded-evidence-visual-v6",
            "required_processing_path": "full_document_visual",
            "model_id": "gpt-5.1-2025-11-13",
        },
    })
    await db["exampen_paper_versions"].insert_one({
        "paper_version_id": "PAPER-V16",
        "document_id": "DOC-V16",
        "paper_context": {
            "ready": True,
            "version": "canonical-full-document-visual-v2",
            "question_paper_asset_id": "Q-ASSET",
        },
        "paper_assets": {
            "question_paper": {
                "asset_id": "Q-ASSET",
                "storage_uri": "s3://papers/question.pdf",
            }
        },
    })
    for index in range(2):
        submission_id = f"SUB-V16-{index}"
        submission = {"submission_id": submission_id, "exam_id": "EXAM-V16"}
        if published and index == 0:
            submission["publication_status"] = "published"
        await db["evalpen_submissions"].insert_one(submission)
        await db["exampen_processing_jobs"].insert_one({
            "job_id": f"JOB-V16-{index}",
            "submission_id": submission_id,
            "exam_id": "EXAM-V16",
            "pipeline_version": 6,
            "mapping_pipeline_version": "bounded-evidence-visual-v6",
            "status": "failed",
            "attempts": 1,
        })
    await db["evalpen_document_grading_runs"].insert_one({
        "run_id": "RUN-V15-FAILED",
        "exam_id": "EXAM-V16",
        "prompt_version": "pcr-full-document-visual-v15",
        "status": "failed",
    })


@pytest.mark.asyncio
async def test_v16_migration_is_guarded_immutable_and_queues_pipeline_7():
    assert V16_CAPABILITY_QUEUED_JOB_STATUS in DISPATCHABLE_JOB_STATUSES
    assert is_supported_pcr_grading_contract({"prompt_version": "pcr-full-document-visual-v16"})
    db = _db()
    await _seed(db)
    plans = await inspect_v15_contracts(db, db_name="skb_v16_test", exam_id="EXAM-V16")
    assert plans[0]["eligible"] is True

    with pytest.raises(GradingContractMigrationError, match="confirmation token"):
        await migrate_v15_exam_to_v16(
            db,
            db_name="skb_v16_test",
            exam_id="EXAM-V16",
            requested_by="OPS",
            confirmation_token="WRONG",
        )

    result = await migrate_v15_exam_to_v16(
        db,
        db_name="skb_v16_test",
        exam_id="EXAM-V16",
        requested_by="OPS",
        confirmation_token=V15_TO_V16_CONFIRMATION_TOKEN,
    )
    assert result["status"] == "migrated"
    assert result["queued_job_count"] == 2
    exam = await db["exampen_exams"].find_one({"exam_id": "EXAM-V16"})
    contract = exam["pcr_grading_contract"]
    assert contract["prompt_version"] == "pcr-full-document-visual-v16"
    assert contract["pipeline_version"] == PCR_V16_PIPELINE_VERSION
    assert contract["mapping_pipeline_version"] == PCR_V16_MAPPING_VERSION
    assert contract["migrated_from"] == "pcr-full-document-visual-v15"
    jobs = await db["exampen_processing_jobs"].find({}).to_list(length=None)
    assert {job["status"] for job in jobs} == {V16_CAPABILITY_QUEUED_JOB_STATUS}
    assert {job["pipeline_version"] for job in jobs} == {7}
    assert {job["mapping_pipeline_version"] for job in jobs} == {
        "whole-copy-rubric-v7"
    }
    old = await db["evalpen_document_grading_runs"].find_one({"run_id": "RUN-V15-FAILED"})
    assert old["status"] == "superseded"
    assert await _pipeline_metadata_for_exam(db, "EXAM-V16") == (
        7,
        "whole-copy-rubric-v7",
    )
    assert _queued_status_for_pipeline(7) == V16_CAPABILITY_QUEUED_JOB_STATUS


@pytest.mark.asyncio
async def test_v16_migration_rejects_published_cohort():
    db = _db()
    await _seed(db, published=True)
    plans = await inspect_v15_contracts(db, db_name="skb_v16_test", exam_id="EXAM-V16")
    assert plans[0]["eligible"] is False
    assert "published submission" in " ".join(plans[0]["blockers"])


@pytest.mark.asyncio
async def test_v16_cli_requires_exact_confirmation_before_database(monkeypatch):
    from scripts import migrate_pcr_v15_to_v16 as script

    initialized = False

    async def initialize(_self):
        nonlocal initialized
        initialized = True

    monkeypatch.setattr(script.DatabaseManager, "initialize", initialize)
    args = script.build_parser().parse_args(["--apply", "--confirm", "WRONG"])
    assert await script.run(args) == 2
    assert initialized is False
