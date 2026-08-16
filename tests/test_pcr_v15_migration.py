from __future__ import annotations

import pytest
from mongomock_motor import AsyncMongoMockClient

from services.pcr_grading_contract_migration import (
    GradingContractMigrationError,
    V14_TO_V15_CONFIRMATION_TOKEN,
    inspect_v14_contracts,
    migrate_v14_exam_to_v15,
)
from services.exampen_workflow import (
    DISPATCHABLE_JOB_STATUSES,
    V15_CAPABILITY_QUEUED_JOB_STATUS,
)


def _db():
    return AsyncMongoMockClient()["skb_v15_test"]


async def _seed(db, *, published: bool = False):
    await db["exampen_exams"].insert_one({
        "exam_id": "EXAM-V15",
        "exam_name": "Physics",
        "prepared_document_id": "DOC-V15",
        "paper_version_id": "PAPER-V15",
        "pcr_grading_contract": {
            "prompt_version": "pcr-full-document-visual-v14",
            "pipeline_version": 5,
            "mapping_pipeline_version": "bounded-evidence-visual-v5",
            "required_processing_path": "full_document_visual",
            "model_id": "gpt-5.1-2025-11-13",
            "locked_at": "frozen-v14",
        },
    })
    await db["exampen_paper_versions"].insert_one({
        "paper_version_id": "PAPER-V15",
        "document_id": "DOC-V15",
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
        submission_id = f"SUB-V15-{index}"
        submission = {"submission_id": submission_id, "exam_id": "EXAM-V15"}
        if published and index == 0:
            submission["publication_status"] = "published"
        await db["evalpen_submissions"].insert_one(submission)
        await db["exampen_processing_jobs"].insert_one({
            "job_id": f"JOB-V15-{index}",
            "submission_id": submission_id,
            "exam_id": "EXAM-V15",
            "pipeline_version": 5,
            "mapping_pipeline_version": "bounded-evidence-visual-v5",
            "status": "failed",
            "attempts": 2,
        })
    await db["evalpen_document_grading_runs"].insert_one({
        "run_id": "RUN-V15-OLD",
        "exam_id": "EXAM-V15",
        "prompt_version": "pcr-full-document-visual-v14",
        "pipeline_version": 5,
        "mapping_pipeline_version": "bounded-evidence-visual-v5",
        "status": "completed",
    })


@pytest.mark.asyncio
async def test_v15_migration_is_guarded_and_requeues_exactly_one_job_per_submission():
    assert V15_CAPABILITY_QUEUED_JOB_STATUS in DISPATCHABLE_JOB_STATUSES
    db = _db()
    await _seed(db)

    plans = await inspect_v14_contracts(db, db_name="skb_v15_test", exam_id="EXAM-V15")
    assert plans[0]["eligible"] is True

    with pytest.raises(GradingContractMigrationError, match="confirmation token"):
        await migrate_v14_exam_to_v15(
            db,
            db_name="skb_v15_test",
            exam_id="EXAM-V15",
            requested_by="OPS",
            confirmation_token="WRONG",
        )

    result = await migrate_v14_exam_to_v15(
        db,
        db_name="skb_v15_test",
        exam_id="EXAM-V15",
        requested_by="OPS",
        confirmation_token=V14_TO_V15_CONFIRMATION_TOKEN,
    )
    assert result["status"] == "migrated"
    assert result["queued_job_count"] == 2

    exam = await db["exampen_exams"].find_one({"exam_id": "EXAM-V15"})
    contract = exam["pcr_grading_contract"]
    assert contract["prompt_version"] == "pcr-full-document-visual-v15"
    assert contract["pipeline_version"] == 6
    assert contract["mapping_pipeline_version"] == "bounded-evidence-visual-v6"
    assert contract["required_processing_path"] == "full_document_visual"
    assert contract["migrated_from"] == "pcr-full-document-visual-v14"

    jobs = await db["exampen_processing_jobs"].find({}).to_list(length=None)
    assert len(jobs) == 2
    assert {job["status"] for job in jobs} == {"queued_pipeline_v6"}
    assert {job["pipeline_version"] for job in jobs} == {6}
    assert {job["mapping_pipeline_version"] for job in jobs} == {"bounded-evidence-visual-v6"}
    assert {job["reprocess_count"] for job in jobs} == {1}
    old_run = await db["evalpen_document_grading_runs"].find_one({"run_id": "RUN-V15-OLD"})
    assert old_run["status"] == "superseded"

    again = await migrate_v14_exam_to_v15(
        db,
        db_name="skb_v15_test",
        exam_id="EXAM-V15",
        requested_by="OPS",
        confirmation_token=V14_TO_V15_CONFIRMATION_TOKEN,
    )
    assert again["status"] == "already_migrated"
    jobs = await db["exampen_processing_jobs"].find({}).to_list(length=None)
    assert {job["reprocess_count"] for job in jobs} == {1}


@pytest.mark.asyncio
async def test_v15_inspection_rejects_published_cohort():
    db = _db()
    await _seed(db, published=True)
    plans = await inspect_v14_contracts(db, db_name="skb_v15_test", exam_id="EXAM-V15")
    assert plans[0]["eligible"] is False
    assert "published submission" in " ".join(plans[0]["blockers"])


@pytest.mark.asyncio
async def test_v15_cli_rejects_apply_without_exact_confirmation_before_db(monkeypatch):
    from scripts import migrate_pcr_v14_to_v15 as script

    initialized = False

    async def initialize(_self):
        nonlocal initialized
        initialized = True

    monkeypatch.setattr(script.DatabaseManager, "initialize", initialize)
    args = script.build_parser().parse_args(["--apply", "--confirm", "WRONG"])
    result = await script.run(args)
    assert result == 2
    assert initialized is False


@pytest.mark.asyncio
async def test_v15_cli_correct_confirmation_reaches_mocked_apply(monkeypatch):
    from scripts import migrate_pcr_v14_to_v15 as script

    calls: list[str] = []

    async def initialize(_self):
        calls.append("initialize")

    async def close(_self):
        calls.append("close")

    async def get_tenant_db(_self, _db_name):
        calls.append("db")
        return object()

    async def inspect(_db, *, db_name, exam_id=None):
        calls.append("inspect")
        return [{"db_name": db_name, "exam_id": exam_id, "eligible": True}]

    async def migrate(_db, **kwargs):
        calls.append(f"migrate:{kwargs['confirmation_token']}")
        return {"status": "migrated", "exam_id": kwargs["exam_id"]}

    monkeypatch.setattr(script.DatabaseManager, "initialize", initialize)
    monkeypatch.setattr(script.DatabaseManager, "close", close)
    monkeypatch.setattr(script.DatabaseManager, "get_tenant_db", get_tenant_db)
    monkeypatch.setattr(script, "inspect_v14_contracts", inspect)
    monkeypatch.setattr(script, "migrate_v14_exam_to_v15", migrate)
    args = script.build_parser().parse_args([
        "--tenant-db", "skb_test",
        "--exam-id", "EXAM-V15",
        "--apply",
        "--confirm", script.CONFIRMATION_TOKEN,
    ])
    result = await script.run(args)
    assert result == 0
    assert calls == [
        "initialize",
        "db",
        "inspect",
        "db",
        f"migrate:{script.CONFIRMATION_TOKEN}",
        "close",
    ]


def test_v15_cli_is_dry_run_by_default_and_requires_confirmation():
    from scripts import migrate_pcr_v14_to_v15 as script

    args = script.build_parser().parse_args([])
    assert args.apply is False
    assert args.confirm is None
    assert args.requested_by == "operations:pcr-v14-v15-migration"
