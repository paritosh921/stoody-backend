from __future__ import annotations

from mongomock_motor import AsyncMongoMockClient
import pytest

from services.exampen_objective_contract_migration import (
    ObjectiveContractMigrationError,
    _commit_objective_contract_migration,
    plan_objective_contract_migration,
)


def _db():
    return AsyncMongoMockClient()["skb_test"]


async def _seed(db):
    exam_id = "EXAM-OBJECTIVE-LEGACY"
    paper_version_id = "paper-legacy-visual"
    await db["exampen_exams"].insert_one(
        {
            "exam_id": exam_id,
            "exam_type": "pcr",
            "paper_version_id": paper_version_id,
            "prepared_document_id": "DOC-OBJECTIVE",
            "lifecycle_state": "in_progress",
        }
    )
    await db["exampen_paper_versions"].insert_one(
        {
            "paper_version_id": paper_version_id,
            "document_id": "DOC-OBJECTIVE",
            "paper_context": {
                "version": "canonical-full-document-visual-v2",
                "mode": "full_document_visual",
                "ready": True,
            },
            "question_layout": [],
        }
    )
    await db["documents"].insert_one(
        {
            "document_id": "DOC-OBJECTIVE",
            "exam_mode": "pcr",
            "file_path": "data/private/question.pdf",
            "sha256": "paper-sha",
            "exam_paper_version_id": paper_version_id,
        }
    )
    await db["evalpen_questions"].insert_one(
        {
            "question_id": f"{exam_id}::Q1",
            "source_question_id": "Q1",
            "exam_id": exam_id,
            "paper_version_id": paper_version_id,
            "question_number": 1,
            "question_text": "Choose one.",
            "question_type": "mcq",
            "grading_mode": "objective",
            "max_marks": 4,
            "penalty_marks": 1,
            "options": [
                {"label": "A", "text": "One"},
                {"label": "B", "text": "Two"},
            ],
            "correct_answer": "B",
        }
    )
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-1",
            "exam_id": exam_id,
            "publication_status": "draft",
        }
    )
    await db["exampen_processing_jobs"].insert_one(
        {
            "job_id": "pcr-job-SUB-1",
            "submission_id": "SUB-1",
            "exam_id": exam_id,
            "status": "failed",
            "attempts": 7,
            "last_error": "old worker failed",
        }
    )
    return exam_id


@pytest.mark.asyncio
async def test_migration_plan_is_read_only_and_selects_objective_contract(
    monkeypatch,
):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("PCR_FULL_DOCUMENT_GRADING_ENABLED", "true")
    monkeypatch.setenv("PCR_FULL_DOCUMENT_GRADING_MODEL", "gpt-5.1")
    db = _db()
    exam_id = await _seed(db)

    plan = await plan_objective_contract_migration(db, exam_id)

    assert plan["eligible"] is True
    assert plan["already_migrated"] is False
    assert plan["migration_mode"] == "ungraded_cohort"
    assert plan["target_context"]["version"] == "objective-answer-ledger-v3"
    assert plan["target_context"]["model_id"] == "gpt-5.6-sol"
    unchanged = await db["exampen_exams"].find_one({"exam_id": exam_id})
    assert unchanged["paper_version_id"] == "paper-legacy-visual"


@pytest.mark.asyncio
async def test_migration_commit_switches_whole_cohort_and_preserves_audit(
    monkeypatch,
):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("PCR_FULL_DOCUMENT_GRADING_ENABLED", "true")
    monkeypatch.setenv("PCR_FULL_DOCUMENT_GRADING_MODEL", "gpt-5.1")
    db = _db()
    exam_id = await _seed(db)
    plan = await plan_objective_contract_migration(db, exam_id)
    await db["evalpen_document_grading_runs"].insert_one(
        {
            "run_id": "RUN-OLD",
            "submission_id": "SUB-1",
            "status": "failed",
            "prompt_version": "pcr-full-document-visual-v5",
        }
    )

    await _commit_objective_contract_migration(
        db,
        plan=plan,
        new_version_id="paper-objective-v1",
        session=None,
    )

    exam = await db["exampen_exams"].find_one({"exam_id": exam_id})
    question = await db["evalpen_questions"].find_one({"exam_id": exam_id})
    job = await db["exampen_processing_jobs"].find_one(
        {"submission_id": "SUB-1"}
    )
    old_run = await db["evalpen_document_grading_runs"].find_one(
        {"run_id": "RUN-OLD"}
    )
    migration = await db["exampen_contract_migrations"].find_one(
        {"exam_id": exam_id}
    )

    assert exam["paper_version_id"] == "paper-objective-v1"
    assert question["paper_version_id"] == "paper-objective-v1"
    assert job["status"] == "queued"
    assert job["attempts"] == 0
    assert old_run["status"] == "superseded"
    assert migration["from_paper_version_id"] == "paper-legacy-visual"
    assert migration["to_paper_version_id"] == "paper-objective-v1"


@pytest.mark.asyncio
async def test_migration_refuses_existing_materialized_answers(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("PCR_FULL_DOCUMENT_GRADING_ENABLED", "true")
    monkeypatch.setenv("PCR_FULL_DOCUMENT_GRADING_MODEL", "gpt-5.1")
    db = _db()
    exam_id = await _seed(db)
    await db["evalpen_detected_responses"].insert_one(
        {
            "response_id": "RESP-1",
            "submission_id": "SUB-1",
        }
    )

    with pytest.raises(
        ObjectiveContractMigrationError,
        match="materialized answer rows",
    ):
        await plan_objective_contract_migration(db, exam_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "legacy_prompt_version",
    [
        "pcr-objective-answer-ledger-v1",
        "pcr-objective-answer-ledger-v2",
    ],
)
async def test_migration_upgrades_legacy_objective_contract_and_requeues_cohort(
    monkeypatch,
    legacy_prompt_version,
):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("PCR_FULL_DOCUMENT_GRADING_ENABLED", "true")
    monkeypatch.setenv("PCR_FULL_DOCUMENT_GRADING_MODEL", "gpt-5.1")
    db = _db()
    exam_id = await _seed(db)
    await db["exampen_exams"].update_one(
        {"exam_id": exam_id},
        {
            "$set": {
                "pcr_grading_contract": {
                    "prompt_version": legacy_prompt_version,
                    "model_id": "gpt-5.1-2025-11-13",
                }
            }
        },
    )
    await db["evalpen_document_grading_runs"].insert_one(
        {
            "run_id": "RUN-OBJECTIVE",
            "submission_id": "SUB-1",
            "status": "completed",
            "prompt_version": legacy_prompt_version,
        }
    )
    await db["evalpen_detected_responses"].insert_one(
        {
            "response_id": "RESP-OBJECTIVE",
            "submission_id": "SUB-1",
        }
    )
    await db["exampen_processing_jobs"].update_one(
        {"submission_id": "SUB-1"},
        {"$set": {"status": "completed", "attempts": 6}},
    )

    plan = await plan_objective_contract_migration(db, exam_id)
    assert plan["migration_mode"] == "upgrade_legacy_objective_contract"

    await _commit_objective_contract_migration(
        db,
        plan=plan,
        new_version_id="paper-objective-aligned",
        session=None,
    )

    job = await db["exampen_processing_jobs"].find_one(
        {"submission_id": "SUB-1"}
    )
    run = await db["evalpen_document_grading_runs"].find_one(
        {"run_id": "RUN-OBJECTIVE"}
    )
    exam = await db["exampen_exams"].find_one({"exam_id": exam_id})
    assert job["status"] == "queued"
    assert job["attempts"] == 0
    assert job["reprocess_count"] == 1
    assert job["generation_revision"] == 1
    assert job["processing_path"] == "objective_answer_ledger"
    assert job["evaluation"]["path"] == "objective_answer_ledger"
    assert job["last_error"] is None
    assert job["diagnostics"]["source"] == "objective_contract_alignment"
    assert run["status"] == "superseded"
    assert (
        exam["pcr_grading_contract"]["prompt_version"]
        == "pcr-objective-answer-ledger-v3"
    )
    assert exam["pcr_grading_contract"]["model_id"] == "gpt-5.6-sol"
    assert (
        exam["pcr_grading_contract"]["migrated_from_prompt_version"]
        == legacy_prompt_version
    )
