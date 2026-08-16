"""Provider-free contract tests for the guarded legacy subjective -> v16 rollout.

The v16 rollout is a whole-exam operation.  Its public migration surface is
deliberately different from the historical one-step helpers:

``inspect_legacy_contracts``
    inspect a complete cohort and return one plan per eligible exam;
``migrate_legacy_exam_to_v16``
    apply one eligible plan only with the exact confirmation token.

These tests use mongomock and never call a model, queue broker, or network.
They are intentionally written against those names so a missing/renamed hook
fails at the boundary instead of silently exercising the old v15-only path.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import pytest
from mongomock_motor import AsyncMongoMockClient


SOURCE_VERSIONS = (
    "pcr-full-document-visual-v4",
    "pcr-full-document-visual-v5",
    "pcr-full-document-visual-v6",
    "pcr-full-document-visual-v11",
    "pcr-full-document-visual-v12",
    "pcr-full-document-visual-v13",
    "pcr-full-document-visual-v14",
    "pcr-full-document-visual-v15",
)
TARGET_VERSION = "pcr-full-document-visual-v16"
TARGET_PIPELINE = 7
TARGET_MAPPING = "whole-copy-rubric-v7"
TARGET_QUEUE_STATUS = "queued_pipeline_v7"


def _migration_module():
    return import_module("services.pcr_grading_contract_migration")


def _legacy_api():
    module = _migration_module()
    inspect = getattr(module, "inspect_legacy_contracts", None)
    migrate = getattr(module, "migrate_legacy_exam_to_v16", None)
    token = getattr(module, "LEGACY_TO_V16_CONFIRMATION_TOKEN", None)
    if not callable(inspect) or not callable(migrate) or not token:
        pytest.fail(
            "v16 legacy migration API is missing: expected "
            "inspect_legacy_contracts, migrate_legacy_exam_to_v16, and "
            "LEGACY_TO_V16_CONFIRMATION_TOKEN"
        )
    return inspect, migrate, str(token)


def _db(name: str = "skb_legacy_v16_test"):
    return AsyncMongoMockClient()[name]


def _contract(source: str, *, objective: bool = False) -> dict[str, Any]:
    return {
        "prompt_version": source,
        "pipeline_version": 4 if source.endswith("-v4") else 2,
        "mapping_pipeline_version": "evidence-first-visual-v4",
        "required_processing_path": "full_document_visual",
        "model_id": "gpt-5.1-2025-11-13",
        "grading_mode": "objective" if objective else "subjective",
    }


async def _seed_exam(
    db,
    *,
    source: str,
    submissions: int = 3,
    exam_status: str = "draft",
    publication_status: str = "draft",
    published_submission: int | None = None,
    objective: bool = False,
    job_status: str = "failed",
    duplicate_submission: int | None = None,
    missing_submission: int | None = None,
    mixed_submission: int | None = None,
    current_v16: bool = False,
    migration_state: dict[str, Any] | None = None,
) -> None:
    exam_id = "EXAM-LEGACY-V16"
    contract = _contract(source, objective=objective)
    if current_v16:
        contract.update(
            {
                "prompt_version": TARGET_VERSION,
                "pipeline_version": TARGET_PIPELINE,
                "mapping_pipeline_version": TARGET_MAPPING,
                "migrated_from": source,
            }
        )
    await db["exampen_exams"].insert_one(
        {
            "exam_id": exam_id,
            "exam_name": "Legacy Hindi Subjective",
            "exam_type": "pcr",
            "status": exam_status,
            "publication_status": publication_status,
            "prepared_document_id": "DOC-LEGACY-V16",
            "paper_version_id": "PAPER-LEGACY-V16",
            "pcr_grading_contract": contract,
            **(
                {"pcr_grading_contract_migration": migration_state}
                if migration_state is not None
                else {}
            ),
        }
    )
    await db["exampen_paper_versions"].insert_one(
        {
            "paper_version_id": "PAPER-LEGACY-V16",
            "document_id": "DOC-LEGACY-V16",
            "paper_context": {
                "ready": True,
                "version": "canonical-full-document-visual-v2",
                "question_paper_asset_id": "Q-ASSET-LEGACY",
            },
            "paper_assets": {
                "question_paper": {
                    "asset_id": "Q-ASSET-LEGACY",
                    "storage_uri": "s3://papers/legacy-question.pdf",
                }
            },
        }
    )
    for index in range(submissions):
        submission_id = f"SUB-LEGACY-{index}"
        submission: dict[str, Any] = {
            "submission_id": submission_id,
            "exam_id": exam_id,
            "grading_mode": "objective" if objective else "subjective",
        }
        if published_submission == index:
            submission["publication_status"] = "published"
            submission["published_at"] = "2026-08-17T00:00:00Z"
        await db["evalpen_submissions"].insert_one(submission)
        if missing_submission == index:
            continue
        job_pipeline = TARGET_PIPELINE if mixed_submission == index else contract.get("pipeline_version", 4)
        job_mapping = TARGET_MAPPING if mixed_submission == index else contract.get(
            "mapping_pipeline_version", "evidence-first-visual-v4"
        )
        await db["exampen_processing_jobs"].insert_one(
            {
                "job_id": f"JOB-LEGACY-{index}-A",
                "submission_id": submission_id,
                "exam_id": exam_id,
                "pipeline_version": job_pipeline,
                "mapping_pipeline_version": job_mapping,
                "status": job_status,
                "attempts": 2,
            }
        )
        if duplicate_submission == index:
            await db["exampen_processing_jobs"].insert_one(
                {
                    "job_id": f"JOB-LEGACY-{index}-B",
                    "submission_id": submission_id,
                    "exam_id": exam_id,
                    "pipeline_version": contract.get("pipeline_version", 4),
                    "mapping_pipeline_version": contract.get(
                        "mapping_pipeline_version", "evidence-first-visual-v4"
                    ),
                    "status": "failed",
                    "attempts": 1,
                }
            )


def test_v16_legacy_source_allowlist_is_exactly_the_subjective_visual_generations():
    module = _migration_module()
    assert tuple(module.LEGACY_V16_SOURCE_PROMPT_VERSIONS) == SOURCE_VERSIONS


@pytest.mark.asyncio
@pytest.mark.parametrize("source", SOURCE_VERSIONS)
async def test_each_legacy_subjective_source_is_eligible_as_one_whole_cohort(source):
    inspect, _migrate, _token = _legacy_api()
    db = _db(f"skb_legacy_v16_{source[-2:]}")
    await _seed_exam(db, source=source)

    plans = await inspect(db, db_name="legacy-test", exam_id="EXAM-LEGACY-V16")

    assert len(plans) == 1
    assert plans[0]["source_prompt_versions"] == list(SOURCE_VERSIONS)
    assert plans[0]["target_prompt_version"] == TARGET_VERSION
    assert plans[0]["submission_count"] == 3
    assert plans[0]["eligible"] is True
    assert plans[0]["blockers"] == []


@pytest.mark.asyncio
async def test_legacy_migration_queues_exactly_one_v16_job_per_submission_and_forbids_mixing():
    _inspect, migrate, token = _legacy_api()
    db = _db("skb_legacy_v16_apply")
    await _seed_exam(db, source=SOURCE_VERSIONS[0])

    result = await migrate(
        db,
        db_name="legacy-test",
        exam_id="EXAM-LEGACY-V16",
        requested_by="OPS",
        confirmation_token=token,
    )

    assert result["status"] == "migrated"
    assert result["queued_job_count"] == 3
    jobs = await db["exampen_processing_jobs"].find(
        {"exam_id": "EXAM-LEGACY-V16"}
    ).to_list(length=None)
    assert len(jobs) == 3
    assert {job["status"] for job in jobs} == {TARGET_QUEUE_STATUS}
    assert {job["pipeline_version"] for job in jobs} == {TARGET_PIPELINE}
    assert {job["mapping_pipeline_version"] for job in jobs} == {TARGET_MAPPING}
    assert {job["migration_id"] for job in jobs} == {result["migration_id"]}

    exam = await db["exampen_exams"].find_one({"exam_id": "EXAM-LEGACY-V16"})
    assert exam["pcr_grading_contract"]["prompt_version"] == TARGET_VERSION
    assert exam["pcr_grading_contract"]["migrated_from"] == SOURCE_VERSIONS[0]
    assert exam["pcr_grading_contract_migration"]["status"] == "complete"

    # The entire cohort is on one target contract; no source job remains mixed.
    assert len(
        await db["exampen_processing_jobs"].distinct(
            "pipeline_version", {"exam_id": "EXAM-LEGACY-V16"}
        )
    ) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seed_kwargs, blocker",
    [
        ({"published_submission": 0}, "published"),
        ({"exam_status": "active"}, "exam is active"),
        ({"publication_status": "published"}, "exam is published"),
        ({"duplicate_submission": 0}, "duplicate"),
        ({"missing_submission": 0}, "no durable processing job"),
    ],
)
async def test_legacy_inspection_blocks_unsafe_whole_cohort(seed_kwargs, blocker):
    inspect, _migrate, _token = _legacy_api()
    db = _db(f"skb_legacy_v16_block_{blocker[:4]}")
    await _seed_exam(db, source=SOURCE_VERSIONS[1], **seed_kwargs)

    plans = await inspect(db, db_name="legacy-test", exam_id="EXAM-LEGACY-V16")

    assert plans[0]["eligible"] is False
    assert blocker in " ".join(plans[0]["blockers"]).lower()


@pytest.mark.asyncio
async def test_legacy_inspection_blocks_a_cohort_with_mixed_job_contracts():
    inspect, _migrate, _token = _legacy_api()
    db = _db("skb_legacy_v16_block_mixed")
    await _seed_exam(db, source=SOURCE_VERSIONS[2], mixed_submission=1)

    plans = await inspect(db, db_name="legacy-test", exam_id="EXAM-LEGACY-V16")

    assert plans[0]["eligible"] is False
    assert "mixed" in " ".join(plans[0]["blockers"]).lower()


@pytest.mark.asyncio
async def test_legacy_migration_is_idempotent_and_resumes_same_migration_without_duplicate_jobs():
    _inspect, migrate, token = _legacy_api()
    db = _db("skb_legacy_v16_resume")
    await _seed_exam(db, source=SOURCE_VERSIONS[-1], submissions=2)

    first = await migrate(
        db,
        db_name="legacy-test",
        exam_id="EXAM-LEGACY-V16",
        requested_by="OPS",
        confirmation_token=token,
    )
    second = await migrate(
        db,
        db_name="legacy-test",
        exam_id="EXAM-LEGACY-V16",
        requested_by="OPS",
        confirmation_token=token,
    )
    assert second["status"] == "already_migrated"
    assert second["migration_id"] == first["migration_id"]
    jobs = await db["exampen_processing_jobs"].find({}).to_list(length=None)
    assert len(jobs) == 2
    assert {job["reprocess_count"] for job in jobs} == {1}
    assert {len(job["reprocess_history"]) for job in jobs} == {1}

    # A worker/process restart resumes the applying migration identity.
    await db["exampen_processing_jobs"].update_many(
        {"exam_id": "EXAM-LEGACY-V16"}, {"$set": {"status": "failed"}}
    )
    await db["exampen_exams"].update_one(
        {"exam_id": "EXAM-LEGACY-V16"},
        {
            "$set": {
                "pcr_grading_contract_migration.status": "applying",
                "pcr_grading_contract_migration.migration_id": first["migration_id"],
            }
        },
    )
    resumed = await migrate(
        db,
        db_name="legacy-test",
        exam_id="EXAM-LEGACY-V16",
        requested_by="OPS",
        confirmation_token=token,
    )
    assert resumed["migration_id"] == first["migration_id"]
    assert await db["exampen_processing_jobs"].count_documents({}) == 2


@pytest.mark.asyncio
async def test_completed_migration_rerun_drains_pending_job_without_double_charging():
    _inspect, migrate, token = _legacy_api()
    db = _db("skb_legacy_v16_pending_drain")
    await _seed_exam(db, source=SOURCE_VERSIONS[1], submissions=2)
    first = await migrate(
        db,
        db_name="legacy-test",
        exam_id="EXAM-LEGACY-V16",
        requested_by="OPS",
        confirmation_token=token,
    )
    job = await db["exampen_processing_jobs"].find_one(
        {"submission_id": "SUB-LEGACY-0"}
    )
    await db["exampen_processing_jobs"].update_one(
        {"job_id": job["job_id"]},
        {
            "$set": {
                "status": "grading_contract_migration_pending",
                "pipeline_version": 2,
                "mapping_pipeline_version": "evidence-first-visual-v4",
            }
        },
    )

    repaired = await migrate(
        db,
        db_name="legacy-test",
        exam_id="EXAM-LEGACY-V16",
        requested_by="OPS",
        confirmation_token=token,
    )

    assert repaired["status"] == "already_migrated"
    stored = await db["exampen_processing_jobs"].find_one({"job_id": job["job_id"]})
    assert stored["status"] == TARGET_QUEUE_STATUS
    assert stored["pipeline_version"] == TARGET_PIPELINE
    assert stored["mapping_pipeline_version"] == TARGET_MAPPING
    assert stored["reprocess_count"] == 1
    assert len(stored["reprocess_history"]) == 1
    assert stored["migration_id"] == first["migration_id"]


@pytest.mark.asyncio
async def test_resume_rejects_target_job_owned_by_a_different_migration():
    _inspect, migrate, token = _legacy_api()
    db = _db("skb_legacy_v16_foreign_job")
    await _seed_exam(db, source=SOURCE_VERSIONS[1], submissions=1)
    first = await migrate(
        db,
        db_name="legacy-test",
        exam_id="EXAM-LEGACY-V16",
        requested_by="OPS",
        confirmation_token=token,
    )
    await db["exampen_exams"].update_one(
        {"exam_id": "EXAM-LEGACY-V16"},
        {"$set": {"pcr_grading_contract_migration.status": "failed"}},
    )
    await db["exampen_processing_jobs"].update_one(
        {"exam_id": "EXAM-LEGACY-V16"},
        {"$set": {"status": "failed", "migration_id": "PCR-MIG-FOREIGN"}},
    )

    with pytest.raises(
        _migration_module().GradingContractMigrationError,
        match="not migration|different migration",
    ):
        await migrate(
            db,
            db_name="legacy-test",
            exam_id="EXAM-LEGACY-V16",
            requested_by="OPS",
            confirmation_token=token,
        )
    exam = await db["exampen_exams"].find_one({"exam_id": "EXAM-LEGACY-V16"})
    assert exam["pcr_grading_contract_migration"]["migration_id"] == first["migration_id"]


@pytest.mark.asyncio
async def test_resume_preserves_completed_v16_submission_review_state():
    _inspect, migrate, token = _legacy_api()
    db = _db("skb_legacy_v16_completed_resume")
    await _seed_exam(db, source=SOURCE_VERSIONS[1], submissions=2)
    first = await migrate(
        db,
        db_name="legacy-test",
        exam_id="EXAM-LEGACY-V16",
        requested_by="OPS",
        confirmation_token=token,
    )
    await db["exampen_processing_jobs"].update_one(
        {"submission_id": "SUB-LEGACY-0"}, {"$set": {"status": "completed"}}
    )
    await db["evalpen_submissions"].update_one(
        {"submission_id": "SUB-LEGACY-0"}, {"$set": {"review_state": "ready"}}
    )
    await db["exampen_processing_jobs"].update_one(
        {"submission_id": "SUB-LEGACY-1"}, {"$set": {"status": "failed"}}
    )
    await db["exampen_exams"].update_one(
        {"exam_id": "EXAM-LEGACY-V16"},
        {
            "$set": {
                "pcr_grading_contract_migration.status": "applying",
                "pcr_grading_contract_migration.migration_id": first["migration_id"],
            }
        },
    )

    await migrate(
        db,
        db_name="legacy-test",
        exam_id="EXAM-LEGACY-V16",
        requested_by="OPS",
        confirmation_token=token,
    )

    submission = await db["evalpen_submissions"].find_one(
        {"submission_id": "SUB-LEGACY-0"}
    )
    assert submission["review_state"] == "ready"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seed_kwargs, message",
    [
        ({"source": "pcr-full-document-visual-v10"}, "source"),
        ({"source": TARGET_VERSION, "current_v16": True}, "v16|already"),
        ({"source": SOURCE_VERSIONS[0], "objective": True}, "objective"),
    ],
)
async def test_legacy_migration_rejects_unsupported_objective_or_current_v16(seed_kwargs, message):
    _inspect, migrate, token = _legacy_api()
    db = _db(f"skb_legacy_v16_reject_{message}")
    await _seed_exam(db, **seed_kwargs)

    with pytest.raises(_migration_module().GradingContractMigrationError, match=message):
        await migrate(
            db,
            db_name="legacy-test",
            exam_id="EXAM-LEGACY-V16",
            requested_by="OPS",
            confirmation_token=token,
        )


@pytest.mark.asyncio
async def test_wrong_confirmation_does_not_touch_legacy_exam_or_jobs():
    _inspect, migrate, _token = _legacy_api()
    db = _db("skb_legacy_v16_confirmation")
    await _seed_exam(db, source=SOURCE_VERSIONS[2])

    with pytest.raises(_migration_module().GradingContractMigrationError, match="confirmation"):
        await migrate(
            db,
            db_name="legacy-test",
            exam_id="EXAM-LEGACY-V16",
            requested_by="OPS",
            confirmation_token="WRONG",
        )
    exam = await db["exampen_exams"].find_one({"exam_id": "EXAM-LEGACY-V16"})
    assert exam["pcr_grading_contract"]["prompt_version"] == SOURCE_VERSIONS[2]
    assert await db["exampen_grading_contract_migrations"].count_documents({}) == 0


@pytest.mark.asyncio
async def test_legacy_cli_rejects_wrong_confirmation_before_database(monkeypatch):
    from scripts import migrate_pcr_legacy_to_v16 as script

    initialized = False

    async def initialize(_self):
        nonlocal initialized
        initialized = True

    monkeypatch.setattr(script.DatabaseManager, "initialize", initialize)
    args = script.build_parser().parse_args(["--apply", "--confirm", "WRONG"])
    assert await script.run(args) == 2
    assert initialized is False


@pytest.mark.asyncio
async def test_legacy_cli_requires_one_explicit_apply_target_before_database(monkeypatch):
    from scripts import migrate_pcr_legacy_to_v16 as script

    initialized = False

    async def initialize(_self):
        nonlocal initialized
        initialized = True

    monkeypatch.setattr(script.DatabaseManager, "initialize", initialize)
    args = script.build_parser().parse_args(
        ["--apply", "--confirm", script.CONFIRMATION_TOKEN]
    )
    assert await script.run(args) == 2
    assert initialized is False


@pytest.mark.asyncio
async def test_legacy_cli_apply_can_resume_when_legacy_inspection_is_empty(monkeypatch):
    from scripts import migrate_pcr_legacy_to_v16 as script

    tenant_db = object()
    calls: list[dict[str, Any]] = []

    class _Manager:
        async def initialize(self):
            return None

        async def get_tenant_db(self, db_name):
            assert db_name == "skb_resume"
            return tenant_db

        async def close(self):
            return None

    async def inspect(*_args, **_kwargs):
        return []

    async def migrate(db, **kwargs):
        assert db is tenant_db
        calls.append(kwargs)
        return {"status": "already_migrated", "migration_id": "PCR-MIG-1"}

    monkeypatch.setattr(script, "DatabaseManager", _Manager)
    monkeypatch.setattr(script, "inspect_legacy_contracts", inspect)
    monkeypatch.setattr(script, "migrate_legacy_exam_to_v16", migrate)
    args = script.build_parser().parse_args(
        [
            "--tenant-db",
            "skb_resume",
            "--exam-id",
            "EXAM-1",
            "--apply",
            "--confirm",
            script.CONFIRMATION_TOKEN,
        ]
    )

    assert await script.run(args) == 0
    assert calls == [
        {
            "db_name": "skb_resume",
            "exam_id": "EXAM-1",
            "requested_by": "operations:pcr-legacy-v16-migration",
            "confirmation_token": script.CONFIRMATION_TOKEN,
        }
    ]


@pytest.mark.asyncio
async def test_fleet_cli_rejects_wrong_confirmation_before_database(monkeypatch):
    from scripts import migrate_pcr_legacy_to_v16 as script

    initialized = False

    async def initialize(_self):
        nonlocal initialized
        initialized = True

    monkeypatch.setattr(script.DatabaseManager, "initialize", initialize)
    args = script.build_parser().parse_args(
        ["--apply-eligible", "--confirm", script.CONFIRMATION_TOKEN]
    )

    assert await script.run(args) == 2
    assert initialized is False


@pytest.mark.asyncio
async def test_fleet_cli_discovers_tenants_and_applies_only_the_bounded_eligible_batch(
    monkeypatch,
):
    from scripts import migrate_pcr_legacy_to_v16 as script

    databases = {"skb_a": object(), "skb_b": object()}
    migrated: list[tuple[str, str]] = []

    class _Manager:
        async def initialize(self):
            return None

        async def get_tenant_db(self, db_name):
            return databases.get(db_name)

        async def close(self):
            return None

    async def tenant_names(_manager):
        return ["skb_a", "skb_b"]

    async def inspect(db, *, db_name, exam_id):
        assert db is databases[db_name]
        assert exam_id is None
        if db_name == "skb_a":
            return [
                {
                    "db_name": db_name,
                    "exam_id": "EXAM-1",
                    "submission_count": 2,
                    "eligible": True,
                    "blockers": [],
                },
                {
                    "db_name": db_name,
                    "exam_id": "EXAM-2",
                    "submission_count": 5,
                    "eligible": True,
                    "blockers": [],
                },
                {
                    "db_name": db_name,
                    "exam_id": "EXAM-PUBLISHED",
                    "submission_count": 1,
                    "eligible": False,
                    "blockers": ["published"],
                },
            ]
        return [
            {
                "db_name": db_name,
                "exam_id": "EXAM-3",
                "submission_count": 3,
                "eligible": True,
                "blockers": [],
            }
        ]

    async def migrate(db, *, db_name, exam_id, **kwargs):
        assert db is databases[db_name]
        assert kwargs["confirmation_token"] == script.CONFIRMATION_TOKEN
        migrated.append((db_name, exam_id))
        return {"db_name": db_name, "exam_id": exam_id, "status": "migrated"}

    monkeypatch.setattr(script, "DatabaseManager", _Manager)
    monkeypatch.setattr(script, "_tenant_names", tenant_names)
    monkeypatch.setattr(script, "inspect_legacy_contracts", inspect)
    monkeypatch.setattr(script, "migrate_legacy_exam_to_v16", migrate)
    args = script.build_parser().parse_args(
        [
            "--apply-eligible",
            "--max-exams",
            "2",
            "--max-submissions",
            "5",
            "--confirm",
            script.FLEET_CONFIRMATION_TOKEN,
        ]
    )

    assert await script.run(args) == 0
    assert migrated == [("skb_a", "EXAM-1"), ("skb_b", "EXAM-3")]


@pytest.mark.asyncio
async def test_fleet_cli_continues_safe_tenants_but_returns_nonzero_on_scan_error(
    monkeypatch,
):
    from scripts import migrate_pcr_legacy_to_v16 as script

    good_db = object()
    migrated: list[str] = []

    class _Manager:
        async def initialize(self):
            return None

        async def get_tenant_db(self, db_name):
            return good_db if db_name == "skb_good" else None

        async def close(self):
            return None

    async def tenant_names(_manager):
        return ["skb_broken", "skb_good"]

    async def inspect(db, *, db_name, exam_id):
        assert db is good_db
        assert exam_id is None
        return [
            {
                "db_name": db_name,
                "exam_id": "EXAM-GOOD",
                "submission_count": 1,
                "eligible": True,
                "blockers": [],
            }
        ]

    async def migrate(_db, *, exam_id, **_kwargs):
        migrated.append(exam_id)
        return {"exam_id": exam_id, "status": "migrated"}

    monkeypatch.setattr(script, "DatabaseManager", _Manager)
    monkeypatch.setattr(script, "_tenant_names", tenant_names)
    monkeypatch.setattr(script, "inspect_legacy_contracts", inspect)
    monkeypatch.setattr(script, "migrate_legacy_exam_to_v16", migrate)
    args = script.build_parser().parse_args(
        [
            "--apply-eligible",
            "--confirm",
            script.FLEET_CONFIRMATION_TOKEN,
        ]
    )

    assert await script.run(args) == 4
    assert migrated == ["EXAM-GOOD"]


@pytest.mark.asyncio
async def test_worker_pauses_v16_job_while_migration_is_failed():
    from services.exampen_workflow import process_pcr_processing_job

    db = _db("skb_legacy_v16_worker_fence")
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "EXAM-WORKER-FENCE",
            "exam_type": "pcr",
            "pcr_grading_contract": {
                "prompt_version": TARGET_VERSION,
                "pipeline_version": TARGET_PIPELINE,
                "mapping_pipeline_version": TARGET_MAPPING,
            },
            "pcr_grading_contract_migration": {
                "status": "failed",
                "migration_id": "PCR-MIG-FENCE",
            },
        }
    )
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-WORKER-FENCE",
            "exam_id": "EXAM-WORKER-FENCE",
            "student_id": "STU-1",
        }
    )
    await db["exampen_processing_jobs"].insert_one(
        {
            "job_id": "pcr-job-SUB-WORKER-FENCE",
            "submission_id": "SUB-WORKER-FENCE",
            "exam_id": "EXAM-WORKER-FENCE",
            "student_id": "STU-1",
            "status": TARGET_QUEUE_STATUS,
            "pipeline_version": TARGET_PIPELINE,
            "mapping_pipeline_version": TARGET_MAPPING,
            "attempts": 0,
        }
    )

    result = await process_pcr_processing_job(db, "pcr-job-SUB-WORKER-FENCE")

    assert result["status"] == "grading_contract_migration_pending"
    stored = await db["exampen_processing_jobs"].find_one(
        {"job_id": "pcr-job-SUB-WORKER-FENCE"}
    )
    assert stored["status"] == "grading_contract_migration_pending"
    assert stored["migration_id"] == "PCR-MIG-FENCE"
    assert "lease_token" not in stored


def test_reprocess_api_guard_returns_stable_migration_required_conflict():
    from fastapi import HTTPException

    from api.v1.exam_orch_async import _require_reprocessable_grading_contract

    with pytest.raises(HTTPException) as exc_info:
        _require_reprocessable_grading_contract(
            {
                "pcr_grading_contract": {
                    "prompt_version": "pcr-full-document-visual-v5"
                }
            }
        )
    assert exc_info.value.status_code == 409
    assert "grading_contract_migration_required" in str(exc_info.value.detail)

    with pytest.raises(HTTPException) as failed_migration:
        _require_reprocessable_grading_contract(
            {
                "pcr_grading_contract": {
                    "prompt_version": TARGET_VERSION,
                    "pipeline_version": TARGET_PIPELINE,
                },
                "pcr_grading_contract_migration": {"status": "failed"},
            }
        )
    assert failed_migration.value.status_code == 409
    assert "grading_contract_migration_incomplete" in str(
        failed_migration.value.detail
    )

    _require_reprocessable_grading_contract(
        {
            "pcr_grading_contract": {
                "prompt_version": TARGET_VERSION,
                "pipeline_version": TARGET_PIPELINE,
            },
            "pcr_grading_contract_migration": {"status": "complete"},
        }
    )
