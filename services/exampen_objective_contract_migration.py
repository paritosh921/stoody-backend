"""Guarded migration for PCR cohorts finalized under the wrong grading lane.

This module never guesses at request time. It creates a new immutable paper
snapshot and switches the entire unpublished cohort together. Existing legacy
objective runs are superseded and reprocessed as one revision; previous paper,
response, evaluation, and grading-run records remain available for audit.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.exampen_paper_service import (
    create_paper_snapshot,
    full_document_visual_contract,
)
from services.exampen_workflow import (
    CURRENT_PCR_PIPELINE_VERSION,
    FULL_DOCUMENT_PROCESSING_PATH,
)
from services.objective_answer_ledger_contract import (
    LEGACY_OBJECTIVE_PROMPT_VERSIONS,
    OBJECTIVE_PAPER_CONTEXT_VERSION,
    OBJECTIVE_PROMPT_VERSION,
    all_questions_are_objective,
)


class ObjectiveContractMigrationError(RuntimeError):
    """Raised when an exam cannot be migrated without changing graded work."""


async def plan_objective_contract_migration(
    tenant_db: Any,
    exam_id: str,
) -> Dict[str, Any]:
    """Return a fail-closed migration plan without mutating the database."""

    exam = await tenant_db["exampen_exams"].find_one({"exam_id": exam_id})
    if not exam:
        raise ObjectiveContractMigrationError(f"Exam {exam_id} was not found")
    if str(exam.get("exam_type") or "") != "pcr":
        raise ObjectiveContractMigrationError("Only PCR exams can be migrated")

    current_version_id = str(exam.get("paper_version_id") or "")
    if not current_version_id:
        raise ObjectiveContractMigrationError(
            "Exam has no immutable paper version"
        )
    current_version = await tenant_db["exampen_paper_versions"].find_one(
        {"paper_version_id": current_version_id}
    )
    if not current_version:
        raise ObjectiveContractMigrationError(
            "The current immutable paper version is missing"
        )
    current_context = dict(current_version.get("paper_context") or {})
    if (
        current_context.get("ready")
        and str(current_context.get("version") or "")
        == OBJECTIVE_PAPER_CONTEXT_VERSION
    ):
        return {
            "exam_id": exam_id,
            "eligible": True,
            "already_migrated": True,
            "current_paper_version_id": current_version_id,
            "target_context": current_context,
            "submission_ids": [],
        }

    current_contract = dict(exam.get("pcr_grading_contract") or {})
    contract_prompt_version = str(
        current_contract.get("prompt_version") or ""
    )
    if not current_contract:
        migration_mode = "ungraded_cohort"
    elif contract_prompt_version == OBJECTIVE_PROMPT_VERSION:
        # A deployment may have completed the objective strategy while its
        # paper pointer still referenced the preceding canonical version.
        # Aligning metadata to the contract actually used does not regrade or
        # alter any materialized score.
        migration_mode = "align_existing_objective_contract"
    elif contract_prompt_version in LEGACY_OBJECTIVE_PROMPT_VERSIONS:
        migration_mode = "upgrade_legacy_objective_contract"
    else:
        raise ObjectiveContractMigrationError(
            "The cohort is frozen to a different grading contract"
        )

    questions = await tenant_db["evalpen_questions"].find(
        {"exam_id": exam_id}
    ).sort("question_number", 1).to_list(length=2000)
    if not questions or not all_questions_are_objective(questions):
        raise ObjectiveContractMigrationError(
            "The complete immutable question catalog is not objective"
        )

    submissions = await tenant_db["evalpen_submissions"].find(
        {"exam_id": exam_id},
        {"submission_id": 1, "publication_status": 1},
    ).to_list(length=5000)
    if any(
        str(item.get("publication_status") or "") == "published"
        for item in submissions
    ):
        raise ObjectiveContractMigrationError(
            "At least one answer copy has already been published"
        )
    submission_ids = [
        str(item.get("submission_id") or "")
        for item in submissions
        if item.get("submission_id")
    ]

    if submission_ids:
        active_job_count = await tenant_db[
            "exampen_processing_jobs"
        ].count_documents(
            {
                "submission_id": {"$in": submission_ids},
                "status": "processing",
            }
        )
        if active_job_count:
            raise ObjectiveContractMigrationError(
                "A submission worker is active; stop workers before migration"
            )
        completed_runs = await tenant_db[
            "evalpen_document_grading_runs"
        ].find(
            {
                "submission_id": {"$in": submission_ids},
                "status": {
                    "$in": ["validated", "materializing", "completed"]
                },
            },
            {"prompt_version": 1},
        ).to_list(length=5000)
        if completed_runs and migration_mode == "ungraded_cohort":
            raise ObjectiveContractMigrationError(
                "At least one submission already has a validated grading ledger"
            )
        allowed_completed_prompt_versions = (
            LEGACY_OBJECTIVE_PROMPT_VERSIONS
            if migration_mode == "upgrade_legacy_objective_contract"
            else {OBJECTIVE_PROMPT_VERSION}
        )
        if any(
            str(run.get("prompt_version") or "")
            not in allowed_completed_prompt_versions
            for run in completed_runs
        ):
            raise ObjectiveContractMigrationError(
                "A validated grading ledger uses a non-objective contract"
            )
        response_count = await tenant_db[
            "evalpen_detected_responses"
        ].count_documents({"submission_id": {"$in": submission_ids}})
        if response_count and migration_mode == "ungraded_cohort":
            raise ObjectiveContractMigrationError(
                "At least one submission already has materialized answer rows"
            )

    document_id = str(
        exam.get("prepared_document_id")
        or current_version.get("document_id")
        or ""
    )
    document = await tenant_db["documents"].find_one(
        {"document_id": document_id}
    )
    if not document:
        raise ObjectiveContractMigrationError(
            "The canonical source document is missing"
        )
    target_context = full_document_visual_contract(document, questions)
    if (
        not target_context.get("ready")
        or str(target_context.get("version") or "")
        != OBJECTIVE_PAPER_CONTEXT_VERSION
    ):
        blockers = "; ".join(target_context.get("blockers") or [])
        raise ObjectiveContractMigrationError(
            "Objective grading contract is not ready"
            + (f": {blockers}" if blockers else "")
        )

    return {
        "exam_id": exam_id,
        "eligible": True,
        "already_migrated": False,
        "current_paper_version_id": current_version_id,
        "current_paper_version": current_version,
        "document": document,
        "questions": questions,
        "submission_ids": submission_ids,
        "migration_mode": migration_mode,
        "current_grading_contract": current_contract,
        "target_context": target_context,
    }


async def migrate_objective_contract(
    tenant_db: Any,
    exam_id: str,
) -> Dict[str, Any]:
    """Create and atomically activate a new objective paper version."""

    plan = await plan_objective_contract_migration(tenant_db, exam_id)
    if plan["already_migrated"]:
        client = getattr(tenant_db, "client", None)
        if client is None or not hasattr(client, "start_session"):
            raise ObjectiveContractMigrationError(
                "MongoDB transactions are required for contract alignment"
            )
        async with await client.start_session() as session:
            async with session.start_transaction():
                await _align_objective_processing_metadata(
                    tenant_db,
                    exam_id=exam_id,
                    paper_version_id=str(
                        plan["current_paper_version_id"]
                    ),
                    session=session,
                )
        result = _public_result(plan)
        result["processing_metadata_aligned"] = True
        return result

    document = dict(plan["document"])
    document["exam_mode"] = "pcr"
    current_version = dict(plan["current_paper_version"])
    new_version = await create_paper_snapshot(
        tenant_db,
        document,
        list(plan["questions"]),
        question_layout=list(current_version.get("question_layout") or []),
        paper_context=dict(plan["target_context"]),
    )
    new_version_id = str(new_version.get("paper_version_id") or "")
    if not new_version_id:
        raise ObjectiveContractMigrationError(
            "The new immutable paper snapshot was not committed"
        )
    if new_version_id == plan["current_paper_version_id"]:
        raise ObjectiveContractMigrationError(
            "Objective migration did not produce a distinct paper version"
        )

    client = getattr(tenant_db, "client", None)
    if client is None or not hasattr(client, "start_session"):
        raise ObjectiveContractMigrationError(
            "MongoDB transactions are required for contract migration"
        )
    async with await client.start_session() as session:
        async with session.start_transaction():
            await _commit_objective_contract_migration(
                tenant_db,
                plan=plan,
                new_version_id=new_version_id,
                session=session,
            )

    result = _public_result(plan)
    result.update(
        {
            "already_migrated": False,
            "new_paper_version_id": new_version_id,
            "migrated": True,
        }
    )
    return result


async def _commit_objective_contract_migration(
    tenant_db: Any,
    *,
    plan: Dict[str, Any],
    new_version_id: str,
    session: Optional[Any],
) -> None:
    """Commit the pointer switch; exposed internally for transaction tests."""

    now = datetime.now(timezone.utc)
    exam_id = str(plan["exam_id"])
    old_version_id = str(plan["current_paper_version_id"])
    submission_ids: List[str] = list(plan.get("submission_ids") or [])
    migration_mode = str(plan.get("migration_mode") or "ungraded_cohort")
    contract_guard: Dict[str, Any]
    if migration_mode == "align_existing_objective_contract":
        contract_guard = {
            "pcr_grading_contract.prompt_version": OBJECTIVE_PROMPT_VERSION
        }
    elif migration_mode == "upgrade_legacy_objective_contract":
        contract_guard = {
            "pcr_grading_contract.prompt_version": {
                "$in": list(LEGACY_OBJECTIVE_PROMPT_VERSIONS)
            }
        }
    else:
        contract_guard = {
            "$or": [
                {"pcr_grading_contract": {"$exists": False}},
                {"pcr_grading_contract": {}},
            ]
        }

    exam_updates: Dict[str, Any] = {
        "paper_version_id": new_version_id,
        "grading_contract_migrated_at": now,
        "grading_contract_migrated_from": old_version_id,
        "updated_at": now,
    }
    if migration_mode == "upgrade_legacy_objective_contract":
        exam_updates["pcr_grading_contract"] = {
            "prompt_version": OBJECTIVE_PROMPT_VERSION,
            "model_id": str(
                (plan.get("target_context") or {}).get("model_id")
                or "gpt-5.6-sol"
            ),
            "temperature": 0.1,
            "reasoning_effort": "medium",
            "locked_at": now,
            "migrated_from_prompt_version": str(
                (plan.get("current_grading_contract") or {}).get(
                    "prompt_version"
                )
                or next(iter(LEGACY_OBJECTIVE_PROMPT_VERSIONS))
            ),
        }

    switched = await tenant_db["exampen_exams"].update_one(
        {
            "exam_id": exam_id,
            "paper_version_id": old_version_id,
            **contract_guard,
        },
        {
            "$set": exam_updates
        },
        session=session,
    )
    if switched.matched_count != 1:
        raise ObjectiveContractMigrationError(
            "Exam contract changed while migration was committing"
        )

    await tenant_db["evalpen_questions"].update_many(
        {"exam_id": exam_id, "paper_version_id": old_version_id},
        {
            "$set": {
                "paper_version_id": new_version_id,
                "updated_at": now,
            }
        },
        session=session,
    )
    await tenant_db["documents"].update_one(
        {
            "document_id": plan["document"].get("document_id"),
            "exam_paper_version_id": old_version_id,
        },
        {
            "$set": {
                "exam_paper_version_id": new_version_id,
                "updated_at": now,
            }
        },
        session=session,
    )

    if submission_ids:
        run_filter: Dict[str, Any] = {
            "submission_id": {"$in": submission_ids},
            "status": {"$in": ["generating", "failed"]},
            "prompt_version": {"$ne": OBJECTIVE_PROMPT_VERSION},
        }
        if migration_mode == "upgrade_legacy_objective_contract":
            run_filter = {
                "submission_id": {"$in": submission_ids},
                "prompt_version": {
                    "$in": list(LEGACY_OBJECTIVE_PROMPT_VERSIONS)
                },
                "status": {"$ne": "superseded"},
            }
        await tenant_db["evalpen_document_grading_runs"].update_many(
            run_filter,
            {
                "$set": {
                    "status": "superseded",
                    "superseded_at": now,
                    "superseded_by_paper_version_id": new_version_id,
                    "updated_at": now,
                },
                "$unset": {
                    "generation_lease_token": "",
                    "generation_lease_expires_at": "",
                },
            },
            session=session,
        )
        await _align_objective_processing_metadata(
            tenant_db,
            exam_id=exam_id,
            paper_version_id=new_version_id,
            session=session,
        )
        if migration_mode in {
            "ungraded_cohort",
            "upgrade_legacy_objective_contract",
        }:
            job_update: Dict[str, Any] = {
                "$set": {
                    "status": "queued",
                    "pipeline_version": CURRENT_PCR_PIPELINE_VERSION,
                    "required_processing_path": FULL_DOCUMENT_PROCESSING_PATH,
                    "processing_path": "objective_answer_ledger",
                    "attempts": 0,
                    "last_error": None,
                    "contract_migrated_at": now,
                    "updated_at": now,
                },
                "$unset": {
                    "finished_at": "",
                    "started_at": "",
                    "lease_token": "",
                    "lease_expires_at": "",
                    "next_attempt_at": "",
                    "failure_code": "",
                },
            }
            if migration_mode == "upgrade_legacy_objective_contract":
                job_update["$inc"] = {
                    "reprocess_count": 1,
                    "generation_revision": 1,
                }
            await tenant_db["exampen_processing_jobs"].update_many(
                {
                    "submission_id": {"$in": submission_ids},
                    "status": {"$ne": "processing"},
                },
                job_update,
                session=session,
            )

    await tenant_db["exampen_contract_migrations"].insert_one(
        {
            "migration_id": (
                f"objective-contract:{exam_id}:{old_version_id}:{new_version_id}"
            ),
            "exam_id": exam_id,
            "from_paper_version_id": old_version_id,
            "to_paper_version_id": new_version_id,
            "target_contract_version": OBJECTIVE_PAPER_CONTEXT_VERSION,
            "migration_mode": migration_mode,
            "submission_ids": submission_ids,
            "created_at": now,
        },
        session=session,
    )


async def _align_objective_processing_metadata(
    tenant_db: Any,
    *,
    exam_id: str,
    paper_version_id: str,
    session: Optional[Any],
) -> None:
    """Make operational metadata describe the immutable objective strategy."""

    now = datetime.now(timezone.utc)
    submissions = await tenant_db["evalpen_submissions"].find(
        {"exam_id": exam_id},
        {"submission_id": 1},
        session=session,
    ).to_list(length=5000)
    submission_ids = [
        str(item.get("submission_id") or "")
        for item in submissions
        if item.get("submission_id")
    ]
    await tenant_db["evalpen_submissions"].update_many(
        {"exam_id": exam_id},
        {
            "$set": {
                "processing_path": "objective_answer_ledger",
                "paper_version_id": paper_version_id,
                "updated_at": now,
            }
        },
        session=session,
    )
    if not submission_ids:
        return
    await tenant_db["exampen_processing_jobs"].update_many(
        {
            "submission_id": {"$in": submission_ids},
            "status": {"$ne": "processing"},
        },
        {
            "$set": {
                "processing_path": "objective_answer_ledger",
                "segmentation.path": "objective_answer_ledger",
                "evaluation.path": "objective_answer_ledger",
                "contract_migrated_at": now,
                "updated_at": now,
            }
        },
        session=session,
    )
    completed_jobs_with_legacy_errors = await tenant_db[
        "exampen_processing_jobs"
    ].find(
        {
            "submission_id": {"$in": submission_ids},
            "status": "completed",
            "last_error": {"$type": "string", "$ne": ""},
        },
        {"job_id": 1, "last_error": 1},
        session=session,
    ).to_list(length=5000)
    for job in completed_jobs_with_legacy_errors:
        await tenant_db["exampen_processing_jobs"].update_one(
            {
                "job_id": job.get("job_id"),
                "status": "completed",
                "last_error": job.get("last_error"),
            },
            {
                "$set": {
                    "diagnostics": {
                        "errors": [str(job.get("last_error"))],
                        "recorded_at": now,
                        "source": "objective_contract_alignment",
                    },
                    "last_error": None,
                    "updated_at": now,
                },
                "$unset": {
                    "failure_code": "",
                    "next_attempt_at": "",
                },
            },
            session=session,
        )
    await tenant_db["evalpen_evaluations"].update_many(
        {
            "exam_id": exam_id,
            "eval_path": {"$regex": "not_attempted$"},
        },
        {"$set": {"eval_path": "objective_answer_ledger_not_attempted"}},
        session=session,
    )
    await tenant_db["evalpen_evaluations"].update_many(
        {
            "exam_id": exam_id,
            "eval_path": {"$not": {"$regex": "not_attempted$"}},
        },
        {"$set": {"eval_path": "objective_answer_ledger"}},
        session=session,
    )


def _public_result(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "exam_id": plan["exam_id"],
        "eligible": bool(plan["eligible"]),
        "already_migrated": bool(plan["already_migrated"]),
        "current_paper_version_id": plan["current_paper_version_id"],
        "submission_count": len(plan.get("submission_ids") or []),
        "migration_mode": str(plan.get("migration_mode") or ""),
        "target_contract_version": str(
            (plan.get("target_context") or {}).get("version") or ""
        ),
    }
