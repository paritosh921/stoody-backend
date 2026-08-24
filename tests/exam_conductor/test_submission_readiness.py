from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException


def _fresh_db():
    from mongomock_motor import AsyncMongoMockClient

    return AsyncMongoMockClient()["skb_test"]


def _admin_user():
    return {
        "user_id": "TEACHER-1",
        "user_type": "admin",
        "admin_id": "ADMIN-1",
        "db_name": "skb_test",
    }


def test_pcr_marking_plan_source_requires_explicit_review():
    from api.v1.pdf_async import _is_reviewed_pcr_answer_mapping

    base = {"answer_text": "Worked solution", "manual_review_required": False}
    assert _is_reviewed_pcr_answer_mapping({**base, "review_status": "accepted"})
    assert _is_reviewed_pcr_answer_mapping({**base, "review_status": "trusted"})
    assert not _is_reviewed_pcr_answer_mapping({**base, "review_status": "needs_review"})
    assert not _is_reviewed_pcr_answer_mapping(
        {**base, "review_status": "accepted", "manual_review_required": True}
    )


async def _seed_ready_submission(db, *, proven_blank: bool = True, include_q2: bool = True):
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-READY",
            "exam_id": "EXAM-READY",
            "student_id": "STU-1",
            "segmentation_status": "complete",
        }
    )
    await db["exampen_processing_jobs"].insert_one(
        {
            "job_id": "pcr-job-SUB-READY",
            "submission_id": "SUB-READY",
            "exam_id": "EXAM-READY",
            "status": "completed",
        }
    )
    await db["evalpen_questions"].insert_many(
        [
            {
                "question_id": f"EXAM-READY::Q-{number}",
                "exam_id": "EXAM-READY",
                "question_number": number,
                "max_marks": 4.0,
            }
            for number in (1, 2)
        ]
    )
    await db["evalpen_detected_responses"].insert_one(
        {
            "response_id": "RESP-READY-1",
            "submission_id": "SUB-READY",
            "exam_id": "EXAM-READY",
            "student_id": "STU-1",
            "question_id": "EXAM-READY::Q-1",
            "question_number": 1,
            "detected_text": "Worked answer",
            "source_pages": [{"page_number": 1, "y_start": 10, "y_end": 80}],
            "answer_state": "detected",
            "eval_status": "evaluated",
            "flags": [],
        }
    )
    await db["evalpen_evaluations"].insert_one(
        {
            "evaluation_id": "EVAL-READY-1",
            "response_id": "RESP-READY-1",
            "question_id": "EXAM-READY::Q-1",
            "total_score": 3.0,
            "max_score": 4.0,
            "manual_review_required": False,
        }
    )
    if include_q2:
        await db["evalpen_detected_responses"].insert_one(
            {
                "response_id": "RESP-READY-2",
                "submission_id": "SUB-READY",
                "exam_id": "EXAM-READY",
                "student_id": "STU-1",
                "question_id": "EXAM-READY::Q-2",
                "question_number": 2,
                "detected_text": "",
                "source_pages": [],
                "is_missing_response": True,
                "absence_proven": proven_blank,
                "answer_state": "not_attempted",
                "eval_status": "not_attempted",
                "flags": [],
                "question_assignment": {
                    "method": "not_attempted",
                    "absence_proof": {
                        "verified": proven_blank,
                        "method": "document_answer_mapping",
                    },
                },
            }
        )
        await db["evalpen_evaluations"].insert_one(
            {
                "evaluation_id": "EVAL-READY-2",
                "response_id": "RESP-READY-2",
                "question_id": "EXAM-READY::Q-2",
                "total_score": 0.0,
                "max_score": 4.0,
                "manual_review_required": False,
            }
        )


@pytest.mark.asyncio
async def test_readiness_accepts_only_evidence_backed_complete_paper():
    from services.exampen_submission_readiness import assess_submission_readiness

    db = _fresh_db()
    await _seed_ready_submission(db)

    report = await assess_submission_readiness(db, "SUB-READY")

    assert report["ready"] is True
    assert report["counts"] == {
        "question_count": 2,
        "response_count": 2,
        "evaluation_count": 2,
        "blocker_count": 0,
    }


@pytest.mark.asyncio
async def test_readiness_accepts_objective_negative_marking_within_frozen_penalty():
    from services.exampen_submission_readiness import (
        assess_submission_readiness,
        build_publication_snapshot,
    )

    db = _fresh_db()
    await _seed_ready_submission(db)
    await db["evalpen_questions"].update_one(
        {"question_id": "EXAM-READY::Q-1"},
        {"$set": {"grading_mode": "objective", "penalty_marks": 1.0}},
    )
    await db["evalpen_evaluations"].update_one(
        {"evaluation_id": "EVAL-READY-1"},
        {"$set": {"total_score": -1.0}},
    )

    report = await assess_submission_readiness(db, "SUB-READY")
    snapshot = await build_publication_snapshot(
        db,
        "SUB-READY",
        actor_id="TEACHER-1",
    )

    assert report["ready"] is True
    assert snapshot["score_rows"][0]["score"] == -1.0
    assert snapshot["total_score"] == -1.0


@pytest.mark.asyncio
async def test_readiness_blocks_unproven_zero_and_missing_question_state():
    from services.exampen_submission_readiness import assess_submission_readiness

    unproven_db = _fresh_db()
    await _seed_ready_submission(unproven_db, proven_blank=False)
    unproven = await assess_submission_readiness(unproven_db, "SUB-READY")
    assert "absence_not_proven" in {item["code"] for item in unproven["blockers"]}

    missing_db = _fresh_db()
    await _seed_ready_submission(missing_db, include_q2=False)
    missing = await assess_submission_readiness(missing_db, "SUB-READY")
    assert "question_state_missing" in {item["code"] for item in missing["blockers"]}


@pytest.mark.asyncio
async def test_readiness_blocks_unassigned_handwriting_even_when_questions_have_scores():
    from services.exampen_submission_readiness import assess_submission_readiness

    db = _fresh_db()
    await _seed_ready_submission(db)
    await db["evalpen_detected_responses"].insert_one(
        {
            "response_id": "RESP-UNASSIGNED",
            "submission_id": "SUB-READY",
            "question_id": None,
            "detected_text": "Extra handwritten working",
            "source_pages": [{"page_number": 2, "y_start": 0, "y_end": 200}],
            "eval_status": "blocked",
            "flags": [],
        }
    )

    report = await assess_submission_readiness(db, "SUB-READY")

    assert report["ready"] is False
    assert "unassigned_evidence" in {item["code"] for item in report["blockers"]}


@pytest.mark.asyncio
async def test_readiness_blocks_duplicate_evidence_ownership():
    from services.exampen_submission_readiness import assess_submission_readiness

    db = _fresh_db()
    await _seed_ready_submission(db)
    await db["evalpen_detected_responses"].update_many(
        {"submission_id": "SUB-READY"},
        {"$set": {"evidence_atom_ids": ["ocr-shared-atom"]}},
    )

    report = await assess_submission_readiness(db, "SUB-READY")

    assert report["ready"] is False
    blocker = next(
        item
        for item in report["blockers"]
        if item["code"] == "duplicate_evidence_ownership"
    )
    assert blocker["evidence_atoms"]["ocr-shared-atom"] == [
        "RESP-READY-1",
        "RESP-READY-2",
    ]


@pytest.mark.asyncio
async def test_readiness_allows_shared_page_atoms_from_whole_copy_visual_grading():
    from services.exampen_submission_readiness import assess_submission_readiness

    db = _fresh_db()
    await _seed_ready_submission(db)
    await db["evalpen_detected_responses"].update_many(
        {"submission_id": "SUB-READY"},
        {
            "$set": {
                "evidence_atom_ids": ["immutable-page-1"],
                "question_assignment.method": "full_document_visual",
                "visual_evidence.method": "whole_copy_visual",
            }
        },
    )

    report = await assess_submission_readiness(db, "SUB-READY")

    assert report["ready"] is True
    assert "duplicate_evidence_ownership" not in {
        item["code"] for item in report["blockers"]
    }


@pytest.mark.asyncio
async def test_readiness_keeps_scored_review_notes_nonblocking():
    from services.exampen_submission_readiness import assess_submission_readiness

    db = _fresh_db()
    await _seed_ready_submission(db)
    await db["evalpen_detected_responses"].update_one(
        {"response_id": "RESP-READY-1"},
        {
            "$set": {
                "manual_review_required": True,
                "manual_review_reason": "Number label was not readable",
                "question_assignment.manual_review_required": True,
            }
        },
    )
    await db["evalpen_evaluations"].update_one(
        {"response_id": "RESP-READY-1"},
        {"$set": {"manual_review_required": True}},
    )

    report = await assess_submission_readiness(db, "SUB-READY")

    assert report["ready"] is True
    assert report["blockers"] == []
    notes_by_code = {item["code"]: item for item in report["review_notes"]}
    assert notes_by_code["response_assignment_requires_review"]["response_id"] == (
        "RESP-READY-1"
    )
    assert notes_by_code["response_assignment_requires_review"]["reason"] == (
        "Number label was not readable"
    )
    assert notes_by_code["evaluation_requires_review"]["response_id"] == (
        "RESP-READY-1"
    )


@pytest.mark.asyncio
async def test_document_coverage_is_required_before_publish_and_teacher_can_resolve_it():
    from api.v1.evalpen_review_async import (
        DocumentCoverageReviewRequest,
        PublishRequest,
        confirm_document_coverage_review,
        publish_submission,
    )
    from services.exampen_submission_readiness import assess_submission_readiness

    db = _fresh_db()
    await _seed_ready_submission(db)
    await db["evalpen_submissions"].update_one(
        {"submission_id": "SUB-READY"},
        {
            "$set": {
                "review_state": "needs_review",
                "document_review": {
                    "status": "pending_review",
                    "required": True,
                    "all_student_work_accounted": False,
                    "confidence": 0.78,
                    "warnings": ["Confirm that the faint bottom edge has no work."],
                    "grading_run_id": "DOCGR-run-1",
                },
            }
        },
    )
    await db["evalpen_document_grading_runs"].insert_one(
        {
            "run_id": "DOCGR-run-1",
            "submission_id": "SUB-READY",
            "validated_payload": {
                "unassigned_student_regions": [
                    {
                        "region_id": "page-8-unassigned",
                        "page_number": 8,
                        "x_start": 2,
                        "y_start": 4,
                        "x_end": 98,
                        "y_end": 96,
                        "evidence": "Chemistry work unrelated to this Physics exam",
                        "mapping_confidence": 0.99,
                    }
                ]
            },
        }
    )
    await db["exampen_processing_jobs"].update_one(
        {"submission_id": "SUB-READY"},
        {
            "$set": {
                "status": "blocked_for_review",
                "last_error": "Teacher review is required",
                "failure_code": "ManualReviewRequired",
            }
        },
    )

    before = await assess_submission_readiness(db, "SUB-READY")
    assert before["ready"] is False
    assert before["grading_complete"] is True
    assert before["blockers"] == []
    assert before["review_notes"] == []
    assert [item["code"] for item in before["required_actions"]] == [
        "document_coverage_requires_review"
    ]
    assert before["required_actions"][0]["findings"] == [
        {
            "region_id": "page-8-unassigned",
            "page_number": 8,
            "x_start": 2.0,
            "y_start": 4.0,
            "x_end": 98.0,
            "y_end": 96.0,
            "mapping_confidence": 0.99,
            "evidence": "Chemistry work unrelated to this Physics exam",
        }
    ]

    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async._get_tutor_scoped_student_ids",
            return_value=None,
        ),
        pytest.raises(HTTPException) as publish_error,
    ):
        await publish_submission(
            "SUB-READY",
            PublishRequest(),
            current_user=_admin_user(),
            db=None,
        )
    assert publish_error.value.status_code == 409

    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async._get_tutor_scoped_student_ids",
            return_value=None,
        ),
        pytest.raises(HTTPException) as incomplete_review_error,
    ):
        await confirm_document_coverage_review(
            "SUB-READY",
            DocumentCoverageReviewRequest(
                grading_run_id="DOCGR-run-1",
                note="Checked only part of the uploaded answer copy",
                region_ids=[],
            ),
            current_user=_admin_user(),
            db=None,
        )
    assert incomplete_review_error.value.status_code == 409

    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async._get_tutor_scoped_student_ids",
            return_value=None,
        ),
    ):
        result = await confirm_document_coverage_review(
            "SUB-READY",
            DocumentCoverageReviewRequest(
                grading_run_id="DOCGR-run-1",
                note="Checked every uploaded page against the original copy",
                region_ids=["page-8-unassigned"],
            ),
            current_user=_admin_user(),
            db=None,
        )

    assert result["readiness"]["ready"] is True
    assert result["review_state"] == "ready"
    stored = await db["evalpen_submissions"].find_one(
        {"submission_id": "SUB-READY"}
    )
    assert stored["document_review"]["status"] == "accepted"
    assert stored["document_review"]["required"] is False
    assert stored["document_review"]["resolution"] == "exclude_unassigned_work"
    assert stored["document_review"]["resolved_page_numbers"] == [8]
    assert stored["document_review_history"][-1]["actor_id"] == "TEACHER-1"
    assert stored["document_review_history"][-1]["action"] == (
        "unassigned_work_excluded"
    )
    job = await db["exampen_processing_jobs"].find_one(
        {"submission_id": "SUB-READY"}
    )
    assert job["status"] == "completed"
    assert job["last_error"] is None
    assert "failure_code" not in job


@pytest.mark.asyncio
async def test_publication_builds_valid_hashed_snapshot():
    from api.v1.evalpen_review_async import PublishRequest, publish_submission
    from services.exampen_submission_readiness import validate_publication_snapshot

    db = _fresh_db()
    await _seed_ready_submission(db)
    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async._get_tutor_scoped_student_ids",
            return_value=None,
        ),
    ):
        result = await publish_submission(
            "SUB-READY",
            PublishRequest(note="Checked by teacher"),
            current_user=_admin_user(),
            db=None,
        )

    stored = await db["evalpen_submissions"].find_one(
        {"submission_id": "SUB-READY"}
    )
    snapshot = stored["publication_snapshot"]
    assert result["publication_status"] == "published"
    assert snapshot["total_score"] == 3.0
    assert snapshot["total_max_score"] == 8.0
    assert stored["publication_history"] == [
        {
            "action": "published",
            "published_at": stored["published_at"],
            "published_by": "TEACHER-1",
            "publication_note": "Checked by teacher",
            "snapshot_hash": snapshot["snapshot_hash"],
        }
    ]
    assert "review_mutation_lease_token" not in stored
    assert validate_publication_snapshot(
        snapshot,
        submission_id="SUB-READY",
        exam_id="EXAM-READY",
    )

    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async._get_tutor_scoped_student_ids",
            return_value=None,
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await publish_submission(
            "SUB-READY",
            PublishRequest(note="Second publication must not replace history"),
            current_user=_admin_user(),
            db=None,
        )
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_publication_allows_nonblocking_manual_review_notes():
    from api.v1.evalpen_review_async import PublishRequest, publish_submission
    from services.exampen_submission_readiness import assess_submission_readiness

    db = _fresh_db()
    await _seed_ready_submission(db)
    response = await db["evalpen_detected_responses"].find_one(
        {"submission_id": "SUB-READY"}
    )
    await db["evalpen_detected_responses"].update_one(
        {"response_id": response["response_id"]},
        {
            "$set": {
                "eval_status": "manual_review",
                "manual_review_required": True,
            }
        },
    )
    await db["evalpen_evaluations"].update_one(
        {"response_id": response["response_id"]},
        {"$set": {"manual_review_required": True}},
    )

    readiness = await assess_submission_readiness(db, "SUB-READY")
    assert readiness["ready"] is True
    assert readiness["review_notes"]

    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async._get_tutor_scoped_student_ids",
            return_value=None,
        ),
    ):
        result = await publish_submission(
            "SUB-READY",
            PublishRequest(),
            current_user=_admin_user(),
            db=None,
        )

    assert result["publication_status"] == "published"


@pytest.mark.asyncio
async def test_publish_ready_batch_uses_exam_scope_and_shared_publish_contract():
    from api.v1.evalpen_review_async import (
        PublishReadyRequest,
        publish_ready_submissions,
    )

    db = _fresh_db()
    await _seed_ready_submission(db)
    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async._get_tutor_scoped_student_ids",
            return_value=None,
        ),
    ):
        result = await publish_ready_submissions(
            "EXAM-READY",
            PublishReadyRequest(submission_ids=["SUB-READY", "SUB-MISSING"]),
            current_user=_admin_user(),
            db=None,
        )

    assert result["requested_count"] == 2
    assert result["published_count"] == 1
    assert result["failed_count"] == 1
    assert result["results"][0]["status"] == "published"
    assert result["results"][1]["status"] == "failed"


@pytest.mark.asyncio
async def test_publish_rejects_incomplete_question_coverage():
    from api.v1.evalpen_review_async import PublishRequest, publish_submission

    db = _fresh_db()
    await _seed_ready_submission(db, include_q2=False)
    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async._get_tutor_scoped_student_ids",
            return_value=None,
        ),
    ):
        with pytest.raises(HTTPException) as exc:
            await publish_submission(
                "SUB-READY",
                PublishRequest(),
                current_user=_admin_user(),
                db=None,
            )

    assert exc.value.status_code == 409
    assert exc.value.detail["readiness"]["ready"] is False


@pytest.mark.asyncio
async def test_teacher_confirmed_blank_is_audited_evaluated_and_can_complete_readiness():
    from api.v1.evalpen_review_async import (
        ResponseAssignmentCorrectionRequest,
        correct_response_assignment,
    )

    db = _fresh_db()
    await _seed_ready_submission(db, include_q2=False)
    await db["exampen_processing_jobs"].update_one(
        {"submission_id": "SUB-READY"},
        {"$set": {"status": "blocked_for_review"}},
    )

    class _FakeEvalCore:
        async def evaluate_response(self, response_id, *, question_id):
            response = await db["evalpen_detected_responses"].find_one(
                {"response_id": response_id}
            )
            await db["evalpen_evaluations"].insert_one(
                {
                    "evaluation_id": f"EVAL-{response_id}",
                    "response_id": response_id,
                    "question_id": question_id,
                    "total_score": 0.0,
                    "max_score": 4.0,
                    "manual_review_required": False,
                }
            )
            await db["evalpen_detected_responses"].update_one(
                {"response_id": response_id},
                {"$set": {"eval_status": "not_attempted"}},
            )
            assert response["absence_proven"] is True
            return SimpleNamespace(error=None)

    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async._get_tutor_scoped_student_ids",
            return_value=None,
        ),
        patch(
            "api.v1.evalpen_evaluate_async._build_eval_core",
            new=AsyncMock(return_value=_FakeEvalCore()),
        ),
    ):
        result = await correct_response_assignment(
            "SUB-READY",
            ResponseAssignmentCorrectionRequest(
                action="confirm_not_attempted",
                question_id="EXAM-READY::Q-2",
                reason="Visually checked every uploaded page",
            ),
            current_user=_admin_user(),
            db=None,
        )

    assert result["readiness"]["ready"] is True
    audit = await db["evalpen_response_assignment_audit"].find_one(
        {"submission_id": "SUB-READY"}
    )
    assert audit["action"] == "confirm_not_attempted"
    job = await db["exampen_processing_jobs"].find_one(
        {"submission_id": "SUB-READY"}
    )
    assert job["status"] == "completed"


@pytest.mark.asyncio
async def test_teacher_split_creates_disjoint_evidence_and_supersedes_the_source():
    from api.v1.evalpen_review_async import (
        ResponseAssignmentCorrectionRequest,
        ResponseRegionCorrection,
        ResponseSplitPart,
        correct_response_assignment,
    )

    db = _fresh_db()
    await _seed_ready_submission(db)
    await db["exampen_processing_jobs"].update_one(
        {"submission_id": "SUB-READY"},
        {"$set": {"status": "blocked_for_review"}},
    )
    await db["evalpen_detected_responses"].update_one(
        {"response_id": "RESP-READY-1"},
        {
            "$set": {
                "evidence_atom_ids": ["original-full-region"],
                "source_pages": [
                    {
                        "page_number": 1,
                        "x_start": 25,
                        "y_start": 10,
                        "x_end": 150,
                        "y_end": 80,
                        "region_id": "model-region-1",
                        "evidence_kind": "mixed",
                    }
                ],
            }
        },
    )

    class _FakeEvalCore:
        async def evaluate_response(self, response_id, *, question_id):
            await db["evalpen_evaluations"].insert_one(
                {
                    "evaluation_id": f"EVAL-{response_id}",
                    "response_id": response_id,
                    "question_id": question_id,
                    "total_score": 2.0,
                    "max_score": 4.0,
                    "manual_review_required": False,
                }
            )
            await db["evalpen_detected_responses"].update_one(
                {"response_id": response_id},
                {"$set": {"eval_status": "evaluated_teacher_reviewed"}},
            )
            return SimpleNamespace(error=None)

    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async._get_tutor_scoped_student_ids",
            return_value=None,
        ),
        patch(
            "api.v1.evalpen_evaluate_async._build_eval_core",
            new=AsyncMock(return_value=_FakeEvalCore()),
        ),
    ):
        result = await correct_response_assignment(
            "SUB-READY",
            ResponseAssignmentCorrectionRequest(
                action="split",
                response_id="RESP-READY-1",
                reason="Two answers were written in one continuous region",
                parts=[
                    ResponseSplitPart(
                        question_id="EXAM-READY::Q-1",
                        detected_text="First corrected answer",
                        source_pages=[
                            ResponseRegionCorrection(
                                page_number=1,
                                x_start=25,
                                y_start=10,
                                x_end=150,
                                y_end=45,
                                region_id="model-region-1",
                                evidence_kind="mathematics",
                            )
                        ],
                    ),
                    ResponseSplitPart(
                        question_id="EXAM-READY::Q-2",
                        detected_text="Second corrected answer",
                        source_pages=[
                            ResponseRegionCorrection(
                                page_number=1,
                                x_start=25,
                                y_start=45,
                                x_end=150,
                                y_end=80,
                                region_id="model-region-1",
                                evidence_kind="diagram",
                            )
                        ],
                    ),
                ],
            ),
            current_user=_admin_user(),
            db=None,
        )

    assert result["readiness"]["ready"] is True
    active = await db["evalpen_detected_responses"].find(
        {"submission_id": "SUB-READY", "superseded_at": {"$exists": False}}
    ).sort("question_number", 1).to_list(length=10)
    assert len(active) == 2
    assert active[0]["evidence_atom_ids"] != active[1]["evidence_atom_ids"]
    assert all(item["evidence_source"] == "teacher_split" for item in active)
    assert all(item["source_pages"][0]["x_start"] == 25 for item in active)
    assert all(item["source_pages"][0]["x_end"] == 150 for item in active)
    assert active[0]["source_pages"][0]["region_id"].startswith("teacher-split-")
    assert active[1]["source_pages"][0]["region_id"].startswith("teacher-split-")
    assert (
        active[0]["source_pages"][0]["region_id"]
        != active[1]["source_pages"][0]["region_id"]
    )
    original = await db["evalpen_detected_responses"].find_one(
        {"response_id": "RESP-READY-1"}
    )
    assert original["eval_status"] == "superseded"


@pytest.mark.asyncio
async def test_teacher_correction_rejects_an_active_submission_lease():
    from api.v1.evalpen_review_async import (
        ResponseAssignmentCorrectionRequest,
        correct_response_assignment,
    )

    db = _fresh_db()
    await _seed_ready_submission(db)
    await db["evalpen_submissions"].update_one(
        {"submission_id": "SUB-READY"},
        {
            "$set": {
                "review_mutation_lease_token": "another-request",
                "review_mutation_lease_expires_at": datetime.now(timezone.utc)
                + timedelta(minutes=5),
            }
        },
    )

    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async._get_tutor_scoped_student_ids",
            return_value=None,
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await correct_response_assignment(
            "SUB-READY",
            ResponseAssignmentCorrectionRequest(
                action="discard_non_answer",
                response_id="RESP-READY-1",
                reason="This region is unrelated working",
            ),
            current_user=_admin_user(),
            db=None,
        )

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_teacher_correction_releases_its_lease_after_a_validation_error():
    from api.v1.evalpen_review_async import (
        ResponseAssignmentCorrectionRequest,
        correct_response_assignment,
    )

    db = _fresh_db()
    await _seed_ready_submission(db)

    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async._get_tutor_scoped_student_ids",
            return_value=None,
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await correct_response_assignment(
            "SUB-READY",
            ResponseAssignmentCorrectionRequest(
                action="assign",
                response_id="RESP-READY-1",
                question_id="EXAM-READY::UNKNOWN",
                reason="Testing an invalid immutable question",
            ),
            current_user=_admin_user(),
            db=None,
        )

    assert exc_info.value.status_code == 400
    submission = await db["evalpen_submissions"].find_one(
        {"submission_id": "SUB-READY"}
    )
    assert "review_mutation_lease_token" not in submission
    assert "review_mutation_lease_expires_at" not in submission
