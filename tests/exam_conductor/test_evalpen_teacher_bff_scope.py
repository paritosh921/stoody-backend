from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import pytest
from fastapi import HTTPException


def _fresh_db():
    from mongomock_motor import AsyncMongoMockClient

    return AsyncMongoMockClient()["skb_test"]


def _tutor_user(tutor_id: str = "TUT-1") -> Dict[str, Any]:
    return {
        "user_id": f"user-{tutor_id}",
        "user_type": "tutor",
        "tutor_id": tutor_id,
        "admin_id": "64f000000000000000000001",
        "db_name": "skb_test",
    }


async def _seed_visible_exam_with_hub_submission(db) -> None:
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "EXAM-1",
            "exam_type": "pcr",
            "lifecycle_state": "collection_closed",
            "created_by_tutor_id": "TUT-1",
            "teacher_ids": ["TUT-1"],
        }
    )
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-1",
            "exam_id": "EXAM-1",
            "student_id": "lavyansh",
            "source": "ble_pen",
            # Deliberately stale metadata: the queue must prefer the canonical
            # page collection rather than showing the response count as pages.
            "page_count": 1,
            "segmentation_status": "complete",
        }
    )
    await db["evalpen_answer_pages"].insert_many(
        [
            {
                "page_id": f"PAGE-{number}",
                "submission_id": "SUB-1",
                "page_number": number,
            }
            for number in range(1, 5)
        ]
    )
    await db["evalpen_detected_responses"].insert_many(
        [
            {
                "response_id": "RESP-1",
                "submission_id": "SUB-1",
                "question_id": "EXAM-1::Q-1",
                "eval_status": "evaluated",
                "flags": [],
            },
            {
                "response_id": "RESP-2",
                "submission_id": "SUB-1",
                "question_id": "EXAM-1::Q-2",
                "eval_status": "evaluated",
                "flags": [],
            },
        ]
    )
    await db["evalpen_questions"].insert_many(
        [
            {
                "question_id": f"EXAM-1::Q-{number}",
                "exam_id": "EXAM-1",
                "question_number": number,
                "max_marks": 4,
            }
            for number in (1, 2)
        ]
    )
    await db["evalpen_evaluations"].insert_many(
        [
            {
                "evaluation_id": f"EVAL-{number}",
                "response_id": f"RESP-{number}",
                "question_id": f"EXAM-1::Q-{number}",
                "total_score": 3,
                "max_score": 4,
                "manual_review_required": False,
            }
            for number in (1, 2)
        ]
    )
    await db["exampen_processing_jobs"].insert_one(
        {
            "job_id": "JOB-1",
            "submission_id": "SUB-1",
            "status": "completed",
        }
    )


@pytest.mark.asyncio
async def test_teacher_bff_counts_hub_submission_by_visible_exam_not_student_scope():
    from api.v1.evalpen_teacher_bff_async import list_exams

    db = _fresh_db()
    await _seed_visible_exam_with_hub_submission(db)

    with (
        patch("api.v1.evalpen_teacher_bff_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_teacher_bff_async.get_tutor_scoped_students",
            return_value=[{"student_id": "different-student-id"}],
        ),
    ):
        result = await list_exams(current_user=_tutor_user(), db=None)

    items = {item.exam_id: item for item in result.items}
    assert items["EXAM-1"].total_students == 1
    assert items["EXAM-1"].evaluated_count == 1
    assert items["EXAM-1"].blocked_count == 0
    assert items["EXAM-1"].pending_count == 0
    assert items["EXAM-1"].ready_to_publish_count == 1
    assert items["EXAM-1"].workflow_status == "ready_to_publish"


@pytest.mark.asyncio
async def test_teacher_bff_list_exposes_class_section_and_prioritizes_student_requests():
    from api.v1.evalpen_teacher_bff_async import list_exams

    db = _fresh_db()
    await db["documents"].insert_one(
        {
            "document_id": "DOC-1",
            "title": "Term exam",
            "exam_finalized": True,
            "exam_mode": "pcr",
            "standard": "Class 11",
            "section": "Section A",
        }
    )
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "EXAM-REQUEST",
            "title": "Term exam",
            "exam_type": "pcr",
            "lifecycle_state": "collection_closed",
            "prepared_document_id": "DOC-1",
            "created_by_tutor_id": "TUT-1",
            "teacher_ids": ["TUT-1"],
        }
    )
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-REQUEST",
            "exam_id": "EXAM-REQUEST",
            "student_id": "student-1",
            "publication_status": "published",
        }
    )
    await db["evalpen_recheck_requests"].insert_one(
        {
            "request_id": "REQ-1",
            "exam_id": "EXAM-REQUEST",
            "student_id": "student-1",
            "status": "open",
        }
    )

    admin_user = {
        "user_id": "admin-1",
        "user_type": "admin",
        "db_name": "skb_test",
    }
    with patch(
        "api.v1.evalpen_teacher_bff_async._get_tenant_db",
        return_value=db,
    ):
        result = await list_exams(current_user=admin_user, db=None)

    item = next(row for row in result.items if row.exam_id == "EXAM-REQUEST")
    assert item.class_name == "11"
    assert item.section_name == "A"
    assert item.class_label == "Class 11 - Section A"
    assert item.published_count == 1
    assert item.open_recheck_count == 1
    assert item.workflow_status == "under_review"


@pytest.mark.asyncio
async def test_exam_roster_shows_full_owned_exam_not_only_generic_student_scope():
    from api.v1.evalpen_review_async import get_exam_roster

    db = _fresh_db()
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "EXAM-ROSTER",
            "created_by_tutor_id": "TUT-1",
            "teacher_ids": ["TUT-1"],
            "roster": ["student-a", "student-b"],
        }
    )
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-ROSTER-A",
            "exam_id": "EXAM-ROSTER",
            "student_id": "student-a",
            "publication_status": "published",
        }
    )
    await db["evalpen_recheck_requests"].insert_one(
        {
            "request_id": "REQ-ROSTER-A",
            "exam_id": "EXAM-ROSTER",
            "student_id": "student-a",
            "status": "under_review",
        }
    )

    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async.get_tutor_scoped_students",
            return_value=[{"student_id": "different-student-id"}],
        ),
    ):
        result = await get_exam_roster(
            exam_id="EXAM-ROSTER",
            current_user=_tutor_user(),
            db=None,
        )

    assert result.total_expected == 2
    assert [row.student_id for row in result.expected_students] == [
        "student-a",
        "student-b",
    ]
    first_student = result.expected_students[0]
    assert first_student.status == "review"
    assert first_student.open_recheck_count == 1
    assert result.total_submitted == 1
    assert result.total_published == 1
    assert result.total_needs_review == 1


@pytest.mark.asyncio
async def test_exam_roster_marks_ingest_failed_student_copy_as_blocked():
    from api.v1.evalpen_review_async import get_exam_roster

    db = _fresh_db()
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "EXAM-FAILED-COPY",
            "created_by_tutor_id": "TUT-1",
            "teacher_ids": ["TUT-1"],
            "roster": ["rohan21"],
        }
    )
    await db["exampen_student_copy_uploads"].insert_one(
        {
            "attempt_id": "attempt-ingest-failed",
            "exam_id": "EXAM-FAILED-COPY",
            "student_id": "rohan21",
            "status": "ingest_failed",
            "last_error": "'student_web' is not a valid ArtifactSource",
        }
    )

    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async.get_tutor_scoped_students",
            return_value=[{"student_id": "rohan21"}],
        ),
    ):
        result = await get_exam_roster(
            exam_id="EXAM-FAILED-COPY",
            current_user=_tutor_user(),
            db=None,
        )

    assert result.expected_students[0].student_id == "rohan21"
    assert result.expected_students[0].status == "blocked"
    assert result.expected_students[0].submission_id is None
    assert result.total_blocked == 1
    assert result.total_submitted == 0


@pytest.mark.asyncio
async def test_teacher_bff_queue_uses_visible_exam_for_hub_submission():
    from api.v1.evalpen_teacher_bff_async import get_exam_queue

    db = _fresh_db()
    await _seed_visible_exam_with_hub_submission(db)

    with (
        patch("api.v1.evalpen_teacher_bff_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_teacher_bff_async.get_tutor_scoped_students",
            return_value=[{"student_id": "different-student-id"}],
        ),
    ):
        result = await get_exam_queue(
            exam_id="EXAM-1",
            current_user=_tutor_user(),
            db=None,
        )

    assert result.pending == []
    assert result.blocked == []
    assert len(result.ready_to_publish) == 1
    assert result.ready_to_publish[0].submission_id == "SUB-1"
    assert result.ready_to_publish[0].student_id == "lavyansh"
    assert result.ready_to_publish[0].response_count == 2
    assert result.ready_to_publish[0].page_count == 4
    assert result.ready_to_publish[0].source == "ble_pen"


@pytest.mark.asyncio
async def test_teacher_bff_queue_separates_review_from_technical_failure():
    from api.v1.evalpen_teacher_bff_async import get_exam_queue

    db = _fresh_db()
    await _seed_visible_exam_with_hub_submission(db)
    await db["evalpen_submissions"].update_one(
        {"submission_id": "SUB-1"},
        {
            "$set": {
                "review_state": "needs_review",
                "document_review": {
                    "status": "pending_review",
                    "required": True,
                    "confidence": 0.76,
                    "warnings": ["Confirm the faint page edge."],
                    "grading_run_id": "DOCGR-review",
                },
            }
        },
    )

    with (
        patch("api.v1.evalpen_teacher_bff_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_teacher_bff_async.get_tutor_scoped_students",
            return_value=[{"student_id": "different-student-id"}],
        ),
    ):
        result = await get_exam_queue(
            exam_id="EXAM-1",
            current_user=_tutor_user(),
            db=None,
        )

    assert result.blocked == []
    assert result.ready_to_publish == []
    assert len(result.needs_review) == 1
    assert result.needs_review[0].submission_id == "SUB-1"
    assert result.needs_review[0].processing_status == "completed"


@pytest.mark.asyncio
async def test_teacher_bff_approved_submission_with_advisory_is_ready_to_publish():
    """A completed coverage decision must not remain in the review queue."""
    from api.v1.evalpen_teacher_bff_async import get_exam_queue, list_exams

    db = _fresh_db()
    await _seed_visible_exam_with_hub_submission(db)
    await db["evalpen_submissions"].update_one(
        {"submission_id": "SUB-1"},
        {
            "$set": {
                "review_state": "ready",
                "publication_status": "ready",
                "document_review": {
                    "status": "accepted",
                    "required": False,
                    "confidence": 0.93,
                    "warnings": ["Confirm four unassigned visible regions."],
                    "grading_run_id": "DOCGR-approved",
                },
            }
        },
    )
    await db["evalpen_evaluations"].update_many(
        {"response_id": {"$in": ["RESP-1", "RESP-2"]}},
        {
            "$set": {
                "teacher_reviewed": True,
                "teacher_review_status": "approved",
            }
        },
    )

    with (
        patch("api.v1.evalpen_teacher_bff_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_teacher_bff_async.get_tutor_scoped_students",
            return_value=[{"student_id": "lavyansh"}],
        ),
    ):
        queue = await get_exam_queue(
            exam_id="EXAM-1",
            current_user=_tutor_user(),
            db=None,
        )
        exams = await list_exams(current_user=_tutor_user(), db=None)

    assert queue.blocked == []
    assert queue.needs_review == []
    assert len(queue.ready_to_publish) == 1
    assert queue.ready_to_publish[0].submission_id == "SUB-1"
    exam = next(item for item in exams.items if item.exam_id == "EXAM-1")
    assert exam.needs_review_count == 0
    assert exam.ready_to_publish_count == 1
    assert exam.workflow_status == "ready_to_publish"


@pytest.mark.asyncio
async def test_teacher_bff_queue_treats_scored_manual_review_as_ready_to_publish():
    from api.v1.evalpen_teacher_bff_async import get_exam_queue

    db = _fresh_db()
    await _seed_visible_exam_with_hub_submission(db)
    await db["evalpen_detected_responses"].update_one(
        {"response_id": "RESP-1"},
        {
            "$set": {
                "eval_status": "manual_review",
                "manual_review_required": True,
                "manual_review_reason": "Confirm ownership of this visual evidence.",
                "question_assignment.manual_review_required": True,
            }
        },
    )
    await db["evalpen_evaluations"].update_one(
        {"response_id": "RESP-1"},
        {"$set": {"manual_review_required": True}},
    )

    with (
        patch("api.v1.evalpen_teacher_bff_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_teacher_bff_async.get_tutor_scoped_students",
            return_value=[{"student_id": "lavyansh"}],
        ),
    ):
        result = await get_exam_queue(
            exam_id="EXAM-1",
            current_user=_tutor_user(),
            db=None,
        )

    assert result.blocked == []
    assert result.needs_review == []
    assert len(result.ready_to_publish) == 1
    assert result.ready_to_publish[0].submission_id == "SUB-1"


@pytest.mark.asyncio
async def test_teacher_bff_queue_rejects_exam_hidden_from_tutor():
    from api.v1.evalpen_teacher_bff_async import get_exam_queue

    db = _fresh_db()
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "HIDDEN-1",
            "created_by_tutor_id": "TUT-2",
            "teacher_ids": ["TUT-2"],
        }
    )

    with (
        patch("api.v1.evalpen_teacher_bff_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_teacher_bff_async.get_tutor_scoped_students",
            return_value=[{"student_id": "lavyansh"}],
        ),
    ):
        with pytest.raises(HTTPException) as exc:
            await get_exam_queue(
                exam_id="HIDDEN-1",
                current_user=_tutor_user(),
                db=None,
            )

    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_teacher_bff_queue_exposes_queued_pcr_copy_before_responses_exist():
    from api.v1.evalpen_teacher_bff_async import get_exam_queue

    db = _fresh_db()
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "EXAM-QUEUED",
            "exam_type": "pcr",
            "created_by_tutor_id": "TUT-1",
            "teacher_ids": ["TUT-1"],
        }
    )
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-QUEUED",
            "exam_id": "EXAM-QUEUED",
            "student_id": "harsh21",
        }
    )
    await db["exampen_processing_jobs"].insert_one(
        {
            "job_id": "JOB-QUEUED",
            "submission_id": "SUB-QUEUED",
            "status": "queued",
        }
    )

    with (
        patch("api.v1.evalpen_teacher_bff_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_teacher_bff_async.get_tutor_scoped_students",
            return_value=[{"student_id": "different-student-id"}],
        ),
    ):
        result = await get_exam_queue(
            exam_id="EXAM-QUEUED",
            current_user=_tutor_user(),
            db=None,
        )

    assert len(result.pending) == 1
    item = result.pending[0]
    assert item.student_id == "harsh21"
    assert item.response_count == 0
    assert item.processing_status == "queued"
    assert item.processing_error is None
    assert item.status_summary == "AI checking queued"


@pytest.mark.asyncio
async def test_teacher_bff_queue_marks_failed_pcr_copy_for_staff_attention():
    from api.v1.evalpen_teacher_bff_async import get_exam_queue

    db = _fresh_db()
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "EXAM-FAILED",
            "exam_type": "pcr",
            "created_by_tutor_id": "TUT-1",
            "teacher_ids": ["TUT-1"],
        }
    )
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-FAILED",
            "exam_id": "EXAM-FAILED",
            "student_id": "harsh21",
        }
    )
    await db["exampen_processing_jobs"].insert_one(
        {
            "job_id": "JOB-FAILED",
            "submission_id": "SUB-FAILED",
            "status": "failed",
            "last_error": "OCR provider did not respond",
        }
    )

    with (
        patch("api.v1.evalpen_teacher_bff_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_teacher_bff_async.get_tutor_scoped_students",
            return_value=[{"student_id": "different-student-id"}],
        ),
    ):
        result = await get_exam_queue(
            exam_id="EXAM-FAILED",
            current_user=_tutor_user(),
            db=None,
        )

    assert result.pending == []
    assert len(result.blocked) == 1
    item = result.blocked[0]
    assert item.processing_status == "failed"
    assert item.processing_error == "OCR provider did not respond"
    assert item.status_summary == "AI checking failed"

