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

