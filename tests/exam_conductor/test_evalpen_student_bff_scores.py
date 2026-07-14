from __future__ import annotations

from unittest.mock import patch

import pytest


def _fresh_db():
    from mongomock_motor import AsyncMongoMockClient

    return AsyncMongoMockClient()["skb_test"]


def _student_user():
    return {
        "user_id": "STU-1",
        "student_id": "STU-1",
        "user_type": "student",
        "db_name": "skb_test",
    }


@pytest.mark.asyncio
async def test_student_score_breakdown_keeps_unanswered_questions_in_total():
    """A published partial copy must show 4/12, with two explicit zero rows."""
    from api.v1.evalpen_student_bff_async import get_student_exam_scores

    db = _fresh_db()
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-STUDENT-FULL",
            "exam_id": "EXAM-STUDENT-FULL",
            "student_id": "STU-1",
            "publication_status": "published",
        }
    )
    await db["evalpen_questions"].insert_many(
        [
            {
                "question_id": f"EXAM-STUDENT-FULL::Q-{number}",
                "exam_id": "EXAM-STUDENT-FULL",
                "question_number": number,
                "max_marks": 4,
            }
            for number in (1, 2, 3)
        ]
    )
    await db["evalpen_detected_responses"].insert_one(
        {
            "response_id": "RESP-STUDENT-FULL-1",
            "submission_id": "SUB-STUDENT-FULL",
            "question_id": "EXAM-STUDENT-FULL::Q-1",
            "question_number": 1,
            "eval_status": "evaluated",
            "answer_state": "detected",
        }
    )
    await db["evalpen_evaluations"].insert_one(
        {
            "evaluation_id": "EVAL-STUDENT-FULL-1",
            "response_id": "RESP-STUDENT-FULL-1",
            "total_score": 4.0,
            "max_score": 4.0,
            "overall_feedback": "Correct.",
        }
    )

    with (
        patch(
            "api.v1.evalpen_student_bff_async._get_tenant_db",
            return_value=db,
        ),
        patch(
            "api.v1.evalpen_student_bff_async._get_student_identity_ids",
            return_value=["STU-1"],
        ),
    ):
        result = await get_student_exam_scores(
            exam_id="EXAM-STUDENT-FULL",
            current_user=_student_user(),
            db=None,
        )

    assert result.total_score == 4.0
    assert result.max_score == 12.0
    assert [item.question_id for item in result.questions] == [
        "EXAM-STUDENT-FULL::Q-1",
        "EXAM-STUDENT-FULL::Q-2",
        "EXAM-STUDENT-FULL::Q-3",
    ]
    assert [item.score for item in result.questions] == [4.0, 0.0, 0.0]
    assert [item.answer_state for item in result.questions] == [
        "detected",
        "not_attempted",
        "not_attempted",
    ]
