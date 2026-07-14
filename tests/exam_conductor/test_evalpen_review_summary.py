from __future__ import annotations

from unittest.mock import patch

import pytest


def _fresh_db():
    from mongomock_motor import AsyncMongoMockClient

    return AsyncMongoMockClient()["skb_test"]


def _admin_user():
    return {
        "user_id": "admin-1",
        "user_type": "admin",
        "db_name": "skb_test",
    }


@pytest.mark.asyncio
async def test_review_summary_includes_staff_visible_answer_and_ai_correction():
    """The teacher workspace must receive the artefacts used to mark a copy."""
    from api.v1.evalpen_review_async import get_submission_summary

    db = _fresh_db()
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-1",
            "exam_id": "EXAM-1",
            "student_id": "STU-1",
            "source": "web_upload",
            "segmentation_status": "complete",
        }
    )
    await db["evalpen_detected_responses"].insert_one(
        {
            "response_id": "RESP-1",
            "submission_id": "SUB-1",
            "question_id": "EXAM-1::Q-1",
            "question_number": 1,
            "content_type": "TEXT_ONLY",
            "detected_text": "T = 2u sin(theta) / g = 2.4 s",
            "eval_status": "evaluated",
            "flags": [],
        }
    )
    await db["evalpen_evaluations"].insert_one(
        {
            "evaluation_id": "EVAL-1",
            "response_id": "RESP-1",
            "total_score": 4.0,
            "max_score": 4.0,
            "overall_feedback": "Correct method and final answer.",
            "reference_solution": "Resolve velocity, then calculate time, range, and height.",
            "step_marks": [
                {
                    "step": "Time of flight",
                    "marks_awarded": 2.0,
                    "max_marks": 2.0,
                    "rationale": "Correct substitution.",
                }
            ],
        }
    )

    with (
        patch(
            "api.v1.evalpen_review_async._get_tenant_db",
            return_value=db,
        ),
        patch(
            "api.v1.evalpen_review_async._get_tutor_scoped_student_ids",
            return_value=None,
        ),
    ):
        result = await get_submission_summary(
            submission_id="SUB-1",
            current_user=_admin_user(),
            db=None,
        )

    assert result.total_score == 4.0
    assert len(result.responses) == 1
    response = result.responses[0]
    assert response.question_number == 1
    assert response.detected_text == "T = 2u sin(theta) / g = 2.4 s"
    assert response.reference_solution == (
        "Resolve velocity, then calculate time, range, and height."
    )
    assert response.step_marks == [
        {
            "step": "Time of flight",
            "marks_awarded": 2.0,
            "max_marks": 2.0,
            "rationale": "Correct substitution.",
        }
    ]
