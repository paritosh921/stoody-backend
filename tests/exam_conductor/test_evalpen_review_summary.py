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


@pytest.mark.asyncio
async def test_review_summary_keeps_full_paper_denominator_and_shows_blank_questions():
    """Historical partial OCR records must never turn a 12-mark paper into 4/4."""
    from api.v1.evalpen_review_async import get_submission_summary

    db = _fresh_db()
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-FULL-PAPER",
            "exam_id": "EXAM-FULL-PAPER",
            "student_id": "STU-1",
            "source": "web_upload",
            "segmentation_status": "complete",
        }
    )
    await db["evalpen_questions"].insert_many(
        [
            {
                "question_id": f"EXAM-FULL-PAPER::Q-{number}",
                "exam_id": "EXAM-FULL-PAPER",
                "question_number": number,
                "max_marks": 4,
            }
            for number in (1, 2, 3)
        ]
    )
    await db["evalpen_detected_responses"].insert_one(
        {
            "response_id": "RESP-FULL-PAPER-1",
            "submission_id": "SUB-FULL-PAPER",
            "question_id": "EXAM-FULL-PAPER::Q-1",
            "question_number": 1,
            "content_type": "TEXT_ONLY",
            "detected_text": "The only submitted answer",
            "eval_status": "evaluated",
            "flags": [],
        }
    )
    await db["evalpen_evaluations"].insert_one(
        {
            "evaluation_id": "EVAL-FULL-PAPER-1",
            "response_id": "RESP-FULL-PAPER-1",
            "total_score": 4.0,
            "max_score": 4.0,
            "overall_feedback": "Correct.",
        }
    )

    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async._get_tutor_scoped_student_ids",
            return_value=None,
        ),
    ):
        result = await get_submission_summary(
            submission_id="SUB-FULL-PAPER",
            current_user=_admin_user(),
            db=None,
        )

    assert result.total_score == 4.0
    assert result.total_max_score == 12.0
    assert [response.question_number for response in result.responses] == [1, 2, 3]
    assert [response.is_missing_response for response in result.responses] == [False, True, True]
    assert [response.total_score for response in result.responses[1:]] == [0.0, 0.0]
    assert [response.max_score for response in result.responses[1:]] == [4.0, 4.0]


@pytest.mark.asyncio
async def test_teacher_criterion_review_recomputes_total_and_keeps_audit_history():
    """Teachers can adjust only frozen criterion awards, never the rubric itself."""
    import os
    import sys

    ec_dir = os.path.join(os.path.dirname(__file__), "..", "..", "exam-conductor")
    if ec_dir not in sys.path:
        sys.path.insert(0, ec_dir)
    from pcr.storage.evaluation_repo import EvaluationRepository

    db = _fresh_db()
    await db["evalpen_evaluations"].insert_one(
        {
            "evaluation_id": "EVAL-rubric",
            "response_id": "RESP-rubric",
            "student_id": "STU-1",
            "total_score": 1.0,
            "max_score": 4.0,
            "manual_review_required": True,
            "criterion_marks": [
                {
                    "criterion_id": "method",
                    "description": "Uses the correct equation",
                    "marks_awarded": 1.0,
                    "max_marks": 1.0,
                    "rationale": "Equation present",
                },
                {
                    "criterion_id": "result",
                    "description": "Obtains the correct result",
                    "marks_awarded": 0.0,
                    "max_marks": 3.0,
                    "rationale": "AI could not read final line",
                },
            ],
            "audit_trail": [],
        }
    )

    updated = await EvaluationRepository(db).override_criterion_marks(
        "EVAL-rubric",
        marks_by_criterion={"method": 1.0, "result": 2.5},
        actor_id="teacher-1",
        reason="Final calculation is legible in the uploaded copy",
    )

    assert updated is not None
    assert updated["total_score"] == 3.5
    stored = await db["evalpen_evaluations"].find_one({"evaluation_id": "EVAL-rubric"})
    assert stored["criterion_marks"][0]["description"] == "Uses the correct equation"
    assert stored["criterion_marks"][1]["marks_awarded"] == 2.5
    assert stored["manual_review_required"] is False
    assert stored["audit_trail"][-1]["action"] == "criterion_marks_override"
