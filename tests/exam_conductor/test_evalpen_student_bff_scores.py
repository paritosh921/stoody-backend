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
    await db["evalpen_detected_responses"].insert_many(
        [{
            "response_id": "RESP-STUDENT-FULL-1",
            "submission_id": "SUB-STUDENT-FULL",
            "question_id": "EXAM-STUDENT-FULL::Q-1",
            "question_number": 1,
            "eval_status": "evaluated",
            "answer_state": "detected",
        }] + [
            {
                "response_id": f"RESP-STUDENT-FULL-{number}",
                "submission_id": "SUB-STUDENT-FULL",
                "question_id": f"EXAM-STUDENT-FULL::Q-{number}",
                "question_number": number,
                "eval_status": "not_attempted",
                "answer_state": "not_attempted",
                "is_missing_response": True,
                "absence_proven": True,
                "detected_text": "",
                "source_pages": [],
                "question_assignment": {
                    "method": "not_attempted",
                    "absence_proof": {"verified": True, "method": "document_answer_mapping"},
                },
            }
            for number in (2, 3)
        ]
    )
    await db["evalpen_evaluations"].insert_one(
        {
            "evaluation_id": "EVAL-STUDENT-FULL-1",
            "response_id": "RESP-STUDENT-FULL-1",
            "total_score": 4.0,
            "max_score": 4.0,
            "overall_feedback": "Correct.",
            "reference_solution": "u_x = 16 m/s, u_y = 12 m/s.",
            "teacher_feedback": "Keep showing your working clearly.",
            "criterion_marks": [
                {
                    "criterion_id": "internal-only-id",
                    "description": "Resolve the initial velocity into components",
                    "marks_awarded": 1.0,
                    "max_marks": 1.0,
                    "rationale": "Both components are correct.",
                    "evidence": "Internal OCR evidence must not be exposed.",
                },
                {
                    "criterion_id": "internal-only-id-2",
                    "description": "Calculate the time of flight",
                    "marks_awarded": 3.0,
                    "max_marks": 3.0,
                    "rationale": "Correct formula and substitution.",
                },
            ],
        }
    )
    await db["evalpen_evaluations"].insert_many(
        [
            {
                "evaluation_id": f"EVAL-STUDENT-FULL-{number}",
                "response_id": f"RESP-STUDENT-FULL-{number}",
                "question_id": f"EXAM-STUDENT-FULL::Q-{number}",
                "total_score": 0.0,
                "max_score": 4.0,
                "overall_feedback": "Not attempted.",
                "manual_review_required": False,
            }
            for number in (2, 3)
        ]
    )
    from services.exampen_submission_readiness import build_publication_snapshot

    snapshot = await build_publication_snapshot(
        db, "SUB-STUDENT-FULL", actor_id="TEACHER-1"
    )
    snapshot.pop("published_at_dt")
    await db["evalpen_submissions"].update_one(
        {"submission_id": "SUB-STUDENT-FULL"},
        {"$set": {"publication_snapshot": snapshot}},
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
    first_question = result.questions[0]
    assert first_question.question_number == 1
    assert first_question.reference_answer == "u_x = 16 m/s, u_y = 12 m/s."
    assert first_question.teacher_feedback == "Keep showing your working clearly."
    assert [row.description for row in first_question.mark_breakdown] == [
        "Resolve the initial velocity into components",
        "Calculate the time of flight",
    ]
    assert [row.marks_awarded for row in first_question.mark_breakdown] == [1.0, 3.0]
    assert all("evidence" not in row.model_dump() for row in first_question.mark_breakdown)


def test_student_mark_breakdown_supports_legacy_step_marks_without_leaking_evidence():
    from api.v1.evalpen_student_bff_async import _student_mark_breakdown

    breakdown = _student_mark_breakdown(
        {
            "step_marks": [
                {
                    "step": "State the relevant formula",
                    "marks_awarded": 1.0,
                    "marks_possible": 2.0,
                    "justification": "The formula is correct but substitution is missing.",
                    "evidence": "Internal OCR excerpt",
                }
            ]
        }
    )

    assert breakdown == [
        {
            "description": "State the relevant formula",
            "marks_awarded": 1.0,
            "max_marks": 2.0,
            "feedback": "The formula is correct but substitution is missing.",
        }
    ]
