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
            # Response status and evaluation persistence are deliberately
            # separate writes.  A saved evaluation must win over this stale
            # OCR-era status in the teacher workspace.
            "eval_status": "ready_with_warnings",
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
    assert result.score_state == "available"
    assert result.evaluated_count == 1
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
async def test_manual_review_score_remains_visible_and_counted_as_evaluated():
    from api.v1.evalpen_review_async import get_submission_summary

    db = _fresh_db()
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-REVIEW-SCORE",
            "exam_id": "EXAM-REVIEW-SCORE",
            "student_id": "STU-1",
            "source": "camera",
            "segmentation_status": "complete",
            "review_state": "needs_review",
        }
    )
    await db["evalpen_questions"].insert_one(
        {
            "question_id": "EXAM-REVIEW-SCORE::Q-1",
            "exam_id": "EXAM-REVIEW-SCORE",
            "question_number": 1,
            "max_marks": 4,
        }
    )
    await db["evalpen_detected_responses"].insert_one(
        {
            "response_id": "RESP-REVIEW-SCORE",
            "submission_id": "SUB-REVIEW-SCORE",
            "question_id": "EXAM-REVIEW-SCORE::Q-1",
            "question_number": 1,
            "detected_text": "A visually scored answer",
            "eval_status": "manual_review",
            "manual_review_required": True,
            "flags": [],
        }
    )
    await db["evalpen_evaluations"].insert_one(
        {
            "evaluation_id": "EVAL-REVIEW-SCORE",
            "response_id": "RESP-REVIEW-SCORE",
            "question_id": "EXAM-REVIEW-SCORE::Q-1",
            "eval_path": "full_document_visual",
            "total_score": 3.0,
            "max_score": 4.0,
            "manual_review_required": True,
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
            submission_id="SUB-REVIEW-SCORE",
            current_user=_admin_user(),
            db=None,
        )

    assert result.total_score == 3.0
    assert result.evaluated_count == 1
    assert result.pending_count == 0
    assert result.review_count == 1
    assert result.score_state == "available"
    assert result.responses[0].eval_path == "full_document_visual"


@pytest.mark.asyncio
async def test_review_summary_keeps_full_denominator_but_blocks_unproven_blanks():
    """Historical missing rows remain unresolved instead of becoming false zeros."""
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
    assert result.score_state == "available"
    assert result.evaluated_count == 1
    assert result.blocked_count == 2
    assert [response.question_number for response in result.responses] == [1, 2, 3]
    assert [response.answer_state for response in result.responses] == [None, "unresolved", "unresolved"]
    assert [response.total_score for response in result.responses[1:]] == [None, None]
    assert [response.max_score for response in result.responses[1:]] == [4.0, 4.0]


@pytest.mark.asyncio
async def test_review_summary_stays_processing_until_each_detected_answer_has_evaluation():
    """OCR completion alone must never expose a transient final zero score."""
    from api.v1.evalpen_review_async import get_submission_summary

    db = _fresh_db()
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-WAIT-FOR-EVAL",
            "exam_id": "EXAM-WAIT-FOR-EVAL",
            "student_id": "STU-1",
            "source": "web_upload",
            "segmentation_status": "complete",
        }
    )
    await db["evalpen_questions"].insert_one(
        {
            "question_id": "EXAM-WAIT-FOR-EVAL::Q-1",
            "exam_id": "EXAM-WAIT-FOR-EVAL",
            "question_number": 1,
            "max_marks": 4,
        }
    )
    await db["evalpen_detected_responses"].insert_one(
        {
            "response_id": "RESP-WAIT-FOR-EVAL-1",
            "submission_id": "SUB-WAIT-FOR-EVAL",
            "question_id": "EXAM-WAIT-FOR-EVAL::Q-1",
            "question_number": 1,
            "content_type": "TEXT_ONLY",
            "detected_text": "Student answer awaiting AI marking",
            "eval_status": "ready_with_warnings",
            "flags": [],
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
            submission_id="SUB-WAIT-FOR-EVAL",
            current_user=_admin_user(),
            db=None,
        )

    assert result.score_state == "processing"
    assert result.total_score == 0.0
    assert result.evaluated_count == 0
    assert result.pending_count == 1
    assert result.responses[0].total_score is None


@pytest.mark.asyncio
async def test_failed_ocr_does_not_look_like_a_zero_mark_blank_paper():
    """A storage/OCR failure must remain retryable, not become ten fake zeros."""
    from api.v1.evalpen_review_async import get_submission_summary

    db = _fresh_db()
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-OCR-FAILED",
            "exam_id": "EXAM-OCR-FAILED",
            "student_id": "STU-1",
            "source": "camera",
            "segmentation_status": "failed",
        }
    )
    await db["evalpen_questions"].insert_many(
        [
            {
                "question_id": f"EXAM-OCR-FAILED::Q-{number}",
                "exam_id": "EXAM-OCR-FAILED",
                "question_number": number,
                "max_marks": 4,
            }
            for number in (1, 2, 3)
        ]
    )
    await db["exampen_processing_jobs"].insert_one(
        {
            "submission_id": "SUB-OCR-FAILED",
            "status": "failed",
            "last_error": "OCR produced no text blocks",
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
            submission_id="SUB-OCR-FAILED",
            current_user=_admin_user(),
            db=None,
        )

    assert result.score_state == "unavailable"
    assert result.processing_status == "failed"
    assert result.processing_error == "OCR produced no text blocks"
    assert result.responses == []
    assert result.evaluated_count == 0
    assert result.total_max_score == 12.0


@pytest.mark.asyncio
async def test_failed_reprocess_keeps_last_committed_score_visible():
    from api.v1.evalpen_review_async import get_submission_summary

    db = _fresh_db()
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-RETAINED",
            "exam_id": "EXAM-RETAINED",
            "student_id": "STU-1",
            "source": "camera",
            "segmentation_status": "complete",
            "document_grading_run_id": "DOCGR-active",
            "document_grading_materialization_id": "DOCGR-active:g0",
        }
    )
    await db["evalpen_questions"].insert_one(
        {
            "question_id": "EXAM-RETAINED::Q-1",
            "exam_id": "EXAM-RETAINED",
            "question_number": 1,
            "max_marks": 4,
        }
    )
    await db["evalpen_detected_responses"].insert_one(
        {
            "response_id": "RESP-RETAINED-1",
            "submission_id": "SUB-RETAINED",
            "question_id": "EXAM-RETAINED::Q-1",
            "question_number": 1,
            "eval_status": "evaluated",
            "flags": [],
        }
    )
    await db["evalpen_evaluations"].insert_one(
        {
            "evaluation_id": "EVAL-RETAINED-1",
            "response_id": "RESP-RETAINED-1",
            "question_id": "EXAM-RETAINED::Q-1",
            "total_score": 3,
            "max_score": 4,
            "manual_review_required": False,
        }
    )
    await db["exampen_processing_jobs"].insert_one(
        {
            "submission_id": "SUB-RETAINED",
            "status": "failed",
            "last_error": "new grading generation could not load the canonical paper",
            "active_result_retained": True,
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
            submission_id="SUB-RETAINED",
            current_user=_admin_user(),
            db=None,
        )

    assert result.score_state == "available"
    assert result.active_result_retained is True
    assert result.processing_status == "failed"
    assert result.processing_error is not None
    assert result.total_score == 3
    assert result.review_state == "ready"


@pytest.mark.asyncio
async def test_review_summary_keeps_unassigned_evidence_out_of_question_navigator():
    """An extra OCR segment must not become a fake twelfth paper question."""
    from api.v1.evalpen_review_async import get_submission_summary

    db = _fresh_db()
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-ELEVEN",
            "exam_id": "EXAM-ELEVEN",
            "student_id": "STU-1",
            "segmentation_status": "complete",
        }
    )
    await db["evalpen_questions"].insert_many(
        [
            {
                "question_id": f"EXAM-ELEVEN::Q-{number}",
                "exam_id": "EXAM-ELEVEN",
                "question_number": number,
                "question_text": f"Question {number}",
                "max_marks": 1,
            }
            for number in range(1, 12)
        ]
    )
    await db["evalpen_detected_responses"].insert_many(
        [
            {
                "response_id": "RESP-Q1",
                "submission_id": "SUB-ELEVEN",
                "question_id": "EXAM-ELEVEN::Q-1",
                "question_number": 1,
                "detected_text": "Answer one",
                "eval_status": "evaluated",
                "flags": [],
            },
            {
                "response_id": "RESP-EXTRA",
                "submission_id": "SUB-ELEVEN",
                "question_id": "EXAM-ELEVEN::Q-12",
                "question_number": 12,
                "detected_text": "Unmapped working from the margin",
                "source_pages": [
                    {"page_number": 4, "y_start": 120, "y_end": 260}
                ],
                "eval_status": "blocked",
                "flags": [],
            },
        ]
    )
    await db["evalpen_evaluations"].insert_one(
        {
            "evaluation_id": "EVAL-Q1",
            "response_id": "RESP-Q1",
            "question_id": "EXAM-ELEVEN::Q-1",
            "total_score": 1,
            "max_score": 1,
        }
    )
    await db["evalpen_answer_pages"].insert_many(
        [
            {
                "page_id": f"PAGE-{number}",
                "submission_id": "SUB-ELEVEN",
                "page_number": number,
            }
            for number in range(1, 5)
        ]
    )

    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async._get_tutor_scoped_student_ids",
            return_value=None,
        ),
    ):
        result = await get_submission_summary(
            submission_id="SUB-ELEVEN",
            current_user=_admin_user(),
            db=None,
        )

    assert len(result.question_catalog) == 11
    assert len(result.responses) == 11
    assert {item.question_number for item in result.responses} == set(range(1, 12))
    assert len(result.unassigned_responses) == 1
    assert result.unassigned_responses[0].response_id == "RESP-EXTRA"
    assert result.unassigned_responses[0].question_number == 12
    assert result.total_max_score == 11
    assert result.page_count == 4
    assert result.score_state == "available"


@pytest.mark.asyncio
async def test_exam_results_exclude_fake_questions_and_use_the_paper_denominator():
    from api.v1.evalpen_review_async import get_exam_results

    db = _fresh_db()
    await db["evalpen_questions"].insert_many(
        [
            {
                "question_id": f"EXAM-RESULTS::Q-{number}",
                "exam_id": "EXAM-RESULTS",
                "question_number": number,
                "max_marks": 2,
            }
            for number in (1, 2)
        ]
    )
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-RESULTS",
            "exam_id": "EXAM-RESULTS",
            "student_id": "STU-1",
        }
    )
    await db["evalpen_detected_responses"].insert_many(
        [
            {
                "response_id": "RESP-RESULTS-Q1",
                "submission_id": "SUB-RESULTS",
                "question_id": "EXAM-RESULTS::Q-1",
                "eval_status": "evaluated",
            },
            {
                "response_id": "RESP-RESULTS-FAKE-Q3",
                "submission_id": "SUB-RESULTS",
                "question_id": "EXAM-RESULTS::Q-3",
                "eval_status": "evaluated",
            },
        ]
    )
    await db["evalpen_evaluations"].insert_many(
        [
            {
                "evaluation_id": "EVAL-RESULTS-Q1",
                "response_id": "RESP-RESULTS-Q1",
                "total_score": 1.5,
                "max_score": 2,
            },
            {
                "evaluation_id": "EVAL-RESULTS-FAKE-Q3",
                "response_id": "RESP-RESULTS-FAKE-Q3",
                "total_score": 9,
                "max_score": 9,
            },
        ]
    )

    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async._get_tutor_scoped_student_ids",
            return_value=None,
        ),
    ):
        result = await get_exam_results(
            exam_id="EXAM-RESULTS",
            current_user=_admin_user(),
            db=None,
        )

    assert result.total_students == 1
    student = result.students[0]
    assert student.pcr_total_score == 1.5
    assert student.pcr_max_score == 4
    assert student.blocked_responses == 1


@pytest.mark.asyncio
async def test_staff_page_preview_uses_short_lived_private_s3_url():
    from api.v1.evalpen_review_async import get_submission_pages

    db = _fresh_db()
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-PREVIEW",
            "student_id": "STU-1",
        }
    )
    await db["evalpen_answer_pages"].insert_one(
        {
            "page_id": "PAGE-1",
            "submission_id": "SUB-PREVIEW",
            "page_number": 1,
            "raw_image_ref": (
                "s3://stoody-test/private/exampen/student-answer-copies/"
                "tenant/exam/attempt/page-1.png"
            ),
            "image_width_px": 1200,
            "image_height_px": 1600,
        }
    )
    await db["evalpen_detected_responses"].insert_one(
        {
            "response_id": "RESP-PREVIEW",
            "submission_id": "SUB-PREVIEW",
            "question_id": "EXAM-PREVIEW::Q-3",
            "question_number": 3,
            "source_pages": [
                {"page_number": 1, "y_start": 240, "y_end": 620}
            ],
            "eval_status": "evaluated",
        }
    )

    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async._get_tutor_scoped_student_ids",
            return_value=None,
        ),
        patch(
            "api.v1.evalpen_review_async.create_private_download_url",
            return_value="https://private.test/signed-preview",
        ) as presign,
    ):
        result = await get_submission_pages(
            submission_id="SUB-PREVIEW",
            current_user=_admin_user(),
            db=None,
        )

    assert result.total_pages == 1
    assert result.pages[0].image_url == "https://private.test/signed-preview"
    assert result.pages[0].width == 1200
    assert result.pages[0].regions == [
        {
            "page_number": 1,
            "y_start": 240,
            "y_end": 620,
            "response_id": "RESP-PREVIEW",
            "question_id": "EXAM-PREVIEW::Q-3",
            "question_number": 3,
            "answer_state": None,
        }
    ]
    assert presign.call_count == 1
    assert presign.call_args.kwargs["allowed_key_prefix"] == "private/exampen/"


@pytest.mark.asyncio
async def test_staff_llm_debug_endpoint_uses_normal_submission_scope():
    from api.v1.evalpen_review_async import get_submission_llm_debug

    db = _fresh_db()
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-LLM-DEBUG",
            "exam_id": "EXAM-DEBUG",
            "student_id": "STU-1",
        }
    )
    payload = {
        "submission_id": "SUB-LLM-DEBUG",
        "exam_id": "EXAM-DEBUG",
        "run": {"run_id": "RUN-DEBUG"},
        "security": {
            "answer_key_sent_to_llm": False,
            "api_credentials_exposed": False,
        },
        "pages": [],
    }

    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async._get_tutor_scoped_student_ids",
            return_value=None,
        ) as scope,
        patch(
            "services.exampen_llm_debug_service.build_submission_llm_debug_bundle",
            return_value=payload,
        ) as build,
    ):
        result = await get_submission_llm_debug(
            submission_id="SUB-LLM-DEBUG",
            current_user=_admin_user(),
            db=None,
        )

    assert result == payload
    assert scope.await_count == 1
    build.assert_awaited_once_with(db, "SUB-LLM-DEBUG")


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


@pytest.mark.asyncio
async def test_published_criterion_review_requires_explicit_amendment():
    """A normal edit request cannot silently mutate a released student result."""
    from fastapi import HTTPException
    from api.v1.evalpen_review_async import (
        CriterionMarkOverrideItem,
        CriterionMarksOverrideRequest,
        override_criterion_marks,
    )

    db = _fresh_db()
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-PUBLISHED-GUARD",
            "exam_id": "EXAM-PUBLISHED-GUARD",
            "student_id": "STU-1",
            "publication_status": "published",
        }
    )
    await db["evalpen_detected_responses"].insert_one(
        {
            "response_id": "RESP-PUBLISHED-GUARD",
            "submission_id": "SUB-PUBLISHED-GUARD",
            "student_id": "STU-1",
            "question_id": "EXAM-PUBLISHED-GUARD::Q-1",
        }
    )
    await db["evalpen_evaluations"].insert_one(
        {
            "evaluation_id": "EVAL-PUBLISHED-GUARD",
            "response_id": "RESP-PUBLISHED-GUARD",
            "student_id": "STU-1",
            "total_score": 1.0,
            "max_score": 2.0,
            "criterion_marks": [
                {
                    "criterion_id": "answer",
                    "description": "Correct answer",
                    "marks_awarded": 1.0,
                    "max_marks": 2.0,
                }
            ],
        }
    )

    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async._get_tutor_scoped_student_ids",
            return_value=None,
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await override_criterion_marks(
                evaluation_id="EVAL-PUBLISHED-GUARD",
                body=CriterionMarksOverrideRequest(
                    criteria=[
                        CriterionMarkOverrideItem(
                            criterion_id="answer",
                            marks_awarded=2.0,
                        )
                    ],
                    reason="The final answer is clearly visible",
                ),
                current_user=_admin_user(),
                db=None,
            )

    assert exc_info.value.status_code == 409
    stored = await db["evalpen_evaluations"].find_one(
        {"evaluation_id": "EVAL-PUBLISHED-GUARD"}
    )
    assert stored["total_score"] == 1.0


@pytest.mark.asyncio
async def test_published_criterion_review_releases_audited_revised_snapshot():
    """A confirmed correction stays published and preserves the prior release."""
    from api.v1.evalpen_review_async import (
        CriterionMarkOverrideItem,
        CriterionMarksOverrideRequest,
        override_criterion_marks,
    )
    from services.exampen_submission_readiness import (
        build_publication_snapshot,
        validate_publication_snapshot,
    )

    db = _fresh_db()
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "EXAM-PUBLISHED-AMEND",
            "paper_version_id": "PAPER-V1",
        }
    )
    await db["evalpen_questions"].insert_one(
        {
            "question_id": "EXAM-PUBLISHED-AMEND::Q-1",
            "exam_id": "EXAM-PUBLISHED-AMEND",
            "question_number": 1,
            "max_marks": 4.0,
        }
    )
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-PUBLISHED-AMEND",
            "exam_id": "EXAM-PUBLISHED-AMEND",
            "student_id": "STU-1",
            "publication_status": "published",
        }
    )
    await db["evalpen_detected_responses"].insert_one(
        {
            "response_id": "RESP-PUBLISHED-AMEND",
            "submission_id": "SUB-PUBLISHED-AMEND",
            "student_id": "STU-1",
            "question_id": "EXAM-PUBLISHED-AMEND::Q-1",
            "question_number": 1,
            "answer_state": "detected",
            "eval_status": "evaluated",
        }
    )
    await db["evalpen_evaluations"].insert_one(
        {
            "evaluation_id": "EVAL-PUBLISHED-AMEND",
            "response_id": "RESP-PUBLISHED-AMEND",
            "student_id": "STU-1",
            "total_score": 1.0,
            "max_score": 4.0,
            "criterion_marks": [
                {
                    "criterion_id": "method",
                    "description": "Uses the correct method",
                    "marks_awarded": 1.0,
                    "max_marks": 1.0,
                },
                {
                    "criterion_id": "answer",
                    "description": "Obtains the correct answer",
                    "marks_awarded": 0.0,
                    "max_marks": 3.0,
                },
            ],
        }
    )

    first_snapshot = await build_publication_snapshot(
        db,
        "SUB-PUBLISHED-AMEND",
        actor_id="admin-original",
    )
    original_published_at = first_snapshot.pop("published_at_dt")
    await db["evalpen_submissions"].update_one(
        {"submission_id": "SUB-PUBLISHED-AMEND"},
        {
            "$set": {
                "published_at": original_published_at,
                "published_by": "admin-original",
                "publication_snapshot": first_snapshot,
                "publication_snapshot_hash": first_snapshot["snapshot_hash"],
            }
        },
    )
    stored_original_published_at = (
        await db["evalpen_submissions"].find_one(
            {"submission_id": "SUB-PUBLISHED-AMEND"}
        )
    )["published_at"]

    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async._get_tutor_scoped_student_ids",
            return_value=None,
        ),
    ):
        result = await override_criterion_marks(
            evaluation_id="EVAL-PUBLISHED-AMEND",
            body=CriterionMarksOverrideRequest(
                criteria=[
                    CriterionMarkOverrideItem(
                        criterion_id="method",
                        marks_awarded=1.0,
                    ),
                    CriterionMarkOverrideItem(
                        criterion_id="answer",
                        marks_awarded=2.5,
                    ),
                ],
                reason="The final calculation is legible on page two",
                amend_published=True,
            ),
            current_user=_admin_user(),
            db=None,
        )

    assert result["publication_status"] == "published"
    assert result["result_revision"] == 1
    assert result["total_score"] == 3.5

    stored_submission = await db["evalpen_submissions"].find_one(
        {"submission_id": "SUB-PUBLISHED-AMEND"}
    )
    assert stored_submission["publication_status"] == "published"
    assert stored_submission["published_at"] == stored_original_published_at
    assert stored_submission["publication_snapshot"]["total_score"] == 3.5
    assert stored_submission["publication_snapshot_hash"] != first_snapshot["snapshot_hash"]
    assert stored_submission["result_revision"] == 1
    assert "review_mutation_lease_token" not in stored_submission
    assert validate_publication_snapshot(
        stored_submission["publication_snapshot"],
        submission_id="SUB-PUBLISHED-AMEND",
        exam_id="EXAM-PUBLISHED-AMEND",
    )

    history = stored_submission["publication_history"][-1]
    assert history["action"] == "published_score_amended"
    assert history["reason"] == "The final calculation is legible on page two"
    assert history["previous_snapshot"]["snapshot_hash"] == first_snapshot["snapshot_hash"]
    assert history["previous_total_score"] == 1.0
    assert history["total_score"] == 3.5


@pytest.mark.asyncio
async def test_published_total_override_uses_the_same_revision_contract():
    """Legacy non-rubric scores can be corrected without bypassing publication audit."""
    from api.v1.evalpen_review_async import (
        ScoreOverrideRequest,
        override_evaluation_score,
    )
    from services.exampen_submission_readiness import build_publication_snapshot

    db = _fresh_db()
    await db["exampen_exams"].insert_one(
        {"exam_id": "EXAM-PUBLISHED-TOTAL", "paper_version_id": "PAPER-V1"}
    )
    await db["evalpen_questions"].insert_one(
        {
            "question_id": "EXAM-PUBLISHED-TOTAL::Q-1",
            "exam_id": "EXAM-PUBLISHED-TOTAL",
            "question_number": 1,
            "max_marks": 5.0,
        }
    )
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-PUBLISHED-TOTAL",
            "exam_id": "EXAM-PUBLISHED-TOTAL",
            "student_id": "STU-1",
            "publication_status": "published",
        }
    )
    await db["evalpen_detected_responses"].insert_one(
        {
            "response_id": "RESP-PUBLISHED-TOTAL",
            "submission_id": "SUB-PUBLISHED-TOTAL",
            "student_id": "STU-1",
            "question_id": "EXAM-PUBLISHED-TOTAL::Q-1",
            "question_number": 1,
            "answer_state": "detected",
            "eval_status": "evaluated",
        }
    )
    await db["evalpen_evaluations"].insert_one(
        {
            "evaluation_id": "EVAL-PUBLISHED-TOTAL",
            "response_id": "RESP-PUBLISHED-TOTAL",
            "student_id": "STU-1",
            "total_score": 2.0,
            "max_score": 5.0,
        }
    )
    first_snapshot = await build_publication_snapshot(
        db,
        "SUB-PUBLISHED-TOTAL",
        actor_id="admin-original",
    )
    published_at = first_snapshot.pop("published_at_dt")
    await db["evalpen_submissions"].update_one(
        {"submission_id": "SUB-PUBLISHED-TOTAL"},
        {
            "$set": {
                "published_at": published_at,
                "published_by": "admin-original",
                "publication_snapshot": first_snapshot,
                "publication_snapshot_hash": first_snapshot["snapshot_hash"],
            }
        },
    )

    with (
        patch("api.v1.evalpen_review_async._get_tenant_db", return_value=db),
        patch(
            "api.v1.evalpen_review_async._get_tutor_scoped_student_ids",
            return_value=None,
        ),
    ):
        result = await override_evaluation_score(
            evaluation_id="EVAL-PUBLISHED-TOTAL",
            body=ScoreOverrideRequest(
                new_score=4.0,
                reason="Teacher confirmed the complete working",
                amend_published=True,
            ),
            current_user=_admin_user(),
            db=None,
        )

    assert result["new_score"] == 4.0
    assert result["publication_status"] == "published"
    assert result["result_revision"] == 1
    stored_submission = await db["evalpen_submissions"].find_one(
        {"submission_id": "SUB-PUBLISHED-TOTAL"}
    )
    assert stored_submission["publication_snapshot"]["total_score"] == 4.0
    assert stored_submission["publication_history"][-1]["previous_total_score"] == 2.0
    assert stored_submission["publication_history"][-1]["total_score"] == 4.0
