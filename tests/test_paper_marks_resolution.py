from __future__ import annotations

import pytest
from fastapi import HTTPException
from mongomock_motor import AsyncMongoMockClient


async def _seed_mismatched_paper(db) -> None:
    await db.documents.insert_one(
        {
            "document_id": "paper-1",
            "document_type": "Test Series",
            "exam_mode": "pcr",
            "exam_finalized": False,
            "total_points": 4,
            "total_points_source": "visual_question_marks",
            "marks_extraction_summary": {
                "expected_total": 3,
                "calculated_total": 4,
                "reconciled": False,
            },
        }
    )
    for number in (1, 2):
        question = {
                "id": f"q-{number}",
                "document_id": "paper-1",
                "question_number": number,
                "text": f"Question {number}",
                "points": 2,
                "metadata": {
                    "marks_status": "verified",
                    "marks_source": "visual_printed_evidence",
                    "paper_marks_summary": {
                        "expected_total": 3,
                        "calculated_total": 4,
                        "reconciled": False,
                    },
                    "paper_marks_reconciled": False,
                },
            }
        if number == 2:
            question.update(
                {
                    "assessment_units": [{"unit_id": "u-1", "max_marks": 2}],
                    "marking_criteria": [{"criterion_id": "c-1", "max_marks": 2}],
                    "reference_solution": "Existing answer",
                    "marking_plan_generation_status": "completed",
                }
            )
        await db.questions.insert_one(question)


async def _current_fingerprint(db) -> str:
    from services.question_paper_marks_contract import (
        expected_paper_total,
        paper_marks_issue_fingerprint,
    )

    document = await db.documents.find_one({"document_id": "paper-1"})
    questions = await db.questions.find({"document_id": "paper-1"}).to_list(length=100)
    return paper_marks_issue_fingerprint(
        "paper-1",
        questions,
        expected_total=expected_paper_total(document, questions),
    )


@pytest.mark.asyncio
async def test_confirm_question_total_reconciles_document_questions_and_audit():
    from api.v1.pdf_async import (
        PaperMarksResolutionRequest,
        _resolve_paper_total_in_session,
    )

    db = AsyncMongoMockClient()["paper_marks_resolution"]
    await _seed_mismatched_paper(db)
    fingerprint = await _current_fingerprint(db)

    response = await _resolve_paper_total_in_session(
        db,
        document_id="paper-1",
        body=PaperMarksResolutionRequest(
            action="confirm_question_total",
            issue_fingerprint=fingerprint,
            note="Header total was incorrect",
        ),
        actor_id="teacher-1",
    )

    document = await db.documents.find_one({"document_id": "paper-1"})
    questions = await db.questions.find({"document_id": "paper-1"}).to_list(length=100)
    assert response["marks_summary"]["reconciled"] is True
    assert document["total_points"] == 4
    assert document["total_points_source"] == "teacher"
    assert document["marks_review_required"] is False
    assert document["authoring_revision"] == 1
    assert document["paper_marks_resolution"]["resolved_by"] == "teacher-1"
    assert document["paper_marks_resolution_history"][0]["note"] == "Header total was incorrect"
    assert all(question["metadata"]["paper_marks_reconciled"] is True for question in questions)
    assert all(
        question["metadata"]["paper_marks_summary"]["expected_total"] == 4
        for question in questions
    )


@pytest.mark.asyncio
async def test_stale_marks_issue_cannot_overwrite_new_question_marks():
    from api.v1.pdf_async import (
        PaperMarksResolutionRequest,
        _resolve_paper_total_in_session,
    )

    db = AsyncMongoMockClient()["paper_marks_stale"]
    await _seed_mismatched_paper(db)
    stale_fingerprint = await _current_fingerprint(db)
    await db.questions.update_one({"id": "q-2"}, {"$set": {"points": 1}})

    with pytest.raises(HTTPException) as exc:
        await _resolve_paper_total_in_session(
            db,
            document_id="paper-1",
            body=PaperMarksResolutionRequest(
                action="confirm_question_total",
                issue_fingerprint=stale_fingerprint,
            ),
            actor_id="teacher-1",
        )

    assert exc.value.status_code == 409
    assert exc.value.detail["code"] == "STALE_READINESS_ISSUE"
    document = await db.documents.find_one({"document_id": "paper-1"})
    assert document["total_points_source"] == "visual_question_marks"
    assert "paper_marks_resolution" not in document


@pytest.mark.asyncio
async def test_question_mark_correction_resolves_printed_total_atomically():
    from api.v1.pdf_async import (
        PaperMarksQuestionCorrection,
        PaperMarksResolutionRequest,
        _resolve_paper_total_in_session,
    )

    db = AsyncMongoMockClient()["paper_marks_correction"]
    await _seed_mismatched_paper(db)
    fingerprint = await _current_fingerprint(db)

    response = await _resolve_paper_total_in_session(
        db,
        document_id="paper-1",
        body=PaperMarksResolutionRequest(
            action="save_question_marks",
            issue_fingerprint=fingerprint,
            question_marks=[
                PaperMarksQuestionCorrection(question_id="q-2", points=1),
            ],
            note="Q2 is printed as 1 mark",
        ),
        actor_id="teacher-1",
    )

    document = await db.documents.find_one({"document_id": "paper-1"})
    q1 = await db.questions.find_one({"id": "q-1"})
    q2 = await db.questions.find_one({"id": "q-2"})
    assert response["marks_summary"]["reconciled"] is True
    assert response["resolution"]["question_mark_changes"] == [
        {
            "question_id": "q-2",
            "question_number": 2,
            "previous_points": 2.0,
            "corrected_points": 1.0,
        }
    ]
    assert document["total_points"] == 3
    assert document["total_points_source"] == "teacher"
    assert document["marks_review_required"] is False
    assert document["authoring_revision"] == 1
    assert q1["points"] == 2
    assert q1["metadata"]["paper_marks_reconciled"] is True
    assert q2["points"] == 1
    assert q2["metadata"]["marks_status"] == "teacher_confirmed"
    assert q2["assessment_units"] == []
    assert q2["marking_criteria"] == []
    assert q2["reference_solution"] is None
    assert q2["marking_plan_generation_status"] == "not_generated"


@pytest.mark.asyncio
async def test_question_mark_correction_must_match_paper_total_before_any_write():
    from api.v1.pdf_async import (
        PaperMarksQuestionCorrection,
        PaperMarksResolutionRequest,
        _resolve_paper_total_in_session,
    )

    db = AsyncMongoMockClient()["paper_marks_incomplete_correction"]
    await _seed_mismatched_paper(db)
    fingerprint = await _current_fingerprint(db)

    with pytest.raises(HTTPException) as exc:
        await _resolve_paper_total_in_session(
            db,
            document_id="paper-1",
            body=PaperMarksResolutionRequest(
                action="save_question_marks",
                issue_fingerprint=fingerprint,
                question_marks=[
                    PaperMarksQuestionCorrection(question_id="q-2", points=1.5),
                ],
            ),
            actor_id="teacher-1",
        )

    assert exc.value.status_code == 422
    assert exc.value.detail["code"] == "QUESTION_MARKS_STILL_MISMATCH"
    document = await db.documents.find_one({"document_id": "paper-1"})
    q2 = await db.questions.find_one({"id": "q-2"})
    assert document["total_points"] == 4
    assert "paper_marks_resolution" not in document
    assert q2["points"] == 2
    assert q2["assessment_units"] != []


@pytest.mark.asyncio
async def test_finalized_paper_total_cannot_be_resolved_in_place():
    from api.v1.pdf_async import (
        PaperMarksResolutionRequest,
        _resolve_paper_total_in_session,
    )

    db = AsyncMongoMockClient()["paper_marks_finalized"]
    await _seed_mismatched_paper(db)
    fingerprint = await _current_fingerprint(db)
    await db.documents.update_one(
        {"document_id": "paper-1"},
        {"$set": {"exam_finalized": True}},
    )

    with pytest.raises(HTTPException) as exc:
        await _resolve_paper_total_in_session(
            db,
            document_id="paper-1",
            body=PaperMarksResolutionRequest(
                action="confirm_question_total",
                issue_fingerprint=fingerprint,
            ),
            actor_id="teacher-1",
        )

    assert exc.value.status_code == 409
