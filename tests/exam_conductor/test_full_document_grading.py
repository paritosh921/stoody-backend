from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest


def _fresh_db():
    from mongomock_motor import AsyncMongoMockClient

    return AsyncMongoMockClient()["skb_test"]


def _module():
    from api.v1._exampen_imports import load_exampen

    return load_exampen("pcr.services.full_document_grading")


class _FakeGate:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    async def call(self, model_id: str, prompt: str, caller_id: str, **kwargs: Any):
        self.calls.append(
            {
                "model_id": model_id,
                "prompt": prompt,
                "caller_id": caller_id,
                **kwargs,
            }
        )
        return SimpleNamespace(
            content=json.dumps(self.payload),
            usage=SimpleNamespace(
                model="gpt-5.1-2025-11-13",
                caller=caller_id,
                input_tokens=20_000,
                output_tokens=1_000,
                cache_read_tokens=12_000,
                total_tokens=21_000,
                estimated_cost_usd=0.1,
            ),
        )


async def _seed(db, *, submission_id: str = "SUB-DOC-1") -> None:
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": submission_id,
            "exam_id": "EXAM-DOC-1",
            "student_id": "STU-1",
            "source": "camera",
            "content_hash": "student-copy-hash",
            "segmentation_status": "pending",
        }
    )
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "EXAM-DOC-1",
            "exam_type": "pcr",
            "prepared_document_id": "paper-1",
            "paper_version_id": "paper-version-1",
            "paper_content_hash": "paper-content-hash",
        }
    )
    await db["exampen_paper_versions"].insert_one(
        {"paper_version_id": "paper-version-1", "document_id": "paper-1"}
    )
    await db["documents"].insert_one(
        {
            "document_id": "paper-1",
            "file_path": "question.pdf",
            "filename": "question.pdf",
            "answer_sheet_path": "solution.pdf",
            "answer_sheet_filename": "solution.pdf",
        }
    )
    await db["evalpen_questions"].insert_many(
        [
            {
                "question_id": "EXAM-DOC-1::Q1",
                "exam_id": "EXAM-DOC-1",
                "question_number": 1,
                "question_text": "Draw and label the circuit.",
                "reference_solution": "A correctly connected labelled circuit.",
                "max_marks": 2,
                "expects_diagram": True,
                "marking_criteria": [
                    {
                        "criterion_id": "diagram",
                        "description": "Correct circuit and labels",
                        "max_marks": 2,
                    }
                ],
            },
            {
                "question_id": "EXAM-DOC-1::Q2",
                "exam_id": "EXAM-DOC-1",
                "question_number": 2,
                "question_text": "Explain the observation.",
                "reference_solution": "A correct explanation.",
                "max_marks": 2,
                "marking_criteria": [
                    {
                        "criterion_id": "explain",
                        "description": "Correct explanation",
                        "max_marks": 2,
                    }
                ],
            },
        ]
    )
    await db["evalpen_answer_pages"].insert_one(
        {
            "page_id": "PAGE-1",
            "submission_id": submission_id,
            "page_number": 1,
            "raw_image_ref": "private-page.jpg",
            "content_hash": "page-hash",
        }
    )


def _document_review(*, complete: bool = True, confidence: float = 0.97):
    return {
        "all_student_work_accounted": complete,
        "confidence": confidence,
        "warnings": [],
    }


def _attempted_diagram():
    return {
        "question_number": 1,
        "attempt_status": "attempted",
        "confidence": 0.96,
        "student_answer": "Student drew a battery, switch, lamp and connecting wires.",
        "content_type": "DIAGRAM_HEAVY",
        "evidence_regions": [
            {
                "page_number": 1,
                "y_start": 100,
                "y_end": 520,
                "evidence": "Visible labelled circuit diagram",
            }
        ],
        "criterion_marks": [
            {
                "criterion_id": "diagram",
                "marks_awarded": 2,
                "rationale": "Circuit and labels are correct.",
                "evidence": "Page 1 visible labelled circuit",
            }
        ],
        "total_score": 2,
        "overall_feedback": "Correct diagram.",
        "needs_review": False,
        "review_reason": "",
    }


async def _grade(monkeypatch, db, payload):
    module = _module()
    monkeypatch.setattr(
        module,
        "_read_canonical_file",
        lambda *args, **kwargs: _async_value(b"%PDF-1.4 canonical"),
    )
    monkeypatch.setattr(
        module,
        "_student_copy_content",
        lambda pages: _async_value(
            ([{"type": "input_text", "text": "student page fixture"}], 100)
        ),
    )
    gate = _FakeGate(payload)
    service = module.FullDocumentGradingService(
        db,
        gate,
        model_id="gpt-5.1-2025-11-13",
    )
    return await service.grade_submission("SUB-DOC-1"), gate


async def _async_value(value):
    return value


@pytest.mark.asyncio
async def test_full_document_grading_keeps_diagram_visual_and_missing_question_unresolved(
    monkeypatch,
):
    db = _fresh_db()
    await _seed(db)
    result, gate = await _grade(
        monkeypatch,
        db,
        {
            "document_review": _document_review(),
            # Q2 is deliberately omitted. It must not become an automatic zero.
            "questions": [_attempted_diagram()],
        },
    )

    assert result.handled is True
    assert result.status == "blocked_for_review"
    assert result.evaluated_count == 1
    assert result.blocked_count == 1
    assert len(gate.calls) == 1
    assert gate.calls[0]["responses_input"]
    assert gate.calls[0]["json_schema"]["properties"]["questions"]
    assert gate.calls[0]["prompt_cache_key"].startswith("pcr-paper-")

    q1 = await db["evalpen_detected_responses"].find_one(
        {"question_id": "EXAM-DOC-1::Q1", "superseded_at": {"$exists": False}}
    )
    q2 = await db["evalpen_detected_responses"].find_one(
        {"question_id": "EXAM-DOC-1::Q2", "superseded_at": {"$exists": False}}
    )
    assert q1["content_type"] == "DIAGRAM_HEAVY"
    assert q1["answer_state"] == "detected"
    assert q2["answer_state"] == "unresolved"
    assert q2["eval_status"] == "blocked"
    assert q2["is_missing_response"] is False
    assert await db["evalpen_evaluations"].count_documents(
        {"question_id": "EXAM-DOC-1::Q2"}
    ) == 0


@pytest.mark.asyncio
async def test_not_attempted_zero_requires_explicit_high_confidence_full_copy_proof(
    monkeypatch,
):
    db = _fresh_db()
    await _seed(db)
    result, _gate = await _grade(
        monkeypatch,
        db,
        {
            "document_review": _document_review(complete=True, confidence=0.97),
            "questions": [
                _attempted_diagram(),
                {
                    "question_number": 2,
                    "attempt_status": "not_attempted",
                    "confidence": 0.96,
                    "student_answer": "",
                    "content_type": "TEXT_ONLY",
                    "evidence_regions": [],
                    "criterion_marks": [],
                    "total_score": 0,
                    "overall_feedback": "Question not attempted.",
                    "needs_review": False,
                    "review_reason": "",
                },
            ],
        },
    )

    assert result.status == "completed"
    assert result.evaluated_count == 2
    assert result.blocked_count == 0
    q2 = await db["evalpen_detected_responses"].find_one(
        {"question_id": "EXAM-DOC-1::Q2", "superseded_at": {"$exists": False}}
    )
    evaluation = await db["evalpen_evaluations"].find_one(
        {"response_id": q2["response_id"]}
    )
    assert q2["answer_state"] == "not_attempted"
    assert q2["absence_proven"] is True
    assert evaluation["total_score"] == 0
    assert evaluation["model_used"] == "gpt-5.1-2025-11-13"


@pytest.mark.asyncio
async def test_invalid_criterion_award_is_blocked_instead_of_clamped_or_scored(
    monkeypatch,
):
    db = _fresh_db()
    await _seed(db)
    invalid = _attempted_diagram()
    invalid["criterion_marks"][0]["marks_awarded"] = 5
    invalid["total_score"] = 5
    result, _gate = await _grade(
        monkeypatch,
        db,
        {
            "document_review": _document_review(),
            "questions": [
                invalid,
                {
                    "question_number": 2,
                    "attempt_status": "not_attempted",
                    "confidence": 0.96,
                    "student_answer": "",
                    "content_type": "TEXT_ONLY",
                    "evidence_regions": [],
                    "criterion_marks": [],
                    "total_score": 0,
                    "overall_feedback": "Question not attempted.",
                    "needs_review": False,
                    "review_reason": "",
                },
            ],
        },
    )

    assert result.status == "blocked_for_review"
    q1 = await db["evalpen_detected_responses"].find_one(
        {"question_id": "EXAM-DOC-1::Q1", "superseded_at": {"$exists": False}}
    )
    assert q1["answer_state"] == "unresolved"
    assert "outside its locked range" in q1["manual_review_reason"]
    assert await db["evalpen_evaluations"].count_documents(
        {"question_id": "EXAM-DOC-1::Q1"}
    ) == 0
