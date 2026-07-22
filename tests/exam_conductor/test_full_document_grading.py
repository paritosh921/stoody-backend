from __future__ import annotations

import asyncio
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


class _SlowFakeGate(_FakeGate):
    async def call(self, model_id: str, prompt: str, caller_id: str, **kwargs: Any):
        await asyncio.sleep(0.05)
        return await super().call(model_id, prompt, caller_id, **kwargs)


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
                        "acceptable_evidence": (
                            "A closed circuit with a battery, switch, lamp, connecting "
                            "wires, and readable labels."
                        ),
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
                        "acceptable_evidence": (
                            "Explains that the observation follows from the complete circuit."
                        ),
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


async def _seed_duplicate_submission(db) -> None:
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-DOC-2",
            "exam_id": "EXAM-DOC-1",
            "student_id": "STU-2",
            "source": "camera",
            "content_hash": "student-copy-hash",
            "segmentation_status": "pending",
        }
    )
    await db["evalpen_answer_pages"].insert_one(
        {
            "page_id": "PAGE-2",
            "submission_id": "SUB-DOC-2",
            "page_number": 1,
            "raw_image_ref": "private-page-copy.jpg",
            "content_hash": "page-hash",
        }
    )


def _document_review(*, complete: bool = True, confidence: float = 0.97):
    return {
        "all_student_work_accounted": complete,
        "confidence": confidence,
        "warnings": [],
    }


def _method_analysis(
    *,
    classification: str = "reference_method",
    validity: str = "valid",
    requirement_satisfied: bool = True,
):
    return {
        "detected_method": "A visible, internally coherent student method.",
        "method_classification": classification,
        "method_validity": validity,
        "method_requirement_satisfied": requirement_satisfied,
        "confidence": 0.95,
        "explanation": "The visible working establishes the locked criterion.",
        "error_carried_forward": "not_applicable",
        "error_carried_forward_reason": "",
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
        "method_analysis": _method_analysis(),
        "criterion_marks": [
            {
                "criterion_id": "diagram",
                "decision": "met",
                "confidence": 0.96,
                "marks_awarded": 2,
                "rationale": "Circuit and labels are correct.",
                "evidence": "Page 1 visible labelled circuit",
                "missing_evidence": "",
                "credit_basis": "direct_evidence",
            }
        ],
        "total_score": 2,
        "overall_feedback": "Correct diagram.",
        "needs_review": False,
        "review_reason": "",
    }


def _attempted_explanation():
    return {
        "question_number": 2,
        "attempt_status": "attempted",
        "confidence": 0.95,
        "student_answer": "The observation follows because the circuit is complete.",
        "content_type": "TEXT_ONLY",
        "evidence_regions": [
            {"page_number": 1, "y_start": 540, "y_end": 850, "evidence": "Text"}
        ],
        "method_analysis": _method_analysis(),
        "criterion_marks": [
            {
                "criterion_id": "explain",
                "decision": "met",
                "confidence": 0.95,
                "marks_awarded": 2,
                "rationale": "The explanation is correct.",
                "evidence": "Page 1 explanation",
                "missing_evidence": "",
                "credit_basis": "direct_evidence",
            }
        ],
        "total_score": 2,
        "overall_feedback": "Correct explanation.",
        "needs_review": False,
        "review_reason": "",
    }


async def _grade(monkeypatch, db, payload, *, submission_id="SUB-DOC-1"):
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
    return await service.grade_submission(submission_id), gate


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
    assert result.status == "completed"
    assert result.review_state == "blocked"
    assert result.evaluated_count == 1
    assert result.blocked_count == 1
    assert len(gate.calls) == 1
    assert gate.calls[0]["responses_input"]
    assert gate.calls[0]["json_schema"]["properties"]["questions"]
    method_schema = gate.calls[0]["json_schema"]["properties"]["questions"]["items"][
        "properties"
    ]["method_analysis"]
    assert "alternative_method" in method_schema["properties"][
        "method_classification"
    ]["enum"]
    assert "valid_alternative" not in method_schema["properties"][
        "method_classification"
    ]["enum"]
    assert "method_requirement_satisfied" not in method_schema["properties"]
    assert gate.calls[0]["temperature"] == 0.10
    assert gate.calls[0]["reasoning_effort"] == "medium"
    assert gate.calls[0]["prompt_cache_key"].startswith("pcr-paper-")
    static_catalog = gate.calls[0]["responses_input"][1]["content"][0]["text"]
    assert "acceptable_evidence" in static_catalog
    assert "A closed circuit with a battery" in static_catalog
    assert "marking_standard" in static_catalog

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

    assert result.status == "completed"
    assert result.review_state == "blocked"
    q1 = await db["evalpen_detected_responses"].find_one(
        {"question_id": "EXAM-DOC-1::Q1", "superseded_at": {"$exists": False}}
    )
    assert q1["answer_state"] == "unresolved"
    assert "outside its locked range" in q1["manual_review_reason"]
    assert await db["evalpen_evaluations"].count_documents(
        {"question_id": "EXAM-DOC-1::Q1"}
    ) == 0


@pytest.mark.asyncio
async def test_criterion_decision_and_award_must_be_consistent(monkeypatch):
    db = _fresh_db()
    await _seed(db)
    contradictory = _attempted_diagram()
    contradictory["criterion_marks"][0]["decision"] = "met"
    contradictory["criterion_marks"][0]["marks_awarded"] = 1
    contradictory["total_score"] = 1

    result, _gate = await _grade(
        monkeypatch,
        db,
        {
            "document_review": _document_review(),
            "questions": [contradictory, _attempted_explanation()],
        },
    )

    assert result.review_state == "blocked"
    q1 = await db["evalpen_detected_responses"].find_one(
        {"question_id": "EXAM-DOC-1::Q1", "superseded_at": {"$exists": False}}
    )
    assert q1["answer_state"] == "unresolved"
    assert "met but was not awarded its full locked mark" in q1["manual_review_reason"]
    assert await db["evalpen_evaluations"].count_documents(
        {"question_id": "EXAM-DOC-1::Q1"}
    ) == 0


@pytest.mark.asyncio
async def test_low_confidence_criterion_is_review_gated(monkeypatch):
    db = _fresh_db()
    await _seed(db)
    uncertain = _attempted_diagram()
    uncertain["criterion_marks"][0]["confidence"] = 0.60

    result, _gate = await _grade(
        monkeypatch,
        db,
        {
            "document_review": _document_review(),
            "questions": [uncertain, _attempted_explanation()],
        },
    )

    assert result.review_state == "blocked"
    q1 = await db["evalpen_detected_responses"].find_one(
        {"question_id": "EXAM-DOC-1::Q1", "superseded_at": {"$exists": False}}
    )
    assert q1["answer_state"] == "unresolved"
    assert "sufficient confidence" in q1["manual_review_reason"]


@pytest.mark.asyncio
async def test_document_warning_is_one_submission_gate_not_every_question(
    monkeypatch,
):
    db = _fresh_db()
    await _seed(db)
    review = _document_review()
    review["warnings"] = ["A page edge is faint; confirm that no work is cropped."]
    result, _gate = await _grade(
        monkeypatch,
        db,
        {
            "document_review": review,
            "questions": [_attempted_diagram(), _attempted_explanation()],
        },
    )

    assert result.status == "completed"
    assert result.review_state == "needs_review"
    assert result.document_review_required is True
    assert result.warning_count == 1
    responses = await db["evalpen_detected_responses"].find(
        {"submission_id": "SUB-DOC-1"}
    ).to_list(length=10)
    assert len(responses) == 2
    assert all(response["manual_review_required"] is False for response in responses)
    assert all(response["eval_status"] == "evaluated" for response in responses)
    submission = await db["evalpen_submissions"].find_one(
        {"submission_id": "SUB-DOC-1"}
    )
    assert submission["document_review"]["required"] is True


@pytest.mark.asyncio
async def test_identical_bytes_for_different_students_are_graded_independently(
    monkeypatch,
):
    db = _fresh_db()
    await _seed(db)
    await _seed_duplicate_submission(db)
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
    gate = _FakeGate(
        {
            "document_review": _document_review(),
            "questions": [_attempted_diagram(), _attempted_explanation()],
        }
    )
    service = module.FullDocumentGradingService(
        db,
        gate,
        model_id="gpt-5.1",
    )

    first = await service.grade_submission("SUB-DOC-1")
    second = await service.grade_submission("SUB-DOC-2")

    assert first.run_id != second.run_id
    assert len(gate.calls) == 2
    assert gate.calls[0]["model_id"] == "gpt-5.1"
    assert gate.calls[1]["model_id"] == "gpt-5.1-2025-11-13"
    frozen_exam = await db["exampen_exams"].find_one({"exam_id": "EXAM-DOC-1"})
    assert frozen_exam["pcr_grading_contract"] == {
        "prompt_version": "pcr-full-document-visual-v4",
        "model_id": "gpt-5.1-2025-11-13",
        "temperature": 0.10,
        "reasoning_effort": "medium",
        "locked_at": frozen_exam["pcr_grading_contract"]["locked_at"],
    }
    # Provider-side prefix caching is safe: only the immutable paper, solution,
    # and rubric prefix share this key. Student outputs and run ids do not.
    assert gate.calls[0]["prompt_cache_key"] == gate.calls[1]["prompt_cache_key"]
    first_rows = await db["evalpen_detected_responses"].find(
        {"submission_id": "SUB-DOC-1"}
    ).sort("question_number", 1).to_list(length=10)
    second_rows = await db["evalpen_detected_responses"].find(
        {"submission_id": "SUB-DOC-2"}
    ).sort("question_number", 1).to_list(length=10)
    assert {row["response_id"] for row in first_rows}.isdisjoint(
        {row["response_id"] for row in second_rows}
    )
    runs = await db["evalpen_document_grading_runs"].find({}).to_list(length=10)
    assert len(runs) == 2
    assert {run["submission_id"] for run in runs} == {"SUB-DOC-1", "SUB-DOC-2"}
    second_submission = await db["evalpen_submissions"].find_one(
        {"submission_id": "SUB-DOC-2"}
    )
    assert second_submission["resumed_grading_run"] is False
    assert "reused_grading_input" not in second_submission


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("legacy_contract", "expected_temperature", "expected_reasoning_effort"),
    [
        (
            {
                "prompt_version": "pcr-full-document-visual-v4",
                "model_id": "gpt-5.1-2025-11-13",
                "temperature": 0.35,
                "locked_at": "legacy-lock",
            },
            0.35,
            "medium",
        ),
        (
            {
                "prompt_version": "pcr-full-document-visual-v4",
                "model_id": "gpt-5.1-2025-11-13",
                "reasoning_effort": "low",
                "locked_at": "legacy-lock",
            },
            0.10,
            "low",
        ),
    ],
)
async def test_legacy_contract_backfill_preserves_existing_sampling_controls(
    monkeypatch,
    legacy_contract,
    expected_temperature,
    expected_reasoning_effort,
):
    db = _fresh_db()
    await _seed(db)
    await db["exampen_exams"].update_one(
        {"exam_id": "EXAM-DOC-1"},
        {"$set": {"pcr_grading_contract": legacy_contract}},
    )

    result, gate = await _grade(
        monkeypatch,
        db,
        {
            "document_review": _document_review(),
            "questions": [_attempted_diagram(), _attempted_explanation()],
        },
    )

    assert result.status == "completed"
    assert gate.calls[0]["temperature"] == expected_temperature
    assert gate.calls[0]["reasoning_effort"] == expected_reasoning_effort
    frozen_exam = await db["exampen_exams"].find_one({"exam_id": "EXAM-DOC-1"})
    frozen_contract = frozen_exam["pcr_grading_contract"]
    assert frozen_contract["temperature"] == expected_temperature
    assert frozen_contract["reasoning_effort"] == expected_reasoning_effort
    assert frozen_contract["locked_at"] == "legacy-lock"


@pytest.mark.asyncio
async def test_concurrent_identical_bytes_for_different_students_do_not_share_output(
    monkeypatch,
):
    db = _fresh_db()
    await _seed(db)
    await _seed_duplicate_submission(db)
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
    gate = _SlowFakeGate(
        {
            "document_review": _document_review(),
            "questions": [_attempted_diagram(), _attempted_explanation()],
        }
    )
    service = module.FullDocumentGradingService(
        db,
        gate,
        model_id="gpt-5.1-2025-11-13",
    )

    first, second = await asyncio.gather(
        service.grade_submission("SUB-DOC-1"),
        service.grade_submission("SUB-DOC-2"),
    )

    assert first.run_id != second.run_id
    assert len(gate.calls) == 2
    assert await db["evalpen_detected_responses"].count_documents({}) == 4
    assert await db["evalpen_evaluations"].count_documents({}) == 4


@pytest.mark.asyncio
async def test_same_submission_revision_retry_is_idempotent_after_model_freeze(
    monkeypatch,
):
    db = _fresh_db()
    await _seed(db)
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
    gate = _FakeGate(
        {
            "document_review": _document_review(),
            "questions": [_attempted_diagram(), _attempted_explanation()],
        }
    )
    service = module.FullDocumentGradingService(
        db,
        gate,
        model_id="gpt-5.1",
    )

    first = await service.grade_submission("SUB-DOC-1")
    retry = await service.grade_submission("SUB-DOC-1")

    assert retry.run_id == first.run_id
    assert len(gate.calls) == 1
    assert await db["evalpen_document_grading_runs"].count_documents({}) == 1
    assert await db["evalpen_detected_responses"].count_documents({}) == 2


@pytest.mark.asyncio
async def test_explicit_reprocess_reuses_ledger_and_creates_auditable_rows(
    monkeypatch,
):
    db = _fresh_db()
    await _seed(db)
    await db["exampen_processing_jobs"].insert_one(
        {
            "job_id": "pcr-job-SUB-DOC-1",
            "submission_id": "SUB-DOC-1",
            "status": "completed",
            "reprocess_count": 0,
        }
    )
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
    gate = _FakeGate(
        {
            "document_review": _document_review(),
            "questions": [_attempted_diagram(), _attempted_explanation()],
        }
    )
    service = module.FullDocumentGradingService(
        db,
        gate,
        model_id="gpt-5.1-2025-11-13",
    )

    first = await service.grade_submission("SUB-DOC-1")
    first_rows = await db["evalpen_detected_responses"].find(
        {"submission_id": "SUB-DOC-1", "superseded_at": {"$exists": False}}
    ).to_list(length=10)
    await db["exampen_processing_jobs"].update_one(
        {"job_id": "pcr-job-SUB-DOC-1"},
        {"$set": {"reprocess_count": 1}},
    )

    second = await service.grade_submission("SUB-DOC-1")

    assert first.run_id == second.run_id
    assert len(gate.calls) == 1
    active_rows = await db["evalpen_detected_responses"].find(
        {"submission_id": "SUB-DOC-1", "superseded_at": {"$exists": False}}
    ).to_list(length=10)
    assert len(active_rows) == 2
    assert all(row["mapping_version_id"].endswith(":r1") for row in active_rows)
    assert {row["response_id"] for row in first_rows}.isdisjoint(
        {row["response_id"] for row in active_rows}
    )
    assert await db["evalpen_detected_responses"].count_documents(
        {"submission_id": "SUB-DOC-1"}
    ) == 4
    assert await db["evalpen_evaluations"].count_documents(
        {"student_id": "STU-1"}
    ) == 4
    submission = await db["evalpen_submissions"].find_one(
        {"submission_id": "SUB-DOC-1"}
    )
    assert submission["resumed_grading_run"] is True


@pytest.mark.asyncio
async def test_valid_alternative_method_is_accepted_and_audited(monkeypatch):
    db = _fresh_db()
    await _seed(db)
    alternative = _attempted_diagram()
    alternative["method_analysis"] = _method_analysis(
        classification="valid_alternative"
    )
    alternative["method_analysis"]["detected_method"] = (
        "Student used an equivalent circuit representation with the same connectivity."
    )

    result, _gate = await _grade(
        monkeypatch,
        db,
        {
            "document_review": _document_review(),
            "questions": [alternative, _attempted_explanation()],
        },
    )

    assert result.review_state == "ready"
    evaluation = await db["evalpen_evaluations"].find_one(
        {"question_id": "EXAM-DOC-1::Q1"}
    )
    assert evaluation["total_score"] == 2
    assert evaluation["method_policy"]["mode"] == "any_valid_method"
    assert evaluation["method_analysis"]["method_classification"] == "alternative_method"


@pytest.mark.asyncio
async def test_uncertain_legacy_alternative_method_preserves_criterion_score(
    monkeypatch,
):
    db = _fresh_db()
    await _seed(db)
    alternative = _attempted_diagram()
    alternative["method_analysis"] = _method_analysis(
        classification="valid_alternative",
        validity="unresolved",
        requirement_satisfied=False,
    )
    alternative["method_analysis"]["confidence"] = 0.78

    result, _gate = await _grade(
        monkeypatch,
        db,
        {
            "document_review": _document_review(),
            "questions": [alternative, _attempted_explanation()],
        },
    )

    assert result.review_state == "needs_review"
    assert result.blocked_count == 0
    response = await db["evalpen_detected_responses"].find_one(
        {"question_id": "EXAM-DOC-1::Q1", "superseded_at": {"$exists": False}}
    )
    evaluation = await db["evalpen_evaluations"].find_one(
        {"question_id": "EXAM-DOC-1::Q1"}
    )
    assert response["answer_state"] == "detected"
    assert response["eval_status"] == "manual_review"
    assert evaluation["total_score"] == 2
    assert evaluation["method_analysis"]["method_classification"] == "alternative_method"
    assert evaluation["method_analysis"]["method_validity"] == "unresolved"


@pytest.mark.asyncio
async def test_full_marks_fail_closed_when_named_method_was_not_used(monkeypatch):
    db = _fresh_db()
    await _seed(db)
    await db["evalpen_questions"].update_one(
        {"question_id": "EXAM-DOC-1::Q1"},
        {
            "$set": {
                "method_policy": {
                    "mode": "specified_method_required",
                    "required_method": "Kirchhoff's loop method",
                    "allow_error_carried_forward": True,
                }
            }
        },
    )
    wrong_method = _attempted_diagram()
    wrong_method["method_analysis"] = _method_analysis(
        classification="valid_alternative",
        requirement_satisfied=False,
    )

    result, _gate = await _grade(
        monkeypatch,
        db,
        {
            "document_review": _document_review(),
            "questions": [wrong_method, _attempted_explanation()],
        },
    )

    assert result.review_state == "blocked"
    response = await db["evalpen_detected_responses"].find_one(
        {"question_id": "EXAM-DOC-1::Q1", "superseded_at": {"$exists": False}}
    )
    assert response["answer_state"] == "unresolved"
    assert "required method" in response["manual_review_reason"].lower()


@pytest.mark.asyncio
async def test_follow_through_credit_is_linked_to_the_awarded_criterion(monkeypatch):
    db = _fresh_db()
    await _seed(db)
    explanation = _attempted_explanation()
    explanation["method_analysis"]["error_carried_forward"] = "applied"
    explanation["method_analysis"]["error_carried_forward_reason"] = (
        "The student used their earlier value consistently in the explanation."
    )
    explanation["criterion_marks"][0]["credit_basis"] = "error_carried_forward"

    result, _gate = await _grade(
        monkeypatch,
        db,
        {
            "document_review": _document_review(),
            "questions": [_attempted_diagram(), explanation],
        },
    )

    assert result.review_state == "ready"
    evaluation = await db["evalpen_evaluations"].find_one(
        {"question_id": "EXAM-DOC-1::Q2"}
    )
    assert evaluation["criterion_marks"][0]["credit_basis"] == "error_carried_forward"
    assert evaluation["method_analysis"]["error_carried_forward"] == "applied"
