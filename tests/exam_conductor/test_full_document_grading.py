from __future__ import annotations

import asyncio
import io
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


class _SequenceGate:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = list(payloads)
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
        payload = self.payloads.pop(0)
        return SimpleNamespace(
            content=json.dumps(payload),
            usage=SimpleNamespace(
                model="gpt-5.1-2025-11-13",
                caller=caller_id,
                input_tokens=4_000,
                output_tokens=800,
                cache_read_tokens=2_000,
                total_tokens=4_800,
                estimated_cost_usd=0.05,
            ),
        )


class _ScriptedGate:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
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
        payload = self.responses.pop(0)
        if isinstance(payload, BaseException):
            raise payload
        if hasattr(payload, "content"):
            if getattr(payload, "usage", None) is None:
                payload.usage = SimpleNamespace(
                    model="gpt-5.1-2025-11-13",
                    caller=caller_id,
                    input_tokens=3_500,
                    output_tokens=750,
                    cache_read_tokens=2_000,
                    total_tokens=4_250,
                    estimated_cost_usd=0.05,
                )
            return payload
        return SimpleNamespace(
            content=json.dumps(payload),
            usage=SimpleNamespace(
                model="gpt-5.1-2025-11-13",
                caller=caller_id,
                input_tokens=3_500,
                output_tokens=750,
                cache_read_tokens=2_000,
                total_tokens=4_250,
                estimated_cost_usd=0.05,
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


def _page_asset(module, page_number: int):
    from PIL import Image

    output = io.BytesIO()
    Image.new("RGB", (1200, 1600), "white").save(output, format="JPEG")
    value = output.getvalue()
    return module._StudentPageAsset(
        page_number=page_number,
        original_bytes=value,
        global_bytes=value,
        global_media_type="image/jpeg",
    )


def test_subjective_grade_shell_is_not_semantically_complete():
    module = _module()
    question = {
        "question_number": 1,
        "max_marks": 2,
        "marking_criteria": [
            {
                "criterion_id": "step-1",
                "description": "First valid step",
                "max_marks": 1,
                "acceptable_evidence": "Shows the first step.",
            },
            {
                "criterion_id": "step-2",
                "description": "Correct conclusion",
                "max_marks": 1,
                "acceptable_evidence": "States the correct conclusion.",
            },
        ],
    }

    defects = module._semantic_grade_defects(
        question,
        {
            "attempt_status": "attempted",
            "student_answer": "",
            "criterion_marks": [],
        },
    )

    assert "expected 2 criterion results, received 0" in defects
    assert "attempted answer has no readable student work" in defects
    assert module._semantic_grade_defects(
        question,
        {"attempt_status": "not_attempted", "criterion_marks": []},
    ) == []


def test_question_grading_schema_locks_every_question_and_criterion():
    from api.v1._exampen_imports import load_exampen

    graph = load_exampen("pcr.services.visual_evidence_graph")
    schema = graph.question_grading_schema(
        [
            {
                "question_number": 7,
                "max_marks": 2,
                "marking_criteria": [
                    {"criterion_id": "height", "max_marks": 1},
                    {"criterion_id": "range", "max_marks": 1},
                ],
            }
        ]
    )
    root = schema.get("schema", schema)
    questions = root["properties"]["questions"]
    assert questions["type"] == "object"
    assert questions["required"] == ["7"]
    criteria = questions["properties"]["7"]["properties"]["criterion_marks"]
    assert criteria["required"] == ["height", "range"]
    assert criteria["additionalProperties"] is False
    assert criteria["properties"]["height"]["properties"]["marks_awarded"][
        "maximum"
    ] == 1


def test_close_visual_readings_keep_marks_but_require_review():
    module = _module()
    question = {
        "question_id": "Q-AMBIGUOUS",
        "question_number": 1,
        "question_text": "Evaluate the expression.",
        "max_marks": 1,
        "marking_criteria": [
            {
                "criterion_id": "answer",
                "description": "Correct interpretation and value",
                "max_marks": 1,
                "acceptable_evidence": "The superscript is read and evaluated correctly.",
            }
        ],
    }
    item = {
        "question_number": 1,
        "attempt_status": "attempted",
        "confidence": 0.96,
        "content_type": "MIXED",
        "evidence_regions": [
            {
                "region_id": "q1-math",
                "page_number": 1,
                "x_start": 100,
                "y_start": 100,
                "x_end": 700,
                "y_end": 400,
                "evidence_kind": "mathematics",
                "continuation_group": "",
                "evidence": "A handwritten exponent is visible.",
                "mapping_confidence": 0.96,
            }
        ],
        "student_answer": "2 raised to 3",
        "interpretation_hypotheses": [
            {
                "interpretation_id": "superscript",
                "value": "2 raised to 3",
                "confidence": 0.84,
                "evidence_region_ids": ["q1-math"],
                "ambiguity_notes": "The 3 is small and above the baseline.",
            },
            {
                "interpretation_id": "adjacent",
                "value": "23",
                "confidence": 0.80,
                "evidence_region_ids": ["q1-math"],
                "ambiguity_notes": "The camera angle makes the baseline uncertain.",
            },
        ],
        "visual_semantics": {
            "summary": "A base and a small raised numeral are visible.",
            "elements": [
                {
                    "element_id": "expression",
                    "element_type": "mathematics",
                    "label": "2^3",
                    "region_id": "q1-math",
                    "attributes": "possible superscript",
                    "confidence": 0.84,
                }
            ],
            "relationships": [],
            "confidence": 0.90,
        },
        "method_analysis": _method_analysis(),
        "criterion_marks": [
            {
                "criterion_id": "answer",
                "decision": "met",
                "confidence": 0.90,
                "marks_awarded": 1,
                "rationale": "The raised 3 is interpreted as an exponent.",
                "evidence": "The exponent is visible in q1-math.",
                "evidence_region_ids": ["q1-math"],
                "missing_evidence": "",
                "credit_basis": "direct_evidence",
            }
        ],
        "total_score": 1,
        "overall_feedback": "Correct.",
        "needs_review": False,
        "review_reason": "",
    }

    grade = module._validate_question_grade(
        item,
        question=question,
        question_number=1,
        page_count=1,
        coverage_complete=True,
        coverage_confidence=0.98,
    )

    assert grade.total_score == 1
    assert grade.manual_review_required is True
    assert "plausible visual readings" in grade.review_reason


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
async def test_evidence_graph_maps_distant_continuation_and_side_by_side_diagram(
    monkeypatch,
):
    db = _fresh_db()
    await _seed(db)
    await db["exampen_paper_versions"].update_one(
        {"paper_version_id": "paper-version-1"},
        {
            "$set": {
                "paper_context": {
                    "version": "canonical-full-document-visual-v2",
                    "mode": "full_document_visual",
                    "ready": True,
                },
                "paper_assets": {
                    "question_paper": {
                        "asset_id": "asset-question",
                        "storage_uri": "s3://private/exampen/paper-assets/test/question.pdf",
                        "sha256": "test-question-hash",
                        "filename": "question.pdf",
                    },
                    "teacher_solution": {
                        "asset_id": "asset-solution",
                        "storage_uri": "s3://private/exampen/paper-assets/test/solution.pdf",
                        "sha256": "test-solution-hash",
                        "filename": "solution.pdf",
                    },
                },
            }
        },
    )
    await db["evalpen_answer_pages"].insert_one(
        {
            "page_id": "PAGE-2",
            "submission_id": "SUB-DOC-1",
            "page_number": 2,
            "raw_image_ref": "private-page-2.jpg",
            "content_hash": "page-hash-2",
        }
    )
    module = _module()
    monkeypatch.setattr(
        module,
        "_read_canonical_file",
        lambda *args, **kwargs: _async_value(b"%PDF-1.4 canonical"),
    )
    import services.exampen_paper_service as paper_service

    monkeypatch.setattr(
        paper_service,
        "load_canonical_paper_asset",
        lambda *args, **kwargs: _async_value(b"%PDF-1.4 canonical"),
    )
    assets = [_page_asset(module, 1), _page_asset(module, 2)]
    monkeypatch.setattr(
        module,
        "_student_page_assets",
        lambda pages: _async_value((assets, sum(len(a.global_bytes) for a in assets))),
    )
    mapping_payload = {
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v1",
        "document_review": _document_review(),
        "questions": [
            {
                "question_number": 1,
                "attempt_status": "attempted",
                "confidence": 0.96,
                "content_type": "DIAGRAM_HEAVY",
                "evidence_regions": [
                    {
                        "region_id": "q1-diagram",
                        "page_number": 1,
                        "x_start": 20,
                        "y_start": 100,
                        "x_end": 470,
                        "y_end": 520,
                        "evidence_kind": "diagram",
                        "continuation_group": "q1",
                        "evidence": "Circuit diagram and labels",
                        "mapping_confidence": 0.97,
                    },
                    {
                        "region_id": "q1-continuation",
                        "page_number": 2,
                        "x_start": 30,
                        "y_start": 120,
                        "x_end": 970,
                        "y_end": 330,
                        "evidence_kind": "handwriting",
                        "continuation_group": "q1",
                        "evidence": "Continuation explaining the circuit",
                        "mapping_confidence": 0.94,
                    },
                ],
                "mapping_reason": "Diagram semantics and continuation match Q1.",
                "needs_review": False,
                "review_reason": "",
            },
            {
                "question_number": 2,
                "attempt_status": "attempted",
                "confidence": 0.95,
                "content_type": "TEXT_ONLY",
                "evidence_regions": [
                    {
                        "region_id": "q2-explanation",
                        "page_number": 1,
                        "x_start": 530,
                        "y_start": 100,
                        "x_end": 980,
                        "y_end": 520,
                        "evidence_kind": "handwriting",
                        "continuation_group": "",
                        "evidence": "Written explanation beside the diagram",
                        "mapping_confidence": 0.95,
                    }
                ],
                "mapping_reason": "The explanation directly answers Q2.",
                "needs_review": False,
                "review_reason": "",
            },
        ],
        "unassigned_regions": [],
    }
    grading_payload = {
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v1",
        "questions": [
            {
                "question_number": 1,
                "confidence": 0.96,
                "student_answer": "A closed labelled circuit is shown.",
                "interpretation_hypotheses": [
                    {
                        "interpretation_id": "q1-primary",
                        "value": "Closed battery-switch-lamp circuit",
                        "confidence": 0.96,
                        "evidence_region_ids": [
                            "q1-diagram",
                            "q1-continuation",
                        ],
                        "ambiguity_notes": "",
                    }
                ],
                "visual_semantics": {
                    "summary": "Battery, switch and lamp form a closed circuit.",
                    "elements": [
                        {
                            "element_id": "battery",
                            "element_type": "circuit_component",
                            "label": "battery",
                            "region_id": "q1-diagram",
                            "attributes": "connected",
                            "confidence": 0.96,
                        }
                    ],
                    "relationships": [],
                    "confidence": 0.95,
                },
                "method_analysis": _method_analysis(),
                "criterion_marks": [
                    {
                        **_attempted_diagram()["criterion_marks"][0],
                        "evidence_region_ids": [
                            "q1-diagram",
                            "q1-continuation",
                        ],
                    }
                ],
                "total_score": 2,
                "overall_feedback": "Correct diagram.",
                "needs_review": False,
                "review_reason": "",
            },
            {
                "question_number": 2,
                "confidence": 0.95,
                "student_answer": "The observation follows because the circuit is complete.",
                "interpretation_hypotheses": [
                    {
                        "interpretation_id": "q2-primary",
                        "value": "Complete circuit explains the observation",
                        "confidence": 0.95,
                        "evidence_region_ids": ["q2-explanation"],
                        "ambiguity_notes": "",
                    }
                ],
                "visual_semantics": {
                    "summary": "Readable written explanation.",
                    "elements": [],
                    "relationships": [],
                    "confidence": 0.95,
                },
                "method_analysis": _method_analysis(),
                "criterion_marks": [
                    {
                        **_attempted_explanation()["criterion_marks"][0],
                        "evidence_region_ids": ["q2-explanation"],
                    }
                ],
                "total_score": 2,
                "overall_feedback": "Correct explanation.",
                "needs_review": False,
                "review_reason": "",
            },
        ],
    }
    gate = _SequenceGate([mapping_payload, grading_payload])
    service = module.FullDocumentGradingService(
        db,
        gate,
        model_id="gpt-5.1-2025-11-13",
    )

    result = await service.grade_submission("SUB-DOC-1")

    assert result.status == "completed"
    assert result.review_state == "ready"
    assert result.evaluated_count == 2
    assert len(gate.calls) == 2
    assert gate.calls[0]["metadata"]["pcr_stage"] == "full_document_evidence_mapping"
    assert gate.calls[1]["metadata"]["pcr_stage"] == "question_visual_grading"
    q1 = await db["evalpen_detected_responses"].find_one(
        {"question_id": "EXAM-DOC-1::Q1", "superseded_at": {"$exists": False}}
    )
    q2 = await db["evalpen_detected_responses"].find_one(
        {"question_id": "EXAM-DOC-1::Q2", "superseded_at": {"$exists": False}}
    )
    assert q1["evidence_version"] == 3
    assert len(q1["source_pages"]) == 2
    assert q1["source_pages"][0]["region_id"] == "q1-diagram"
    assert q2["source_pages"][0]["x_start"] > 100
    assert q1["visual_evidence"]["visual_semantics"]["elements"][0]["label"] == "battery"
    assert q1["semantic_evidence_signature"]
    run = await db["evalpen_document_grading_runs"].find_one(
        {"run_id": result.run_id}
    )
    assert run["prompt_version"] == "pcr-full-document-visual-v5"
    assert run["evidence_graph_mapping"]["questions"][0]["evidence_regions"]
    frozen_exam = await db["exampen_exams"].find_one({"exam_id": "EXAM-DOC-1"})
    assert frozen_exam["pcr_grading_contract"]["prompt_version"] == (
        "pcr-full-document-visual-v5"
    )


@pytest.mark.asyncio
async def test_evidence_graph_question_grading_splits_batch_on_output_token_exhaustion(
    monkeypatch,
):
    db = _fresh_db()
    await _seed(db)
    await db["exampen_paper_versions"].update_one(
        {"paper_version_id": "paper-version-1"},
        {
            "$set": {
                "paper_context": {
                    "version": "canonical-full-document-visual-v2",
                    "mode": "full_document_visual",
                    "ready": True,
                },
                "paper_assets": {
                    "question_paper": {
                        "asset_id": "asset-question",
                        "storage_uri": "s3://private/exampen/paper-assets/test/question.pdf",
                        "sha256": "test-question-hash",
                        "filename": "question.pdf",
                    },
                    "teacher_solution": {
                        "asset_id": "asset-solution",
                        "storage_uri": "s3://private/exampen/paper-assets/test/solution.pdf",
                        "sha256": "test-solution-hash",
                        "filename": "solution.pdf",
                    },
                },
            },
        },
    )
    module = _module()
    monkeypatch.setattr(
        module,
        "_read_canonical_file",
        lambda *args, **kwargs: _async_value(b"%PDF-1.4 canonical"),
    )
    import services.exampen_paper_service as paper_service

    monkeypatch.setattr(
        paper_service,
        "load_canonical_paper_asset",
        lambda *_args, **_kwargs: _async_value(b"%PDF-1.4 canonical"),
    )
    assets = [_page_asset(module, 1)]
    monkeypatch.setattr(
        module,
        "_student_page_assets",
        lambda pages: _async_value((assets, sum(len(a.global_bytes) for a in assets))),
    )
    monkeypatch.setenv("PCR_VISUAL_QUESTIONS_PER_BATCH", "2")

    mapping_payload = {
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v1",
        "document_review": _document_review(),
        "questions": [
            {
                "question_number": 1,
                "attempt_status": "attempted",
                "confidence": 0.96,
                "content_type": "DIAGRAM_HEAVY",
                "evidence_regions": [
                    {
                        "region_id": "q1-diagram",
                        "page_number": 1,
                        "x_start": 20,
                        "y_start": 100,
                        "x_end": 470,
                        "y_end": 520,
                        "evidence_kind": "diagram",
                        "continuation_group": "q1",
                        "evidence": "Circuit diagram and labels",
                        "mapping_confidence": 0.97,
                    },
                ],
                "mapping_reason": "Diagram semantics and continuation match Q1.",
                "needs_review": False,
                "review_reason": "",
            },
            {
                "question_number": 2,
                "attempt_status": "attempted",
                "confidence": 0.95,
                "content_type": "TEXT_ONLY",
                "evidence_regions": [
                    {
                        "region_id": "q2-explanation",
                        "page_number": 1,
                        "x_start": 530,
                        "y_start": 100,
                        "x_end": 980,
                        "y_end": 520,
                        "evidence_kind": "handwriting",
                        "continuation_group": "",
                        "evidence": "Written explanation beside the diagram",
                        "mapping_confidence": 0.95,
                    },
                ],
                "mapping_reason": "The explanation directly answers Q2.",
                "needs_review": False,
                "review_reason": "",
            },
        ],
        "unassigned_regions": [],
    }
    incomplete_payload = SimpleNamespace(
        content='{"questions": [',
        completion_status="incomplete",
        incomplete_reason="max_output_tokens",
    )
    grade_payload_one = {
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v1",
        "questions": [
            {
                "question_number": 1,
                "confidence": 0.96,
                "student_answer": "A closed labelled circuit is shown.",
                "interpretation_hypotheses": [
                    {
                        "interpretation_id": "q1-primary",
                        "value": "Closed battery-switch-lamp circuit",
                        "confidence": 0.96,
                        "evidence_region_ids": ["q1-diagram"],
                        "ambiguity_notes": "",
                    }
                ],
                "visual_semantics": {
                    "summary": "Battery, switch and lamp form a closed circuit.",
                    "elements": [
                        {
                            "element_id": "battery",
                            "element_type": "circuit_component",
                            "label": "battery",
                            "region_id": "q1-diagram",
                            "attributes": "connected",
                            "confidence": 0.96,
                        }
                    ],
                    "relationships": [],
                    "confidence": 0.95,
                },
                "method_analysis": _method_analysis(),
                "criterion_marks": [
                    {
                        **_attempted_diagram()["criterion_marks"][0],
                        "evidence_region_ids": ["q1-diagram"],
                    }
                ],
                "total_score": 2,
                "overall_feedback": "Correct diagram.",
                "needs_review": False,
                "review_reason": "",
            }
        ],
    }
    grade_payload_two = {
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v1",
        "questions": [
            {
                "question_number": 2,
                "confidence": 0.95,
                "student_answer": "The observation follows because the circuit is complete.",
                "interpretation_hypotheses": [
                    {
                        "interpretation_id": "q2-primary",
                        "value": "Complete circuit explains the observation",
                        "confidence": 0.95,
                        "evidence_region_ids": ["q2-explanation"],
                        "ambiguity_notes": "",
                    }
                ],
                "visual_semantics": {
                    "summary": "Readable written explanation.",
                    "elements": [],
                    "relationships": [],
                    "confidence": 0.95,
                },
                "method_analysis": _method_analysis(),
                "criterion_marks": [
                    {
                        **_attempted_explanation()["criterion_marks"][0],
                        "evidence_region_ids": ["q2-explanation"],
                    }
                ],
                "total_score": 2,
                "overall_feedback": "Correct explanation.",
                "needs_review": False,
                "review_reason": "",
            }
        ],
    }
    gate = _ScriptedGate(
        [
            mapping_payload,
            incomplete_payload,
            grade_payload_one,
            grade_payload_two,
        ]
    )
    service = module.FullDocumentGradingService(
        db,
        gate,
        model_id="gpt-5.1-2025-11-13",
    )

    result = await service.grade_submission("SUB-DOC-1")

    assert result.status == "completed"
    assert result.evaluated_count == 2
    assert len(gate.calls) == 4
    assert gate.calls[1]["metadata"]["question_numbers"] == [1, 2]
    assert gate.calls[2]["metadata"]["question_numbers"] == [1]
    assert gate.calls[3]["metadata"]["question_numbers"] == [2]


@pytest.mark.asyncio
async def test_evidence_graph_question_grading_contract_mismatch_splits_batch(
    monkeypatch,
):
    db = _fresh_db()
    await _seed(db)
    await db["exampen_paper_versions"].update_one(
        {"paper_version_id": "paper-version-1"},
        {
            "$set": {
                "paper_context": {
                    "version": "canonical-full-document-visual-v2",
                    "mode": "full_document_visual",
                    "ready": True,
                },
                "paper_assets": {
                    "question_paper": {
                        "asset_id": "asset-question",
                        "storage_uri": "s3://private/exampen/paper-assets/test/question.pdf",
                        "sha256": "test-question-hash",
                        "filename": "question.pdf",
                    },
                    "teacher_solution": {
                        "asset_id": "asset-solution",
                        "storage_uri": "s3://private/exampen/paper-assets/test/solution.pdf",
                        "sha256": "test-solution-hash",
                        "filename": "solution.pdf",
                    },
                },
            },
        },
    )
    module = _module()
    monkeypatch.setattr(
        module,
        "_read_canonical_file",
        lambda *args, **kwargs: _async_value(b"%PDF-1.4 canonical"),
    )
    import services.exampen_paper_service as paper_service

    monkeypatch.setattr(
        paper_service,
        "load_canonical_paper_asset",
        lambda *_args, **_kwargs: _async_value(b"%PDF-1.4 canonical"),
    )
    assets = [_page_asset(module, 1)]
    monkeypatch.setattr(
        module,
        "_student_page_assets",
        lambda pages: _async_value((assets, sum(len(a.global_bytes) for a in assets))),
    )
    monkeypatch.setenv("PCR_VISUAL_QUESTIONS_PER_BATCH", "2")

    mapping_payload = {
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v1",
        "document_review": _document_review(),
        "questions": [
            {
                "question_number": 1,
                "attempt_status": "attempted",
                "confidence": 0.96,
                "content_type": "DIAGRAM_HEAVY",
                "evidence_regions": [
                    {
                        "region_id": "q1-diagram",
                        "page_number": 1,
                        "x_start": 20,
                        "y_start": 100,
                        "x_end": 470,
                        "y_end": 520,
                        "evidence_kind": "diagram",
                        "continuation_group": "q1",
                        "evidence": "Circuit diagram and labels",
                        "mapping_confidence": 0.97,
                    },
                ],
                "mapping_reason": "Diagram semantics and continuation match Q1.",
                "needs_review": False,
                "review_reason": "",
            },
            {
                "question_number": 2,
                "attempt_status": "attempted",
                "confidence": 0.95,
                "content_type": "TEXT_ONLY",
                "evidence_regions": [
                    {
                        "region_id": "q2-explanation",
                        "page_number": 1,
                        "x_start": 530,
                        "y_start": 100,
                        "x_end": 980,
                        "y_end": 520,
                        "evidence_kind": "handwriting",
                        "continuation_group": "",
                        "evidence": "Written explanation beside the diagram",
                        "mapping_confidence": 0.95,
                    },
                ],
                "mapping_reason": "The explanation directly answers Q2.",
                "needs_review": False,
                "review_reason": "",
            },
        ],
        "unassigned_regions": [],
    }
    mismatched_batch_payload = {
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v1",
        "questions": [
            {
                "question_number": 2,
                "confidence": 0.95,
                "student_answer": "Mismatched payload that belongs to Q2.",
                "interpretation_hypotheses": [
                    {
                        "interpretation_id": "q2-primary",
                        "value": "Mismatched question output.",
                        "confidence": 0.95,
                        "evidence_region_ids": ["q2-explanation"],
                        "ambiguity_notes": "",
                    }
                ],
                "visual_semantics": {
                    "summary": "A short answer about Q2.",
                    "elements": [],
                    "relationships": [],
                    "confidence": 0.95,
                },
                "method_analysis": _method_analysis(),
                "criterion_marks": [
                    {
                        **_attempted_explanation()["criterion_marks"][0],
                        "evidence_region_ids": ["q2-explanation"],
                    }
                ],
                "total_score": 2,
                "overall_feedback": "Correct explanation.",
                "needs_review": False,
                "review_reason": "",
            }
        ],
    }
    q2_single_payload = {
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v1",
        "questions": [
            {
                "question_number": 2,
                "confidence": 0.95,
                "student_answer": "The observation follows because the circuit is complete.",
                "interpretation_hypotheses": [
                    {
                        "interpretation_id": "q2-primary",
                        "value": "Complete circuit explains the observation",
                        "confidence": 0.95,
                        "evidence_region_ids": ["q2-explanation"],
                        "ambiguity_notes": "",
                    }
                ],
                "visual_semantics": {
                    "summary": "Readable written explanation.",
                    "elements": [],
                    "relationships": [],
                    "confidence": 0.95,
                },
                "method_analysis": _method_analysis(),
                "criterion_marks": [
                    {
                        **_attempted_explanation()["criterion_marks"][0],
                        "evidence_region_ids": ["q2-explanation"],
                    }
                ],
                "total_score": 2,
                "overall_feedback": "Correct explanation.",
                "needs_review": False,
                "review_reason": "",
            }
        ],
    }

    gate = _ScriptedGate(
        [
            mapping_payload,
            mismatched_batch_payload,
            mismatched_batch_payload,
            q2_single_payload,
        ]
    )
    service = module.FullDocumentGradingService(
        db,
        gate,
        model_id="gpt-5.1-2025-11-13",
    )

    result = await service.grade_submission("SUB-DOC-1")

    assert result.status == "completed"
    assert result.review_state == "blocked"
    assert result.evaluated_count == 1
    assert result.blocked_count == 1
    assert len(gate.calls) == 4
    assert gate.calls[1]["metadata"]["question_numbers"] == [1, 2]
    assert gate.calls[2]["metadata"]["question_numbers"] == [1]
    assert gate.calls[3]["metadata"]["question_numbers"] == [2]

    q1 = await db["evalpen_detected_responses"].find_one(
        {"question_id": "EXAM-DOC-1::Q1", "superseded_at": {"$exists": False}}
    )
    q2 = await db["evalpen_detected_responses"].find_one(
        {"question_id": "EXAM-DOC-1::Q2", "superseded_at": {"$exists": False}}
    )
    assert q1["answer_state"] == "unresolved"
    assert q2["answer_state"] == "detected"


@pytest.mark.asyncio
async def test_evidence_graph_question_grading_marks_question_unresolved_when_budget_caps(
    monkeypatch,
):
    db = _fresh_db()
    await _seed(db)
    await db["exampen_paper_versions"].update_one(
        {"paper_version_id": "paper-version-1"},
        {
            "$set": {
                "paper_context": {
                    "version": "canonical-full-document-visual-v2",
                    "mode": "full_document_visual",
                    "ready": True,
                },
                "paper_assets": {
                    "question_paper": {
                        "asset_id": "asset-question",
                        "storage_uri": "s3://private/exampen/paper-assets/test/question.pdf",
                        "sha256": "test-question-hash",
                        "filename": "question.pdf",
                    },
                    "teacher_solution": {
                        "asset_id": "asset-solution",
                        "storage_uri": "s3://private/exampen/paper-assets/test/solution.pdf",
                        "sha256": "test-solution-hash",
                        "filename": "solution.pdf",
                    },
                },
            },
        },
    )
    module = _module()
    monkeypatch.setattr(
        module,
        "_read_canonical_file",
        lambda *args, **kwargs: _async_value(b"%PDF-1.4 canonical"),
    )
    import services.exampen_paper_service as paper_service

    monkeypatch.setattr(
        paper_service,
        "load_canonical_paper_asset",
        lambda *_args, **_kwargs: _async_value(b"%PDF-1.4 canonical"),
    )
    assets = [_page_asset(module, 1)]
    monkeypatch.setattr(
        module,
        "_student_page_assets",
        lambda pages: _async_value((assets, sum(len(a.global_bytes) for a in assets))),
    )
    monkeypatch.setenv("PCR_VISUAL_QUESTIONS_PER_BATCH", "1")

    mapping_payload = {
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v1",
        "document_review": _document_review(),
        "questions": [
            {
                "question_number": 1,
                "attempt_status": "attempted",
                "confidence": 0.96,
                "content_type": "DIAGRAM_HEAVY",
                "evidence_regions": [
                    {
                        "region_id": "q1-diagram",
                        "page_number": 1,
                        "x_start": 20,
                        "y_start": 100,
                        "x_end": 470,
                        "y_end": 520,
                        "evidence_kind": "diagram",
                        "continuation_group": "q1",
                        "evidence": "Circuit diagram and labels",
                        "mapping_confidence": 0.97,
                    },
                ],
                "mapping_reason": "Diagram semantics and continuation match Q1.",
                "needs_review": False,
                "review_reason": "",
            },
            {
                "question_number": 2,
                "attempt_status": "attempted",
                "confidence": 0.95,
                "content_type": "TEXT_ONLY",
                "evidence_regions": [
                    {
                        "region_id": "q2-explanation",
                        "page_number": 1,
                        "x_start": 530,
                        "y_start": 100,
                        "x_end": 980,
                        "y_end": 520,
                        "evidence_kind": "handwriting",
                        "continuation_group": "",
                        "evidence": "Written explanation beside the diagram",
                        "mapping_confidence": 0.95,
                    },
                ],
                "mapping_reason": "The explanation directly answers Q2.",
                "needs_review": False,
                "review_reason": "",
            },
        ],
        "unassigned_regions": [],
    }
    incomplete_payload = SimpleNamespace(
        content='{"questions": [',
        completion_status="incomplete",
        incomplete_reason="max_output_tokens",
    )
    grade_payload_two = {
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v1",
        "questions": [
            {
                "question_number": 2,
                "confidence": 0.95,
                "student_answer": "The observation follows because the circuit is complete.",
                "interpretation_hypotheses": [
                    {
                        "interpretation_id": "q2-primary",
                        "value": "Complete circuit explains the observation",
                        "confidence": 0.95,
                        "evidence_region_ids": ["q2-explanation"],
                        "ambiguity_notes": "",
                    }
                ],
                "visual_semantics": {
                    "summary": "Readable written explanation.",
                    "elements": [],
                    "relationships": [],
                    "confidence": 0.95,
                },
                "method_analysis": _method_analysis(),
                "criterion_marks": [
                    {
                        **_attempted_explanation()["criterion_marks"][0],
                        "evidence_region_ids": ["q2-explanation"],
                    }
                ],
                "total_score": 2,
                "overall_feedback": "Correct explanation.",
                "needs_review": False,
                "review_reason": "",
            }
        ],
    }
    gate = _ScriptedGate(
        [
            mapping_payload,
            incomplete_payload,
            incomplete_payload,
            grade_payload_two,
            grade_payload_two,
        ]
    )
    service = module.FullDocumentGradingService(
        db,
        gate,
        model_id="gpt-5.1-2025-11-13",
    )

    result = await service.grade_submission("SUB-DOC-1")

    assert result.status == "completed"
    assert result.evaluated_count == 1
    assert result.blocked_count == 1
    assert result.review_state == "blocked"
    assert len(gate.calls) == 5
    assert gate.calls[1]["metadata"]["question_numbers"] == [1]
    assert gate.calls[2]["metadata"]["question_numbers"] == [1]
    assert gate.calls[3]["metadata"]["question_numbers"] == [1]
    assert gate.calls[4]["metadata"]["question_numbers"] == [2]

    q1 = await db["evalpen_detected_responses"].find_one(
        {"question_id": "EXAM-DOC-1::Q1", "superseded_at": {"$exists": False}}
    )
    q2 = await db["evalpen_detected_responses"].find_one(
        {"question_id": "EXAM-DOC-1::Q2", "superseded_at": {"$exists": False}}
    )
    assert q1["answer_state"] == "unresolved"
    assert q2["answer_state"] == "detected"
    assert q1["manual_review_required"] is True


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
async def test_canonical_visual_exam_never_silently_falls_back_when_asset_is_missing(
    monkeypatch,
):
    db = _fresh_db()
    await _seed(db)
    await db["exampen_paper_versions"].update_one(
        {"paper_version_id": "paper-version-1"},
        {
            "$set": {
                "paper_context": {
                    "version": "canonical-full-document-visual-v1",
                    "mode": "full_document_visual",
                    "ready": True,
                }
            }
        },
    )
    await db["documents"].delete_many({"document_id": "paper-1"})
    monkeypatch.setenv("PCR_FULL_DOCUMENT_GRADING_ENABLED", "true")
    monkeypatch.setenv("AI_PROVIDER", "openai")
    module = _module()
    from services.exampen_paper_service import CanonicalPaperAssetError

    service = module.FullDocumentGradingService(
        db,
        _FakeGate({}),
        model_id="gpt-5.1-2025-11-13",
    )

    with pytest.raises(
        CanonicalPaperAssetError,
        match="immutable paper asset manifest is unavailable",
    ) as error:
        await service.grade_submission("SUB-DOC-1")

    assert error.value.retryable is False
    assert await db["evalpen_document_grading_runs"].count_documents({}) == 0
    assert await db["evalpen_detected_responses"].count_documents({}) == 0


def test_incomplete_strict_response_is_terminal_and_keeps_a_debuggable_record():
    module = _module()
    response = SimpleNamespace(
        completion_status="incomplete",
        incomplete_reason="max_output_tokens",
    )

    with pytest.raises(module.StructuredOutputContractError) as error:
        module._require_structured_payload(
            response,
            raw='{"questions": [',
            stage="Question visual grader",
    )

    assert error.value.retryable is False
    failure = error.value.structured_output_failure
    assert failure["stage"] == "Question visual grader"
    assert failure["completion_status"] == "incomplete"
    assert failure["incomplete_reason"] == "max_output_tokens"
    assert failure["raw_response"] == '{"questions": ['
    assert failure["recorded_at"]


def test_visual_grading_reserves_completion_capacity_for_strict_rubric_json(monkeypatch):
    module = _module()
    monkeypatch.delenv("PCR_VISUAL_MAPPING_OUTPUT_TOKENS_PER_QUESTION", raising=False)
    monkeypatch.delenv("PCR_VISUAL_GRADING_OUTPUT_TOKENS_PER_QUESTION", raising=False)

    assert module._mapping_output_token_budget(11) == 13_200
    assert module._question_output_token_budget(5) == 12_000


@pytest.mark.asyncio
async def test_incomplete_v5_response_is_audited_and_not_retried_as_transient(
    monkeypatch,
):
    db = _fresh_db()
    await _seed(db)
    await db["exampen_paper_versions"].update_one(
        {"paper_version_id": "paper-version-1"},
        {
            "$set": {
                "paper_context": {
                    "version": "canonical-full-document-visual-v2",
                    "mode": "full_document_visual",
                    "ready": True,
                },
                "paper_assets": {
                    "question_paper": {
                        "storage_uri": "s3://private/exampen/paper-assets/test/paper.pdf",
                        "sha256": "paper-hash",
                    }
                },
            }
        },
    )
    import services.exampen_paper_service as paper_service

    monkeypatch.setattr(
        paper_service,
        "load_canonical_paper_asset",
        lambda *_args, **_kwargs: _async_value(b"%PDF-1.4 canonical"),
    )
    module = _module()
    asset = _page_asset(module, 1)
    monkeypatch.setattr(
        module,
        "_student_page_assets",
        lambda _pages: _async_value(([asset], len(asset.global_bytes))),
    )

    class _IncompleteGate:
        async def call(self, *_args, **_kwargs):
            return SimpleNamespace(
                content='{"questions": [',
                completion_status="incomplete",
                incomplete_reason="max_output_tokens",
            )

    service = module.FullDocumentGradingService(
        db,
        _IncompleteGate(),
        model_id="gpt-5.1-2025-11-13",
    )
    with pytest.raises(module.StructuredOutputContractError):
        await service.grade_submission("SUB-DOC-1")

    run = await db["evalpen_document_grading_runs"].find_one(
        {"submission_id": "SUB-DOC-1"}
    )
    assert run["status"] == "failed"
    failure = run["structured_output_failure"]
    assert failure["stage"] == "Evidence mapper"
    assert failure["completion_status"] == "incomplete"
    assert failure["incomplete_reason"] == "max_output_tokens"
    assert failure["raw_response"] == '{"questions": ['
    assert failure["recorded_at"]


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
async def test_criterion_decision_is_derived_from_the_locked_award(monkeypatch):
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

    assert result.review_state == "ready"
    q1 = await db["evalpen_detected_responses"].find_one(
        {"question_id": "EXAM-DOC-1::Q1", "superseded_at": {"$exists": False}}
    )
    assert q1["answer_state"] == "detected"
    assert await db["evalpen_evaluations"].count_documents(
        {"question_id": "EXAM-DOC-1::Q1"}
    ) == 1


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
async def test_document_note_does_not_override_typed_complete_coverage(
    monkeypatch,
):
    db = _fresh_db()
    await _seed(db)
    review = _document_review()
    review["warnings"] = [
        "Questions 1 and 2 were not attempted; all visible work is accounted for."
    ]
    result, _gate = await _grade(
        monkeypatch,
        db,
        {
            "document_review": review,
            "questions": [_attempted_diagram(), _attempted_explanation()],
        },
    )

    assert result.status == "completed"
    assert result.review_state == "ready"
    assert result.document_review_required is False
    assert result.warning_count == 0
    responses = await db["evalpen_detected_responses"].find(
        {"submission_id": "SUB-DOC-1"}
    ).to_list(length=10)
    assert len(responses) == 2
    assert all(response["manual_review_required"] is False for response in responses)
    assert all(response["eval_status"] == "evaluated" for response in responses)
    submission = await db["evalpen_submissions"].find_one(
        {"submission_id": "SUB-DOC-1"}
    )
    assert submission["document_review"]["required"] is False
    assert submission["document_review"]["warnings"] == review["warnings"]


@pytest.mark.asyncio
async def test_incomplete_typed_document_coverage_remains_a_submission_gate(
    monkeypatch,
):
    db = _fresh_db()
    await _seed(db)
    review = _document_review(complete=False, confidence=0.62)
    review["warnings"] = ["A page edge is cropped and some writing may be missing."]
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


@pytest.mark.asyncio
async def test_not_attempted_note_does_not_turn_proven_absence_into_unresolved(
    monkeypatch,
):
    db = _fresh_db()
    await _seed(db)
    review = _document_review(complete=True, confidence=0.93)
    review["warnings"] = [
        "Only Q1 has visible work; Q2 was checked across the full copy and not attempted."
    ]
    result, _gate = await _grade(
        monkeypatch,
        db,
        {
            "document_review": review,
            "questions": [
                _attempted_diagram(),
                {
                    "question_number": 2,
                    "attempt_status": "not_attempted",
                    "confidence": 0.93,
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

    assert result.review_state == "ready"
    assert result.evaluated_count == 2
    assert result.blocked_count == 0
    q2 = await db["evalpen_detected_responses"].find_one(
        {"question_id": "EXAM-DOC-1::Q2", "superseded_at": {"$exists": False}}
    )
    assert q2["answer_state"] == "not_attempted"
    assert q2["absence_proven"] is True


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
async def test_explicit_reprocess_creates_fresh_generation_and_auditable_rows(
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

    assert first.run_id != second.run_id
    assert len(gate.calls) == 2
    assert await db["evalpen_document_grading_runs"].count_documents({}) == 2
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
    assert submission["resumed_grading_run"] is False


@pytest.mark.asyncio
async def test_reprocess_does_not_collide_with_stale_previous_generation(
    monkeypatch,
):
    """Regression: the SGTB rev-1 request must not join the stale rev-0 run."""

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
    await db["evalpen_document_grading_runs"].update_one(
        {"run_id": first.run_id},
        {
            "$set": {
                "status": "generating",
                "generation_lease_expires_at": module.datetime.now(
                    module.timezone.utc
                )
                - module.timedelta(minutes=1),
            },
            "$unset": {
                "validated_payload": "",
                "raw_llm_response": "",
                "result": "",
            },
        },
    )
    await db["exampen_processing_jobs"].update_one(
        {"job_id": "pcr-job-SUB-DOC-1"},
        {"$set": {"reprocess_count": 1}},
    )

    second = await service.grade_submission("SUB-DOC-1")

    assert second.run_id != first.run_id
    assert len(gate.calls) == 2
    stale = await db["evalpen_document_grading_runs"].find_one(
        {"run_id": first.run_id}
    )
    current = await db["evalpen_document_grading_runs"].find_one(
        {"run_id": second.run_id}
    )
    assert stale["status"] == "generating"
    assert stale["generation_revision"] == 0
    assert current["status"] == "completed"
    assert current["generation_revision"] == 1


@pytest.mark.asyncio
async def test_concurrent_workers_share_one_generation_call(monkeypatch):
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
        service.grade_submission("SUB-DOC-1"),
    )

    assert first.run_id == second.run_id
    assert len(gate.calls) == 1
    assert await db["evalpen_document_grading_runs"].count_documents({}) == 1


@pytest.mark.asyncio
async def test_expired_same_generation_run_is_atomically_reclaimed():
    db = _fresh_db()
    module = _module()
    input_fingerprint = "a" * 64
    generation_fingerprint = module._generation_fingerprint(
        submission_id="SUB-DOC-1",
        input_fingerprint=input_fingerprint,
        generation_revision=2,
    )
    run_id = f"DOCGR-{generation_fingerprint[:24]}"
    await db["evalpen_document_grading_runs"].insert_one(
        {
            "run_id": run_id,
            "submission_id": "SUB-DOC-1",
            "student_id": "STU-1",
            "exam_id": "EXAM-DOC-1",
            "grading_revision": 2,
            "generation_revision": 2,
            "input_fingerprint": input_fingerprint,
            "generation_fingerprint": generation_fingerprint,
            "status": "generating",
            "generation_lease_token": "dead-worker",
            "generation_lease_expires_at": module.datetime.now(module.timezone.utc)
            - module.timedelta(minutes=1),
        }
    )

    existing, lease_token = await module._claim_or_wait_for_run(
        db,
        run_id=run_id,
        input_fingerprint=input_fingerprint,
        generation_fingerprint=generation_fingerprint,
        submission_id="SUB-DOC-1",
        student_id="STU-1",
        exam_id="EXAM-DOC-1",
        generation_revision=2,
        requested_model_id="gpt-5.1-2025-11-13",
        page_count=1,
        prompt_version="pcr-full-document-visual-v4",
    )

    assert existing is None
    assert lease_token and lease_token != "dead-worker"
    reclaimed = await db["evalpen_document_grading_runs"].find_one(
        {"run_id": run_id}
    )
    assert reclaimed["status"] == "generating"
    assert reclaimed["generation_lease_token"] == lease_token


@pytest.mark.asyncio
async def test_run_identity_collision_fails_closed_without_mutation():
    db = _fresh_db()
    module = _module()
    input_fingerprint = "b" * 64
    generation_fingerprint = module._generation_fingerprint(
        submission_id="SUB-DOC-1",
        input_fingerprint=input_fingerprint,
        generation_revision=1,
    )
    run_id = f"DOCGR-{generation_fingerprint[:24]}"
    await db["evalpen_document_grading_runs"].insert_one(
        {
            "run_id": run_id,
            "submission_id": "OTHER-SUBMISSION",
            "grading_revision": 1,
            "generation_revision": 1,
            "input_fingerprint": input_fingerprint,
            "generation_fingerprint": generation_fingerprint,
            "status": "completed",
            "result": {"evaluated_count": 99},
        }
    )

    with pytest.raises(module.GradingRunIdentityError):
        await module._claim_or_wait_for_run(
            db,
            run_id=run_id,
            input_fingerprint=input_fingerprint,
            generation_fingerprint=generation_fingerprint,
            submission_id="SUB-DOC-1",
            student_id="STU-1",
            exam_id="EXAM-DOC-1",
            generation_revision=1,
            requested_model_id="gpt-5.1-2025-11-13",
            page_count=1,
            prompt_version="pcr-full-document-visual-v4",
        )

    untouched = await db["evalpen_document_grading_runs"].find_one(
        {"run_id": run_id}
    )
    assert untouched["submission_id"] == "OTHER-SUBMISSION"
    assert untouched["status"] == "completed"
    assert untouched["result"] == {"evaluated_count": 99}


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
