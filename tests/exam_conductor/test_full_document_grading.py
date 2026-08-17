from __future__ import annotations

import base64
import io
import json
from types import SimpleNamespace
from typing import Any

import pytest
from mongomock_motor import AsyncMongoMockClient


def _module():
    from api.v1._exampen_imports import load_exampen

    return load_exampen("pcr.services.full_document_grading")


def _graph_module():
    from api.v1._exampen_imports import load_exampen

    return load_exampen("pcr.services.visual_evidence_graph")


def _db():
    return AsyncMongoMockClient()["skb_test"]


def _usage(caller: str = "pcr_eval_core") -> SimpleNamespace:
    return SimpleNamespace(
        model="gpt-5.1-2025-11-13",
        caller=caller,
        input_tokens=1_000,
        output_tokens=200,
        cache_read_tokens=100,
        total_tokens=1_200,
        estimated_cost_usd=0.01,
    )


def test_v14_reuses_immutable_prompt_prefix_across_mapping_and_grading_units():
    module = _module()
    catalog = [_question(1, "c1"), _question(2, "c2")]
    mapping = SimpleNamespace(
        questions={1: {}, 2: {}},
        as_payload=lambda: {"questions": []},
    )

    mapping_one = module._build_compact_mapping_responses_input(
        catalog=catalog,
        paper_bytes=b"paper",
        student_content=[{"type": "input_text", "text": "page one"}],
        paper_filename="paper.pdf",
        page_numbers=[1],
    )
    mapping_two = module._build_compact_mapping_responses_input(
        catalog=catalog,
        paper_bytes=b"paper",
        student_content=[{"type": "input_text", "text": "page two"}],
        paper_filename="paper.pdf",
        page_numbers=[2],
    )
    assert mapping_one[:2] == mapping_two[:2]
    assert mapping_one[2] != mapping_two[2]

    grading_one = module._build_evidence_grading_responses_input(
        catalog=catalog,
        requested_question_numbers=[1],
        mapping=mapping,
        paper_bytes=b"paper",
        solution_bytes=b"solution",
        student_content=[{"type": "input_text", "text": "answer one"}],
        paper_filename="paper.pdf",
        solution_filename="solution.pdf",
    )
    grading_two = module._build_evidence_grading_responses_input(
        catalog=catalog,
        requested_question_numbers=[2],
        mapping=mapping,
        paper_bytes=b"paper",
        solution_bytes=b"solution",
        student_content=[{"type": "input_text", "text": "answer two"}],
        paper_filename="paper.pdf",
        solution_filename="solution.pdf",
    )
    assert grading_one[:2] == grading_two[:2]
    assert grading_one[2] != grading_two[2]
    static_text = "\n".join(
        str(item.get("text") or "")
        for item in grading_one[1]["content"]
        if item.get("type") == "input_text"
    )
    assert '"question_number":1' in static_text
    assert '"question_number":2' in static_text

    verification_one = module._build_verification_responses_input(
        catalog=catalog,
        requested_question_numbers=[1],
        mapping=mapping,
        paper_bytes=b"paper",
        solution_bytes=b"solution",
        student_content=[{"type": "input_text", "text": "answer one"}],
        paper_filename="paper.pdf",
        solution_filename="solution.pdf",
    )
    verification_two = module._build_verification_responses_input(
        catalog=catalog,
        requested_question_numbers=[2],
        mapping=mapping,
        paper_bytes=b"paper",
        solution_bytes=b"solution",
        student_content=[{"type": "input_text", "text": "answer two"}],
        paper_filename="paper.pdf",
        solution_filename="solution.pdf",
    )
    assert verification_one[:2] == verification_two[:2]
    assert verification_one[2] != verification_two[2]


class _Gate:
    def __init__(self, payloads: list[Any]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    async def call(self, model_id: str, prompt: str, caller_id: str, **kwargs: Any):
        self.calls.append({"model_id": model_id, "caller_id": caller_id, **kwargs})
        value = self.payloads.pop(0)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, SimpleNamespace):
            return value
        return SimpleNamespace(
            content=json.dumps(value, ensure_ascii=False),
            usage=_usage(caller_id),
            completion_status="completed",
            incomplete_reason="",
        )


async def _value(value: Any) -> Any:
    return value


def _question(number: int, criterion_id: str) -> dict[str, Any]:
    return {
        "question_id": f"EXAM-1::Q{number}",
        "exam_id": "EXAM-1",
        "question_number": number,
        "question_text": f"Question {number}",
        "reference_solution": "Reference answer",
        "max_marks": 2,
        "marking_criteria": [
            {
                "criterion_id": criterion_id,
                "description": "Correct answer",
                "max_marks": 2,
                "acceptable_evidence": "A correct response in any equivalent wording.",
            }
        ],
    }


def _region(number: int, *, page: int | None = None, sequence: int = 1) -> dict[str, Any]:
    y_start = 80 + ((number - 1) % 4) * 220
    return {
        "region_id": f"q{number}-r{sequence}",
        "page_number": page or number,
        "x_start": 100,
        "y_start": y_start,
        "x_end": 900,
        "y_end": y_start + 180,
        "evidence_kind": "handwriting",
        "authorship": "student",
        "continuation_group": f"q{number}" if sequence > 1 else "",
        "sequence": sequence,
        "observed_content": f"दिखा हुआ उत्तर {number}",
        "diagram_components": [],
        "mapping_confidence": 0.91,
    }


def _mapped_question(
    number: int,
    *,
    status: str = "attempted",
    regions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    actual_regions = (
        [_region(number, page=1)] if regions is None and status == "attempted" else regions or []
    )
    return {
        "question_number": number,
        "attempt_status": status,
        "content_type": "TEXT_ONLY",
        "student_answer": f"दिखा हुआ उत्तर {number}" if actual_regions else "",
        "evidence_regions": actual_regions,
        "mapping_reason": "Question wording and continuation match.",
        "needs_review": status == "unresolved",
        "review_reason": "Ownership is unclear." if status == "unresolved" else "",
    }


def _mapping_payload(*questions: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
        "document_review": {
            "all_student_work_accounted": True,
            "teacher_annotations_present": True,
            "teacher_annotations_excluded": True,
            "warnings": [],
        },
        "questions": list(questions),
        "unassigned_student_regions": [],
    }


def _graded_question(
    number: int,
    criterion_id: str,
    *,
    marks: float = 2,
    region_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "question_number": number,
        "confidence": 0.9,
        "criterion_marks": [
            {
                "criterion_id": criterion_id,
                "confidence": 0.9,
                "marks_awarded": marks,
                "rationale": "The visible work satisfies the criterion.",
                "evidence": f"Visible answer for question {number}.",
                "evidence_region_ids": region_ids or [f"q{number}-r1"],
                "credit_basis": "direct_evidence" if marks else "no_credit",
            }
        ],
        "total_score": marks,
        "overall_feedback": "Correct method and answer." if marks == 2 else "Check the answer.",
        "needs_review": False,
        "review_reason": "",
    }


def _grading_payload(*questions: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
        "questions": list(questions),
    }


async def _seed(db) -> None:
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-1",
            "exam_id": "EXAM-1",
            "student_id": "STU-1",
            "source": "camera",
            "content_hash": "copy-hash",
        }
    )
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "EXAM-1",
            "exam_type": "pcr",
            "prepared_document_id": "DOC-1",
            "paper_version_id": "PAPER-1",
            "paper_content_hash": "paper-hash",
        }
    )
    await db["exampen_paper_versions"].insert_one(
        {"paper_version_id": "PAPER-1", "document_id": "DOC-1"}
    )
    await db["documents"].insert_one(
        {
            "document_id": "DOC-1",
            "file_path": "question.pdf",
            "filename": "question.pdf",
            "answer_sheet_path": "solution.pdf",
            "answer_sheet_filename": "solution.pdf",
        }
    )
    await db["evalpen_questions"].insert_many(
        [_question(1, "c1"), _question(2, "c2")]
    )
    await db["evalpen_answer_pages"].insert_one(
        {
            "page_id": "PAGE-1",
            "submission_id": "SUB-1",
            "page_number": 1,
            "raw_image_ref": "page.jpg",
            "content_hash": "page-hash",
        }
    )


def test_mapping_catalog_contains_no_answer_or_marking_leakage():
    module = _module()
    catalog = module._mapping_catalog_question(_question(1, "c1"))
    serialized = json.dumps(catalog)

    assert catalog["question_number"] == 1
    assert "question_text" in catalog
    for forbidden in (
        "reference_solution",
        "marking_criteria",
        "acceptable_evidence",
        "max_marks",
    ):
        assert forbidden not in serialized


def test_prompts_separate_mapping_from_grading_responsibility():
    graph = _graph_module()
    mapping_prompt = graph.mapping_system_instructions()
    grading_prompt = graph.grading_system_instructions()

    assert "never grades" in mapping_prompt
    assert "no answer key" in mapping_prompt
    assert "Teacher ticks" in mapping_prompt
    assert "Hindi" in mapping_prompt
    assert "cite only region_id" in grading_prompt
    assert "diagram" in grading_prompt
    assert "must never overwrite" not in grading_prompt


def test_mapping_schema_requires_exact_2d_student_regions():
    graph = _graph_module()
    schema = graph.evidence_mapping_schema([{"question_number": 1}])
    question = schema["properties"]["questions"]["items"]
    props = question["properties"]
    region_props = props["evidence_regions"]["items"]["properties"]

    assert props["question_number"]["enum"] == [1]
    assert "student_answer" in props
    assert region_props["x_end"]["maximum"] == 1000
    assert region_props["authorship"]["enum"] == ["student", "uncertain"]
    assert "continuation_group" in region_props
    assert "diagram_components" in region_props


def test_v14_compact_mapping_schema_excludes_verbose_mapper_fields():
    graph = _graph_module()
    schema = graph.compact_mapping_schema([{"question_number": 1}])
    question = schema["properties"]["questions"]["items"]
    assert graph.V14_PROMPT_VERSION == "pcr-full-document-visual-v14"
    assert "student_answer" not in question["properties"]
    assert "mapping_reason" not in question["properties"]
    region = question["properties"]["evidence_regions"]["items"]["properties"]
    assert "observed_content" not in region
    assert "mapping_confidence" not in region
    assert region["x_end"]["maximum"] == 1000


def test_mapping_validation_preserves_jumbled_multi_page_continuations():
    graph = _graph_module()
    first = _region(1, page=3, sequence=1)
    first["continuation_group"] = "answer-one"
    second = _region(1, page=1, sequence=2)
    second["continuation_group"] = "answer-one"
    payload = _mapping_payload(
        _mapped_question(1, regions=[second, first]),
        _mapped_question(2, regions=[_region(2, page=2)]),
    )

    result = graph.validate_mapping_payload(
        payload, question_numbers=[1, 2], page_count=3
    )

    assert result.errors == []
    assert [
        region["page_number"] for region in result.questions[1]["evidence_regions"]
    ] == [3, 1]
    assert all(
        region["coordinate_space"] == "normalized_1000"
        for region in result.questions[1]["evidence_regions"]
    )


def test_uncertain_authorship_cannot_be_graded_as_student_work():
    graph = _graph_module()
    region = _region(1, page=1)
    region["authorship"] = "uncertain"
    result = graph.validate_mapping_payload(
        _mapping_payload(_mapped_question(1, regions=[region])),
        question_numbers=[1],
        page_count=1,
    )

    assert result.questions[1]["attempt_status"] == "unresolved"
    assert result.questions[1]["needs_review"] is True


def test_same_physical_work_cannot_be_owned_by_two_questions():
    graph = _graph_module()
    first = _region(1, page=1)
    second = _region(2, page=1)
    for key in ("x_start", "y_start", "x_end", "y_end"):
        second[key] = first[key]
    result = graph.validate_mapping_payload(
        _mapping_payload(
            _mapped_question(1, regions=[first]),
            _mapped_question(2, regions=[second]),
        ),
        question_numbers=[1, 2],
        page_count=1,
    )

    assert result.questions[1]["attempt_status"] == "unresolved"
    assert result.questions[2]["attempt_status"] == "unresolved"
    assert "same student evidence" in " ".join(result.errors)


def test_not_attempted_requires_complete_copy_coverage():
    graph = _graph_module()
    payload = _mapping_payload(_mapped_question(1, status="not_attempted"))
    payload["document_review"]["all_student_work_accounted"] = False

    result = graph.validate_mapping_payload(
        payload, question_numbers=[1], page_count=1
    )

    assert result.questions[1]["attempt_status"] == "unresolved"


def test_grading_schema_locks_criterion_to_question_region_ids():
    module = _module()
    graph = _graph_module()
    question = module._catalog_question(_question(1, "c1"))
    mapping = graph.validate_mapping_payload(
        _mapping_payload(_mapped_question(1)), question_numbers=[1], page_count=1
    )
    schema = graph.grading_schema([question], mapping)
    props = schema["properties"]["questions"]["items"]["properties"]
    criterion = props["criterion_marks"]["items"]

    assert "student_answer" not in props
    assert criterion["properties"]["criterion_id"]["enum"] == ["c1"]
    assert criterion["properties"]["evidence_region_ids"]["items"]["enum"] == [
        "q1-r1"
    ]


def test_mapper_owned_hindi_answer_survives_merge_and_scores():
    module = _module()
    graph = _graph_module()
    mapping = graph.validate_mapping_payload(
        _mapping_payload(_mapped_question(1)), question_numbers=[1], page_count=1
    )
    merged = graph.merge_mapping_and_grading(
        mapping, _grading_payload(_graded_question(1, "c1"))
    )

    grades, errors, review = module._validate_ledger(
        merged, questions=[_question(1, "c1")], page_count=1
    )

    assert errors == []
    assert review.required is False
    assert grades[0].student_answer == "दिखा हुआ उत्तर 1"
    assert grades[0].total_score == 2
    assert grades[0].source_pages[0]["coordinate_space"] == "normalized_1000"


def test_criterion_without_owned_region_id_is_rejected():
    module = _module()
    graph = _graph_module()
    mapping = graph.validate_mapping_payload(
        _mapping_payload(_mapped_question(1)), question_numbers=[1], page_count=1
    )
    merged = graph.merge_mapping_and_grading(
        mapping,
        _grading_payload(_graded_question(1, "c1", region_ids=["q2-r1"])),
    )

    grades, _, _ = module._validate_ledger(
        merged, questions=[_question(1, "c1")], page_count=1
    )

    assert grades[0].attempt_status == "unresolved"
    assert "another question" in grades[0].review_reason


@pytest.mark.asyncio
async def test_primary_images_are_sent_without_recompression(monkeypatch):
    from PIL import Image

    module = _module()
    output = io.BytesIO()
    Image.new("RGB", (80, 120), "#222222").save(output, format="JPEG", quality=71)
    original = output.getvalue()
    monkeypatch.setattr(
        module,
        "_resolve_image_base64",
        lambda *args, **kwargs: _value(base64.b64encode(original).decode("ascii")),
    )

    content, size = await module._student_copy_content(
        [{"page_number": 1, "raw_image_ref": "page.jpg"}]
    )
    image_item = next(item for item in content if item["type"] == "input_image")

    assert base64.b64decode(image_item["image_url"].split(",", 1)[1]) == original
    assert size == len(original)
    assert "unaltered original" in content[-2]["text"]


@pytest.mark.asyncio
async def test_normal_service_path_uses_one_mapping_and_one_grading_call(monkeypatch):
    module = _module()
    db = _db()
    await _seed(db)
    await db["evalpen_evaluations"].insert_one(
        {
            "evaluation_id": "PEER-EVAL",
            "exam_id": "EXAM-1",
            "question_id": "EXAM-1::Q1",
            "student_id": "OTHER-STUDENT",
            "prompt_version": "pcr-full-document-visual-v13",
            "model_used": "gpt-5.1-2025-11-13",
            "grading_consistency_key": "same-transcription",
            "manual_review_required": False,
            "total_score": 0.0,
            "max_score": 2.0,
            "criterion_marks": [
                {"criterion_id": "c1", "marks_awarded": 0.0, "max_marks": 2.0}
            ],
        }
    )
    gate = _Gate(
        [
            _mapping_payload(_mapped_question(1), _mapped_question(2)),
            _grading_payload(
                _graded_question(1, "c1"),
                _graded_question(2, "c2"),
            ),
        ]
    )
    monkeypatch.setattr(
        module,
        "_read_canonical_file",
        lambda *args, **kwargs: _value(b"%PDF-1.4 canonical"),
    )
    monkeypatch.setattr(
        module,
        "_student_copy_content",
        lambda pages: _value(([{"type": "input_text", "text": "complete copy"}], 20)),
    )
    monkeypatch.setattr(
        module,
        "_grading_consistency_key",
        lambda **kwargs: "same-transcription",
    )

    result = await module.FullDocumentGradingService(
        db, gate, model_id="gpt-5.1-2025-11-13"
    ).grade_submission("SUB-1")

    assert [call["metadata"]["pcr_stage"] for call in gate.calls] == [
        "student_evidence_mapping",
        "mapped_evidence_grading",
    ]
    assert [call["metadata"]["provider_call_number"] for call in gate.calls] == [1, 2]
    assert all(call["metadata"]["provider_call_limit"] == 2 for call in gate.calls)
    mapping_files = [
        item
        for message in gate.calls[0]["responses_input"]
        for item in message["content"]
        if item["type"] == "input_file"
    ]
    grading_files = [
        item
        for message in gate.calls[1]["responses_input"]
        for item in message["content"]
        if item["type"] == "input_file"
    ]
    mapping_text = "\n".join(
        str(item.get("text") or "")
        for message in gate.calls[0]["responses_input"]
        for item in message["content"]
        if item["type"] == "input_text"
    )
    grading_text = "\n".join(
        str(item.get("text") or "")
        for message in gate.calls[1]["responses_input"]
        for item in message["content"]
        if item["type"] == "input_text"
    )
    assert len(mapping_files) == 1
    assert len(grading_files) == 2
    assert "Reference answer" not in mapping_text
    assert '"acceptable_evidence":' not in mapping_text
    assert "Reference answer" in grading_text
    assert result.evaluated_count == 2
    assert result.blocked_count == 0
    assert result.review_state == "ready"
    run = await db["evalpen_document_grading_runs"].find_one({"run_id": result.run_id})
    assert run["prompt_version"] == "pcr-full-document-visual-v13"
    assert run["token_usage"]["stage_count"] == 2
    exam = await db["exampen_exams"].find_one({"exam_id": "EXAM-1"})
    assert exam["pcr_grading_contract"]["prompt_version"] == (
        "pcr-full-document-visual-v13"
    )
    response = await db["evalpen_detected_responses"].find_one(
        {"submission_id": "SUB-1", "question_number": 1}
    )
    assert response["question_assignment"]["method"] == "evidence_first_visual_v13"
    assert response["source_pages"][0]["region_id"] == "q1-r1"
    evaluation = await db["evalpen_evaluations"].find_one(
        {"student_id": "STU-1", "question_id": "EXAM-1::Q1"}
    )
    assert evaluation["criterion_marks"][0]["evidence_region_ids"] == ["q1-r1"]
    assert evaluation["total_score"] == 2.0
    assert evaluation["consistency_calibration"] is None


@pytest.mark.asyncio
async def test_selected_legacy_copy_uses_v16_without_mutating_exam_contract(monkeypatch):
    from datetime import datetime, timezone
    from services.pcr_grading_contract_policy import build_selected_copy_v16_override

    module = _module()
    db = _db()
    await _seed(db)
    source_contract = {
        "prompt_version": "pcr-full-document-visual-v12",
        "pipeline_version": 3,
        "mapping_pipeline_version": "evidence-first-visual-v3",
        "required_processing_path": "full_document_visual",
        "model_id": "gpt-5.1-2025-11-13",
    }
    await db["exampen_exams"].update_one(
        {"exam_id": "EXAM-1"},
        {"$set": {"pcr_grading_contract": source_contract}},
    )
    override = build_selected_copy_v16_override(
        source_contract,
        submission_id="SUB-1",
        requested_by="TUT-1",
        requested_at=datetime.now(timezone.utc),
    )
    await db["exampen_processing_jobs"].insert_one(
        {
            "job_id": "pcr-job-SUB-1",
            "submission_id": "SUB-1",
            "exam_id": "EXAM-1",
            "status": "processing",
            "pipeline_version": 7,
            "mapping_pipeline_version": "whole-copy-rubric-v7",
            "required_processing_path": "full_document_visual",
            "reprocess_count": 1,
            "grading_contract_override": override,
        }
    )
    payload = {
        "all_student_work_accounted": True,
        "questions": [
            {
                "question_number": number,
                "attempt_status": "attempted",
                "confidence": 0.94,
                "student_answer": f"Visible answer {number}",
                "content_type": "TEXT_ONLY",
                "source_pages": [1],
                "criterion_marks": [
                    {
                        "criterion_id": criterion_id,
                        "marks_awarded": 2,
                        "confidence": 0.92,
                        "rationale": "The visible work satisfies the criterion.",
                        "evidence": "The answer is visible on page 1.",
                        "credit_basis": "direct_evidence",
                    }
                ],
                "total_score": 2,
                "overall_feedback": "Correct.",
                "needs_review": False,
                "review_reason": "",
            }
            for number, criterion_id in ((1, "c1"), (2, "c2"))
        ],
    }
    gate = _Gate([payload])
    monkeypatch.setattr(
        module,
        "_read_canonical_file",
        lambda *args, **kwargs: _value(b"%PDF-1.4 canonical"),
    )
    monkeypatch.setattr(
        module,
        "_student_copy_content",
        lambda *args, **kwargs: _value(
            ([{"type": "input_text", "text": "complete copy"}], 20)
        ),
    )

    async def must_not_freeze_exam(*_args, **_kwargs):
        raise AssertionError("A selected-copy override must not mutate the exam contract")

    monkeypatch.setattr(module, "_freeze_exam_grading_contract", must_not_freeze_exam)

    result = await module.FullDocumentGradingService(
        db, gate, model_id="gpt-5.1-2025-11-13"
    ).grade_submission("SUB-1")

    assert result.evaluated_count == 2
    assert len(gate.calls) == 1
    assert gate.calls[0]["metadata"]["pcr_stage"] == "whole_copy_visual_grading"
    run = await db["evalpen_document_grading_runs"].find_one({"run_id": result.run_id})
    assert run["prompt_version"] == "pcr-full-document-visual-v16"
    assert run["contract_scope"] == "selected_submission_reprocess"
    assert run["contract_override_id"] == override["override_id"]
    assert run["source_prompt_version"] == "pcr-full-document-visual-v12"
    exam = await db["exampen_exams"].find_one({"exam_id": "EXAM-1"})
    assert exam["pcr_grading_contract"] == source_contract


@pytest.mark.asyncio
async def test_no_attempts_need_mapping_only_and_receive_zero(monkeypatch):
    module = _module()
    db = _db()
    await _seed(db)
    gate = _Gate(
        [
            _mapping_payload(
                _mapped_question(1, status="not_attempted"),
                _mapped_question(2, status="not_attempted"),
            )
        ]
    )
    monkeypatch.setattr(
        module,
        "_read_canonical_file",
        lambda *args, **kwargs: _value(b"%PDF-1.4 canonical"),
    )
    monkeypatch.setattr(
        module,
        "_student_copy_content",
        lambda pages: _value(([{"type": "input_text", "text": "complete copy"}], 20)),
    )

    result = await module.FullDocumentGradingService(
        db, gate, model_id="gpt-5.1-2025-11-13"
    ).grade_submission("SUB-1")

    assert len(gate.calls) == 1
    assert result.evaluated_count == 2
    evaluations = await db["evalpen_evaluations"].find(
        {"exam_id": "EXAM-1"}
    ).to_list(length=None)
    assert {row["total_score"] for row in evaluations} == {0.0}


@pytest.mark.asyncio
async def test_incomplete_mapping_fails_terminally_without_grading_call(monkeypatch):
    module = _module()
    db = _db()
    await _seed(db)
    gate = _Gate(
        [
            SimpleNamespace(
                content="",
                usage=_usage(),
                completion_status="incomplete",
                incomplete_reason="max_output_tokens",
            )
        ]
    )
    monkeypatch.setattr(
        module,
        "_read_canonical_file",
        lambda *args, **kwargs: _value(b"%PDF-1.4 canonical"),
    )
    monkeypatch.setattr(
        module,
        "_student_copy_content",
        lambda pages: _value(([{"type": "input_text", "text": "complete copy"}], 20)),
    )

    with pytest.raises(module.StructuredGradingOutputError):
        await module.FullDocumentGradingService(
            db, gate, model_id="gpt-5.1-2025-11-13"
        ).grade_submission("SUB-1")

    assert len(gate.calls) == 1
    assert gate.calls[0]["max_output_tokens"] == 10_000
    run = await db["evalpen_document_grading_runs"].find_one(
        {"prompt_version": "pcr-full-document-visual-v13"}
    )
    assert run["status"] == "failed"
    assert run["structured_output_failure"]["incomplete_reason"] == "max_output_tokens"


@pytest.mark.asyncio
async def test_mapping_checkpoint_resumes_without_purchasing_mapping_again():
    module = _module()
    db = _db()
    mapping_payload = _mapping_payload(_mapped_question(1))
    await db["evalpen_document_grading_runs"].insert_one(
        {
            "run_id": "RUN-1",
            "generation_lease_token": "LEASE-1",
            "evidence_mapping_payload": mapping_payload,
            "evidence_mapping_raw": json.dumps(mapping_payload),
            "evidence_mapping_usage": {
                "model": "gpt-5.1-2025-11-13",
                "total_tokens": 100,
            },
        }
    )
    gate = _Gate([_grading_payload(_graded_question(1, "c1"))])

    merged, _, usage = await module._run_evidence_first_grading(
        db=db,
        gate=gate,
        existing_run=await db["evalpen_document_grading_runs"].find_one(
            {"run_id": "RUN-1"}
        ),
        generation_lease_token="LEASE-1",
        run_id="RUN-1",
        submission_id="SUB-1",
        exam_id="EXAM-1",
        questions=[_question(1, "c1")],
        page_count=1,
        paper_bytes=b"paper",
        solution_bytes=b"solution",
        student_content=[{"type": "input_text", "text": "copy"}],
        paper_filename="paper.pdf",
        solution_filename="solution.pdf",
        model_id="gpt-5.1-2025-11-13",
        reasoning_effort="medium",
        temperature=0,
        paper_hash="paper-hash",
        solution_hash="solution-hash",
    )

    assert len(gate.calls) == 1
    assert gate.calls[0]["metadata"]["provider_call_number"] == 2
    assert merged["questions"][0]["student_answer"] == "दिखा हुआ उत्तर 1"
    assert usage["stage_count"] == 2


@pytest.mark.asyncio
async def test_v14_max_output_splits_question_unit_and_uses_bounded_batches():
    module = _module()
    db = _db()
    await db["evalpen_document_grading_runs"].insert_one(
        {"run_id": "RUN-V14", "generation_lease_token": "LEASE-V14"}
    )
    compact = lambda number: {
        "mapping_version": "pcr-compact-evidence-map-v1",
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
        "all_student_work_accounted": True,
        "questions": [{
            "question_number": number,
            "attempt_status": "attempted",
            "content_type": "TEXT_ONLY",
            "evidence_regions": [{
                key: value for key, value in _region(number, page=1).items()
                if key not in {"observed_content", "diagram_components", "mapping_confidence"}
            }],
        }],
        "unassigned_student_regions": [],
    }
    gate = _Gate([
        SimpleNamespace(content="", usage=_usage(), completion_status="incomplete", incomplete_reason="max_output_tokens"),
        compact(1),
        compact(2),
        _grading_payload(_graded_question(1, "c1"), _graded_question(2, "c2")),
        {
            "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
            "questions": [
                _audit_payload(1, "c1", 2, "q1-r1")["questions"][0],
                _audit_payload(2, "c2", 2, "q2-r1")["questions"][0],
            ],
        },
    ])
    merged, _, usage = await module._run_evidence_first_grading(
        db=db,
        gate=gate,
        existing_run=await db["evalpen_document_grading_runs"].find_one({"run_id": "RUN-V14"}),
        generation_lease_token="LEASE-V14",
        run_id="RUN-V14",
        submission_id="SUB-1",
        exam_id="EXAM-1",
        questions=[_question(1, "c1"), _question(2, "c2")],
        page_count=1,
        paper_bytes=b"paper",
        solution_bytes=b"solution",
        student_content=[{"type": "input_text", "text": "copy"}],
        paper_filename="paper.pdf",
        solution_filename="solution.pdf",
        model_id="gpt-5.1-2025-11-13",
        reasoning_effort="medium",
        temperature=0,
        paper_hash="paper-hash",
        solution_hash="solution-hash",
        pipeline_version="pcr-full-document-visual-v14",
    )

    assert len(gate.calls) == 5
    assert gate.calls[0]["max_output_tokens"] == 4000
    assert gate.calls[1]["metadata"]["unit_id"].endswith("-a")
    assert gate.calls[2]["metadata"]["unit_id"].endswith("-b")
    assert [item["question_number"] for item in merged["questions"]] == [1, 2]
    # Failed max-output attempts have no provider usage record; successful
    # bounded units, grading, and audit are the auditable stages.
    assert usage["stage_count"] == 4
    run = await db["evalpen_document_grading_runs"].find_one({"run_id": "RUN-V14"})
    assert len(run["evidence_mapping_units"]) == 2


def _audit_payload(number: int, criterion_id: str, marks: float, region_id: str) -> dict[str, Any]:
    return {
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
        "questions": [{
            "question_number": number,
            "criterion_marks": [{
                "criterion_id": criterion_id,
                "marks_supported": marks,
                "evidence_region_ids": [region_id],
                "rationale": "Independent visible evidence supports this criterion.",
            }],
        }],
    }


@pytest.mark.asyncio
async def test_v14_full_score_audit_agreement_keeps_score_and_does_not_see_primary_marks():
    module = _module()
    db = _db()
    await db["evalpen_document_grading_runs"].insert_one(
        {"run_id": "RUN-AUDIT-AGREE", "generation_lease_token": "LEASE-AUDIT-AGREE"}
    )
    gate = _Gate([
        _mapping_payload(_mapped_question(1)),
        _grading_payload(_graded_question(1, "c1")),
        _audit_payload(1, "c1", 2, "q1-r1"),
    ])
    merged, raw, usage = await module._run_evidence_first_grading(
        db=db, gate=gate,
        existing_run=await db["evalpen_document_grading_runs"].find_one({"run_id": "RUN-AUDIT-AGREE"}),
        generation_lease_token="LEASE-AUDIT-AGREE", run_id="RUN-AUDIT-AGREE",
        submission_id="SUB-1", exam_id="EXAM-1", questions=[_question(1, "c1")],
        page_count=1, paper_bytes=b"paper", solution_bytes=b"solution",
        student_content=[{"type": "input_text", "text": "copy"}],
        paper_filename="paper.pdf", solution_filename="solution.pdf",
        model_id="gpt-5.1-2025-11-13", reasoning_effort="medium", temperature=0,
        paper_hash="paper-hash", solution_hash="solution-hash",
        pipeline_version=module._V14_PROMPT_VERSION,
    )
    assert merged["questions"][0]["total_score"] == 2
    assert merged["questions"][0]["needs_review"] is False
    assert gate.calls[-1]["metadata"]["pcr_stage"] == "bounded_full_score_verification"
    assert gate.calls[-1]["metadata"]["primary_marks_in_input"] is False
    assert '"marks_awarded"' not in "\n".join(
        str(item.get("text") or "")
        for message in gate.calls[-1]["responses_input"]
        for item in message.get("content", [])
        if item.get("type") == "input_text"
    )
    assert usage["stage_count"] == 3
    assert '"verification"' in raw


@pytest.mark.asyncio
async def test_v14_unsupported_full_score_audit_flags_without_changing_score():
    module = _module()
    db = _db()
    await db["evalpen_document_grading_runs"].insert_one(
        {"run_id": "RUN-AUDIT-FLAG", "generation_lease_token": "LEASE-AUDIT-FLAG"}
    )
    gate = _Gate([
        _mapping_payload(_mapped_question(1)),
        _grading_payload(_graded_question(1, "c1")),
        _audit_payload(1, "c1", 0, "q1-r1"),
    ])
    merged, _, _ = await module._run_evidence_first_grading(
        db=db, gate=gate,
        existing_run=await db["evalpen_document_grading_runs"].find_one({"run_id": "RUN-AUDIT-FLAG"}),
        generation_lease_token="LEASE-AUDIT-FLAG", run_id="RUN-AUDIT-FLAG",
        submission_id="SUB-1", exam_id="EXAM-1", questions=[_question(1, "c1")],
        page_count=1, paper_bytes=b"paper", solution_bytes=b"solution",
        student_content=[{"type": "input_text", "text": "copy"}],
        paper_filename="paper.pdf", solution_filename="solution.pdf",
        model_id="gpt-5.1-2025-11-13", reasoning_effort="medium", temperature=0,
        paper_hash="paper-hash", solution_hash="solution-hash",
        pipeline_version=module._V14_PROMPT_VERSION,
    )
    assert merged["questions"][0]["total_score"] == 2
    assert merged["questions"][0]["needs_review"] is True
    assert "disagreed" in merged["questions"][0]["review_reason"]


@pytest.mark.asyncio
async def test_v14_full_score_audit_skips_non_full_and_objective_questions():
    module = _module()
    db = _db()
    question = _question(1, "c1")
    objective = _question(2, "c2")
    objective["grading_mode"] = "objective"
    await db["evalpen_document_grading_runs"].insert_one(
        {"run_id": "RUN-AUDIT-SKIP", "generation_lease_token": "LEASE-AUDIT-SKIP"}
    )
    partial = _graded_question(1, "c1", marks=1)
    objective_grade = _graded_question(2, "c2", marks=0)
    gate = _Gate([
        _mapping_payload(_mapped_question(1), _mapped_question(2)),
        _grading_payload(partial, objective_grade),
    ])
    merged, _, _ = await module._run_evidence_first_grading(
        db=db, gate=gate,
        existing_run=await db["evalpen_document_grading_runs"].find_one({"run_id": "RUN-AUDIT-SKIP"}),
        generation_lease_token="LEASE-AUDIT-SKIP", run_id="RUN-AUDIT-SKIP",
        submission_id="SUB-1", exam_id="EXAM-1", questions=[question, objective],
        page_count=1, paper_bytes=b"paper", solution_bytes=b"solution",
        student_content=[{"type": "input_text", "text": "copy"}],
        paper_filename="paper.pdf", solution_filename="solution.pdf",
        model_id="gpt-5.1-2025-11-13", reasoning_effort="medium", temperature=0,
        paper_hash="paper-hash", solution_hash="solution-hash",
        pipeline_version=module._V14_PROMPT_VERSION,
    )
    assert len(gate.calls) == 2
    assert all(call["metadata"]["pcr_stage"] != "bounded_full_score_verification" for call in gate.calls)
    assert {item["question_number"] for item in merged["questions"]} == {1, 2}


@pytest.mark.asyncio
async def test_v14_full_score_audit_resume_is_idempotent():
    module = _module()
    db = _db()
    mapped = _mapping_payload(_mapped_question(1))
    compact = module._compact_payload_from_response(mapped)
    grade = _grading_payload(_graded_question(1, "c1"))
    audit = _audit_payload(1, "c1", 2, "q1-r1")
    usage = {"model": "gpt-5.1-2025-11-13", "total_tokens": 1}
    await db["evalpen_document_grading_runs"].insert_one({
        "run_id": "RUN-AUDIT-RESUME",
        "generation_lease_token": "LEASE-AUDIT-RESUME",
        "evidence_mapping_units": [{"unit_id": "pages-1", "payload": compact, "usage": usage}],
        "evidence_grading_batches": [{"batch_id": "batch-1", "payload": grade, "usage": usage}],
        "evidence_verification_batches": [{"batch_id": "audit-1", "payload": audit, "usage": usage}],
    })
    gate = _Gate([])
    merged, _, _ = await module._run_evidence_first_grading(
        db=db, gate=gate,
        existing_run=await db["evalpen_document_grading_runs"].find_one({"run_id": "RUN-AUDIT-RESUME"}),
        generation_lease_token="LEASE-AUDIT-RESUME", run_id="RUN-AUDIT-RESUME",
        submission_id="SUB-1", exam_id="EXAM-1", questions=[_question(1, "c1")],
        page_count=1, paper_bytes=b"paper", solution_bytes=b"solution",
        student_content=[{"type": "input_text", "text": "copy"}],
        paper_filename="paper.pdf", solution_filename="solution.pdf",
        model_id="gpt-5.1-2025-11-13", reasoning_effort="medium", temperature=0,
        paper_hash="paper-hash", solution_hash="solution-hash",
        pipeline_version=module._V14_PROMPT_VERSION,
    )
    assert gate.calls == []
    assert merged["questions"][0]["needs_review"] is False
