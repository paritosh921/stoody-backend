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


def _db():
    return AsyncMongoMockClient()["skb_test"]


def _usage(caller: str = "pcr_eval_core") -> SimpleNamespace:
    return SimpleNamespace(
        model="gpt-5.1-2025-11-13",
        caller=caller,
        input_tokens=1_000,
        output_tokens=200,
        cache_read_tokens=0,
        total_tokens=1_200,
        estimated_cost_usd=0.01,
    )


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


def _attempted(number: int, criterion_id: str, *, text: str = "सही उत्तर") -> dict[str, Any]:
    return {
        "question_number": number,
        "attempt_status": "attempted",
        "student_answer": text,
        "source_pages": [1],
        "criterion_marks": [
            {
                "criterion_id": criterion_id,
                "marks_awarded": 2,
                "rationale": "The answer is correct.",
                "evidence": text,
                "credit_basis": "direct_evidence",
            }
        ],
        "total_score": 2,
        "overall_feedback": "Correct answer.",
        "needs_review": False,
        "review_reason": "",
    }


def _unresolved(number: int) -> dict[str, Any]:
    return {
        "question_number": number,
        "attempt_status": "unresolved",
        "student_answer": "",
        "source_pages": [1],
        "criterion_marks": [],
        "total_score": 0,
        "overall_feedback": "",
        "needs_review": True,
        "review_reason": "The writing could not be read reliably.",
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


def test_prompt_is_whole_copy_and_language_aware():
    prompt = _module()._system_instructions()

    assert "Hindi" in prompt
    assert "every student page" in prompt
    assert "Do not depend on OCR, crops, coordinates, confidence" in prompt
    assert "unresolved only" in prompt
    assert "catalog grading_mode is authoritative" in prompt


def test_schema_is_small_and_has_no_mapping_or_confidence_contract():
    module = _module()
    schema = module._whole_copy_schema([_question(1, "c1"), _question(2, "c2")])
    questions_schema = schema["properties"]["questions"]
    variants = questions_schema["items"]["anyOf"]
    props = variants[0]["properties"]

    assert questions_schema["minItems"] == 2
    assert questions_schema["maxItems"] == 2
    assert [variant["properties"]["question_number"]["enum"] for variant in variants] == [
        [1],
        [2],
    ]
    assert "source_pages" in props
    assert "confidence" not in props
    assert "evidence_regions" not in props
    assert "document_review" not in schema["properties"]


def test_schema_forces_subjective_mcq_subparts_to_return_locked_criteria():
    module = _module()
    question = _question(5, "criterion_1")
    question.update(
        {
            "question_text": "Choose: (क) i/ii and (ख) i/ii",
            "grading_mode": "subjective",
            "question_type": "subjective",
        }
    )

    schema = module._whole_copy_schema([question])
    props = schema["properties"]["questions"]["items"]["properties"]
    criterion_schema = props["criterion_marks"]

    assert props["question_number"]["enum"] == [5]
    assert criterion_schema["minItems"] == 1
    assert criterion_schema["maxItems"] == 1
    assert criterion_schema["items"]["properties"]["criterion_id"]["enum"] == [
        "criterion_1"
    ]
    assert criterion_schema["items"]["properties"]["marks_awarded"]["maximum"] == 2


def test_output_limits_include_reasoning_headroom_without_unbounded_calls():
    module = _module()

    assert module._whole_copy_output_limit(9) == 24_000
    assert module._whole_copy_output_limit(50) == 32_000
    assert module._recovery_output_limit(1) == 12_000
    assert module._recovery_output_limit(50) == 20_000


def test_readable_hindi_grade_is_scoreable_without_confidence_metadata():
    module = _module()
    payload = {"questions": [_attempted(1, "c1")]}

    grades, errors, review = module._validate_ledger(
        payload,
        questions=[_question(1, "c1")],
        page_count=1,
    )

    assert errors == []
    assert review.required is False
    assert grades[0].total_score == 2
    assert grades[0].manual_review_required is False
    assert grades[0].source_pages[0]["page_number"] == 1


def test_not_attempted_is_accepted_after_the_whole_copy_scan():
    module = _module()
    payload = {
        "questions": [
            {
                "question_number": 1,
                "attempt_status": "not_attempted",
                "student_answer": "",
                "source_pages": [],
                "criterion_marks": [],
                "total_score": 0,
                "overall_feedback": "Question not attempted.",
                "needs_review": False,
                "review_reason": "",
            }
        ]
    }

    grades, _, _ = module._validate_ledger(
        payload, questions=[_question(1, "c1")], page_count=1
    )

    assert grades[0].attempt_status == "not_attempted"
    assert grades[0].total_score == 0
    assert grades[0].manual_review_required is False


def test_missing_or_out_of_range_question_fails_only_that_question():
    module = _module()
    invalid = _attempted(2, "c2")
    invalid["criterion_marks"][0]["marks_awarded"] = 3
    grades, _, _ = module._validate_ledger(
        {"questions": [invalid]},
        questions=[_question(1, "c1"), _question(2, "c2")],
        page_count=1,
    )

    assert [grade.attempt_status for grade in grades] == ["unresolved", "unresolved"]
    assert "no result" in grades[0].review_reason
    assert "outside its locked range" in grades[1].review_reason


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
    encoded = image_item["image_url"].split(",", 1)[1]

    assert base64.b64decode(encoded) == original
    assert size == len(original)
    assert "unaltered original" in content[-2]["text"]


@pytest.mark.asyncio
async def test_recovery_keeps_original_and_adds_only_sideways_upright_views(monkeypatch):
    from PIL import Image

    module = _module()
    output = io.BytesIO()
    Image.new("RGB", (80, 120), "#eeeeee").save(output, format="JPEG", quality=80)
    original = output.getvalue()
    monkeypatch.setattr(
        module,
        "_resolve_image_base64",
        lambda *args, **kwargs: _value(base64.b64encode(original).decode("ascii")),
    )
    monkeypatch.setattr(module, "detect_sideways_page", lambda data: (True, {}))

    content, size = await module._student_recovery_content(
        [{"page_number": 1, "raw_image_ref": "page.jpg"}]
    )
    images = [item for item in content if item["type"] == "input_image"]

    assert len(images) == 3
    assert base64.b64decode(images[0]["image_url"].split(",", 1)[1]) == original
    assert all(item["image_url"].startswith("data:image/jpeg;base64,") for item in images)
    for item in images[1:]:
        with Image.open(
            io.BytesIO(base64.b64decode(item["image_url"].split(",", 1)[1]))
        ) as rotated:
            assert rotated.size == (120, 80)
    assert size > len(original)


@pytest.mark.asyncio
async def test_complete_primary_result_makes_exactly_one_provider_call(monkeypatch):
    module = _module()
    gate = _Gate([])
    payload = {"questions": [_attempted(1, "c1")]}

    merged, raw, usage = await module._recover_unresolved_once(
        gate=gate,
        primary_payload=payload,
        primary_raw=json.dumps(payload),
        primary_usage={"model": "gpt-5.1", "total_tokens": 10},
        questions=[_question(1, "c1")],
        answer_pages=[{"page_number": 1}],
        paper_bytes=b"paper",
        solution_bytes=b"solution",
        document={},
        model_id="gpt-5.1",
        reasoning_effort="medium",
        temperature=0,
        submission_id="SUB-1",
        exam_id="EXAM-1",
        run_id="RUN-1",
    )

    assert merged == payload
    assert raw == json.dumps(payload)
    assert usage["total_tokens"] == 10
    assert gate.calls == []


@pytest.mark.asyncio
async def test_one_recovery_call_resolves_only_the_failed_questions(monkeypatch):
    module = _module()
    primary = {"questions": [_attempted(1, "c1"), _unresolved(2)]}
    gate = _Gate([{"questions": [_attempted(2, "c2", text="पुनः पढ़ा उत्तर")]}])
    monkeypatch.setattr(
        module,
        "_student_recovery_content",
        lambda pages: _value(([{"type": "input_text", "text": "complete pages"}], 10)),
    )

    merged, _, usage = await module._recover_unresolved_once(
        gate=gate,
        primary_payload=primary,
        primary_raw=json.dumps(primary),
        primary_usage={
            "model": "gpt-5.1",
            "caller": "pcr_eval_core",
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_tokens": 0,
            "total_tokens": 120,
            "estimated_cost_usd": 0.01,
        },
        questions=[_question(1, "c1"), _question(2, "c2")],
        answer_pages=[{"page_number": 1}],
        paper_bytes=b"paper",
        solution_bytes=b"solution",
        document={},
        model_id="gpt-5.1",
        reasoning_effort="medium",
        temperature=0,
        submission_id="SUB-1",
        exam_id="EXAM-1",
        run_id="RUN-1",
    )

    assert len(gate.calls) == 1
    assert gate.calls[0]["metadata"]["provider_call_number"] == 2
    assert gate.calls[0]["metadata"]["provider_call_limit"] == 2
    assert gate.calls[0]["json_schema"]["properties"]["questions"]["items"][
        "properties"
    ]["question_number"]["enum"] == [2]
    assert merged["questions"][0]["student_answer"] == "सही उत्तर"
    assert merged["questions"][1]["student_answer"] == "पुनः पढ़ा उत्तर"
    assert usage["stage_count"] == 2


@pytest.mark.asyncio
async def test_recovery_failure_keeps_the_primary_partial_result(monkeypatch):
    module = _module()
    primary = {"questions": [_unresolved(1)]}
    gate = _Gate([RuntimeError("provider unavailable")])
    monkeypatch.setattr(
        module,
        "_student_recovery_content",
        lambda pages: _value(([{"type": "input_text", "text": "pages"}], 10)),
    )

    merged, _, _ = await module._recover_unresolved_once(
        gate=gate,
        primary_payload=primary,
        primary_raw=json.dumps(primary),
        primary_usage={"model": "gpt-5.1"},
        questions=[_question(1, "c1")],
        answer_pages=[{"page_number": 1}],
        paper_bytes=b"paper",
        solution_bytes=None,
        document={},
        model_id="gpt-5.1",
        reasoning_effort="medium",
        temperature=0,
        submission_id="SUB-1",
        exam_id="EXAM-1",
        run_id="RUN-1",
    )

    assert len(gate.calls) == 1
    assert merged == primary


@pytest.mark.asyncio
async def test_invalid_recovery_does_not_replace_a_scoreable_primary_result(monkeypatch):
    module = _module()
    primary_item = _attempted(1, "c1")
    primary_item["needs_review"] = True
    primary_item["review_reason"] = "Please double-check one word."
    invalid_retry = _attempted(1, "c1")
    invalid_retry["criterion_marks"][0]["marks_awarded"] = 3
    gate = _Gate([{"questions": [invalid_retry]}])
    monkeypatch.setattr(
        module,
        "_student_recovery_content",
        lambda pages: _value(([{"type": "input_text", "text": "pages"}], 10)),
    )

    merged, _, _ = await module._recover_unresolved_once(
        gate=gate,
        primary_payload={"questions": [primary_item]},
        primary_raw=json.dumps({"questions": [primary_item]}),
        primary_usage={"model": "gpt-5.1"},
        questions=[_question(1, "c1")],
        answer_pages=[{"page_number": 1}],
        paper_bytes=b"paper",
        solution_bytes=None,
        document={},
        model_id="gpt-5.1",
        reasoning_effort="medium",
        temperature=0,
        submission_id="SUB-1",
        exam_id="EXAM-1",
        run_id="RUN-1",
    )

    assert len(gate.calls) == 1
    assert merged["questions"] == [primary_item]


@pytest.mark.asyncio
async def test_service_uses_one_whole_copy_call_and_materializes_scores(monkeypatch):
    module = _module()
    db = _db()
    await _seed(db)
    gate = _Gate([{"questions": [_attempted(1, "c1"), _attempted(2, "c2")] }])
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
    assert gate.calls[0]["metadata"]["provider_call_number"] == 1
    assert gate.calls[0]["metadata"]["provider_call_limit"] == 2
    assert result.evaluated_count == 2
    assert result.blocked_count == 0
    assert result.review_state == "ready"
    run = await db["evalpen_document_grading_runs"].find_one({"run_id": result.run_id})
    assert run["prompt_version"] == "pcr-full-document-visual-v12"
    assert run["token_usage"]["total_tokens"] == 1_200
    exam = await db["exampen_exams"].find_one({"exam_id": "EXAM-1"})
    assert exam["pcr_grading_contract"]["prompt_version"] == (
        "pcr-full-document-visual-v12"
    )


@pytest.mark.asyncio
async def test_incomplete_primary_output_fails_terminally_and_records_usage(monkeypatch):
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

    assert gate.calls[0]["max_output_tokens"] == 24_000
    run = await db["evalpen_document_grading_runs"].find_one(
        {"prompt_version": "pcr-full-document-visual-v12"}
    )
    assert run["status"] == "failed"
    assert run["structured_output_failure"] == {
        "completion_status": "incomplete",
        "incomplete_reason": "max_output_tokens",
        "max_output_tokens": 24_000,
    }
    assert run["token_usage"]["total_tokens"] == 1_200


@pytest.mark.asyncio
async def test_service_persists_good_questions_when_recovery_stays_unresolved(monkeypatch):
    module = _module()
    db = _db()
    await _seed(db)
    gate = _Gate(
        [
            {"questions": [_attempted(1, "c1"), _unresolved(2)]},
            {"questions": [_unresolved(2)]},
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
        "_student_recovery_content",
        lambda pages: _value(([{"type": "input_text", "text": "recovery copy"}], 20)),
    )

    result = await module.FullDocumentGradingService(
        db, gate, model_id="gpt-5.1-2025-11-13"
    ).grade_submission("SUB-1")

    assert len(gate.calls) == 2
    assert result.evaluated_count == 1
    assert result.blocked_count == 1
    responses = await db["evalpen_detected_responses"].find(
        {"submission_id": "SUB-1"}
    ).sort("question_number", 1).to_list(length=None)
    assert [row["answer_state"] for row in responses] == ["detected", "unresolved"]
