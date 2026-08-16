from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest


def _load(name: str):
    from api.v1._exampen_imports import load_exampen

    return load_exampen(name)


def _question(number: int, criterion_id: str = "criterion-1") -> dict[str, Any]:
    return {
        "question_id": f"EXAM-V16::Q{number}",
        "question_number": number,
        "question_text": f"प्रश्न {number}",
        "reference_solution": "Visible steps and diagram labels are checked.",
        "max_marks": 2,
        "marking_criteria": [{
            "criterion_id": criterion_id,
            "description": "Correct visible work",
            "max_marks": 2,
            "acceptable_evidence": "Equivalent correct Hindi wording or diagram work.",
        }],
    }


def _row(
    number: int,
    *,
    status: str = "attempted",
    pages: list[int] | None = None,
    marks: float = 1,
    needs_review: bool = False,
) -> dict[str, Any]:
    actual_pages = [1] if pages is None and status == "attempted" else pages or []
    return {
        "question_number": number,
        "attempt_status": status,
        "confidence": 0.93,
        "student_answer": "हिंदी उत्तर तथा चित्र" if actual_pages else "",
        "content_type": "MIXED",
        "source_pages": actual_pages,
        "criterion_marks": ([{
            "criterion_id": "criterion-1",
            "marks_awarded": marks,
            "confidence": 0.91,
            "rationale": "One required step is visible; another is missing.",
            "evidence": "Page shows the Hindi step and labelled diagram.",
            "credit_basis": "direct_evidence" if marks else "no_credit",
        }] if status == "attempted" else []),
        "total_score": marks if status == "attempted" else 0,
        "overall_feedback": "One step is missing." if status == "attempted" else "",
        "needs_review": needs_review,
        "review_reason": "Ownership needs one re-check." if needs_review else "",
    }


def _payload(*rows: dict[str, Any], accounted: bool = True) -> dict[str, Any]:
    return {
        "all_student_work_accounted": accounted,
        "questions": list(rows),
    }


def _usage() -> SimpleNamespace:
    return SimpleNamespace(
        model="gpt-5.1-2025-11-13",
        caller="pcr_eval_core",
        input_tokens=1_200,
        output_tokens=800,
        cache_read_tokens=500,
        total_tokens=2_000,
        estimated_cost_usd=0.02,
    )


class _Gate:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def call(self, model_id: str, prompt: str, caller_id: str, **kwargs: Any):
        self.calls.append({"model_id": model_id, "caller_id": caller_id, **kwargs})
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, SimpleNamespace):
            return value
        return SimpleNamespace(
            content=json.dumps(value, ensure_ascii=False),
            completion_status="completed",
            incomplete_reason="",
            usage=_usage(),
        )


def _run_args(module, gate: _Gate, questions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "gate": gate,
        "run_id": "RUN-V16",
        "submission_id": "SUB-V16",
        "exam_id": "EXAM-V16",
        "questions": questions,
        "page_count": 4,
        "paper_bytes": b"paper-pdf",
        "solution_bytes": b"solution-pdf",
        "student_content": [
            {"type": "input_text", "text": "physical page 1"},
            {"type": "input_image", "image_url": "data:image/jpeg;base64,AA=="},
            {"type": "input_text", "text": "physical page 4"},
        ],
        "paper_filename": "paper.pdf",
        "solution_filename": "solution.pdf",
        "model_id": "gpt-5.1-2025-11-13",
        "reasoning_effort": "medium",
        "temperature": 0.1,
        "paper_hash": "paper-hash",
        "solution_hash": "solution-hash",
    }


def test_v16_schema_is_strict_compact_and_has_no_coordinate_contract():
    whole = _load("pcr.services.whole_copy_grading")
    graph = _load("pcr.services.visual_evidence_graph")
    catalog = [{
        "question_number": 1,
        "grading_mode": "subjective",
        "max_marks": 2,
        "marking_criteria": _question(1)["marking_criteria"],
    }]
    schema = whole.whole_copy_schema(catalog)
    assert graph.strict_provider_schema(schema) == schema
    serialized = json.dumps(schema)
    assert "x_start" not in serialized
    assert "region_id" not in serialized
    assert "source_pages" in serialized
    assert whole.output_limit(catalog, reasoning_effort="medium") >= 24_000


def test_v16_normalizes_multi_page_hindi_diagram_without_semantic_rewrite():
    module = _load("pcr.services.full_document_grading")
    whole = _load("pcr.services.whole_copy_grading")
    questions = [_question(1)]
    normalized = whole.normalize_payload(_payload(_row(1, pages=[1, 4], marks=1)))
    grades, errors, review = module._validate_ledger(
        normalized,
        questions=questions,
        page_count=4,
    )
    assert errors == []
    assert review.required is False
    assert grades[0].student_answer == "हिंदी उत्तर तथा चित्र"
    assert grades[0].total_score == 1
    assert [region["page_number"] for region in grades[0].source_pages] == [1, 4]
    assert grades[0].criterion_marks[0]["evidence_region_ids"] == [
        "q1-legacy-page-1",
        "q1-legacy-page-4",
    ]


@pytest.mark.asyncio
async def test_v16_success_is_exactly_one_cached_whole_copy_call():
    module = _load("pcr.services.full_document_grading")
    gate = _Gate([_payload(_row(1, pages=[1, 4], marks=1))])
    payload, _, usage = await module._run_whole_copy_grading(
        **_run_args(module, gate, [_question(1)])
    )
    assert len(gate.calls) == 1
    call = gate.calls[0]
    assert call["metadata"]["provider_call_limit"] == 2
    assert call["metadata"]["recursive_splitting"] is False
    assert call["max_output_tokens"] >= 24_000
    assert call["prompt_cache_key"].startswith("pcr-v16-")
    assert call["metadata"]["pcr_stage"] == "whole_copy_visual_grading"
    text = json.dumps(call["responses_input"], ensure_ascii=False)
    assert "ORIGINAL QUESTION PAPER PDF" in text
    assert "TEACHER-UPLOADED SOLUTION" in text
    assert "physical page 1" in text and "physical page 4" in text
    assert "coordinates" in text  # only the instruction saying not to return them
    assert payload["document_review"]["all_student_work_accounted"] is True
    assert usage["total_tokens"] == 2_000


@pytest.mark.asyncio
async def test_v16_unresolved_uses_one_recovery_and_never_splits():
    module = _load("pcr.services.full_document_grading")
    gate = _Gate([
        _payload(_row(1, status="unresolved", needs_review=True)),
        _payload(_row(1, pages=[3], marks=2)),
    ])
    payload, _, usage = await module._run_whole_copy_grading(
        **_run_args(module, gate, [_question(1)])
    )
    assert len(gate.calls) == 2
    assert [call["metadata"]["provider_call_number"] for call in gate.calls] == [1, 2]
    assert all(call["metadata"]["recursive_splitting"] is False for call in gate.calls)
    assert gate.calls[0]["prompt_cache_key"] == gate.calls[1]["prompt_cache_key"]
    assert "ONE AND ONLY RECOVERY PASS" in json.dumps(
        gate.calls[1]["responses_input"], ensure_ascii=False
    )
    grades, _, review = module._validate_ledger(
        payload,
        questions=[_question(1)],
        page_count=4,
    )
    assert grades[0].total_score == 2
    assert review.required is False
    assert usage["stage_count"] == 2


@pytest.mark.asyncio
async def test_v16_output_exhaustion_fails_once_and_keeps_usage():
    module = _load("pcr.services.full_document_grading")
    response = SimpleNamespace(
        content="",
        completion_status="incomplete",
        incomplete_reason="max_output_tokens",
        usage=_usage(),
    )
    gate = _Gate([response])
    with pytest.raises(module.StructuredGradingOutputError) as raised:
        await module._run_whole_copy_grading(
            **_run_args(module, gate, [_question(1)])
        )
    assert len(gate.calls) == 1
    assert raised.value.token_usage["total_tokens"] == 2_000
    assert raised.value.structured_output_failure["incomplete_reason"] == "max_output_tokens"


@pytest.mark.asyncio
async def test_v16_orientation_labels_duplicate_views_without_requesting_geometry(monkeypatch):
    module = _load("pcr.services.full_document_grading")
    image = b"\xff\xd8\xfffake-jpeg"

    async def resolve(_ref, **_kwargs):
        import base64

        return base64.b64encode(image).decode("ascii")

    view = SimpleNamespace(
        is_original=False,
        rotation_degrees_clockwise=90,
        coordinate_frame={"original_width_px": 100, "original_height_px": 200},
        view_id="physical-page-1-rotation-90",
        alternate_of="physical-page-1-original",
        image_bytes=b"\x89PNG\r\n\x1a\nrotated",
    )
    monkeypatch.setattr(module, "_resolve_image_base64", resolve)
    monkeypatch.setattr(module, "build_orientation_views", lambda *_args, **_kwargs: (view,))
    content, _ = await module._student_copy_content(
        [{"page_number": 1, "raw_image_ref": "s3://answer/page1.jpg"}],
        orientation_recovery=True,
        coordinate_evidence=False,
    )
    text = " ".join(
        str(item.get("text") or "") for item in content if item.get("type") == "input_text"
    )
    assert "physical source page 1" in text
    assert "do not return coordinates" in text
