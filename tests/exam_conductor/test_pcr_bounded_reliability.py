"""Provider-free acceptance tests for bounded PCR evidence/grade materialization.

These tests intentionally exercise the server-owned contracts rather than a model
provider.  A provider fake is used only where the durable-run behaviour itself is
under test.  The important invariants are:

* evidence ownership is stable across page order, Hindi text, and diagrams;
* teacher annotations/uncertain authorship never become gradeable evidence;
* criterion rows are complete, cite owned region IDs, and determine the total;
* no-attempts are deterministic zeroes;
* incomplete output is a terminal, recorded failure for that bounded unit; and
* a completed materialization is idempotent and does not buy the same work again.

The bounded worker can evolve behind these contracts.  In particular, this file
does not make a network request and uses mongomock only for the two persistence
tests.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from mongomock_motor import AsyncMongoMockClient


def _load(name: str):
    from api.v1._exampen_imports import load_exampen

    return load_exampen(name)


def _question(number: int, *criterion_ids: str) -> dict[str, Any]:
    return {
        "question_id": f"EXAM-BOUND::Q{number}",
        "exam_id": "EXAM-BOUND",
        "question_number": number,
        "question_text": f"Question {number}",
        "reference_solution": "The reference solution is grading-only.",
        "max_marks": float(2 * len(criterion_ids)),
        "marking_criteria": [
            {
                "criterion_id": criterion_id,
                "description": f"Criterion {criterion_id}",
                "max_marks": 2,
                "acceptable_evidence": "Visible work satisfying this criterion.",
            }
            for criterion_id in criterion_ids
        ],
    }


def _region(
    region_id: str,
    *,
    page: int,
    sequence: int,
    continuation: str = "q1-continuation",
    kind: str = "handwriting",
    authorship: str = "student",
    content: str = "दिखाई दे रहा छात्र कार्य",
) -> dict[str, Any]:
    return {
        "region_id": region_id,
        "page_number": page,
        "x_start": 80,
        "y_start": 100 + sequence * 40,
        "x_end": 920,
        "y_end": 260 + sequence * 40,
        "evidence_kind": kind,
        "authorship": authorship,
        "continuation_group": continuation,
        "sequence": sequence,
        "observed_content": content,
        "diagram_components": ["label A", "arrow A to B"] if kind == "diagram" else [],
        "mapping_confidence": 0.96,
    }


def _mapping(*questions: dict[str, Any], complete: bool = True) -> dict[str, Any]:
    return {
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
        "document_review": {
            "all_student_work_accounted": complete,
            "teacher_annotations_present": True,
            "teacher_annotations_excluded": True,
            "warnings": [],
        },
        "questions": list(questions),
        "unassigned_student_regions": [],
    }


def _mapped(
    number: int,
    regions: list[dict[str, Any]],
    *,
    status: str = "attempted",
) -> dict[str, Any]:
    return {
        "question_number": number,
        "attempt_status": status,
        "content_type": "MIXED",
        "student_answer": "बहु-पृष्ठ उत्तर और चित्र",
        "evidence_regions": regions,
        "mapping_reason": "Question label, continuation and diagram semantics agree.",
        "needs_review": False,
        "review_reason": "",
    }


def _grade(number: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "question_number": number,
        "attempt_status": "attempted",
        "confidence": 0.96,
        "student_answer": "बहु-पृष्ठ उत्तर और चित्र",
        "content_type": "MIXED",
        "criterion_marks": rows,
        # Deliberately disagree: the server must derive the total from criteria.
        "total_score": 999,
        "overall_feedback": "Visible work was checked.",
        "needs_review": False,
        "review_reason": "",
    }


def _criterion(criterion_id: str, marks: float, *region_ids: str) -> dict[str, Any]:
    return {
        "criterion_id": criterion_id,
        "marks_awarded": marks,
        "confidence": 0.95,
        "rationale": "The cited work supports the awarded step marks.",
        "evidence": "Visible student work in the cited region.",
        "evidence_region_ids": list(region_ids),
        "credit_basis": "direct_evidence" if marks else "no_credit",
    }


def test_out_of_order_multipage_hindi_and_diagram_regions_form_one_answer():
    graph = _load("pcr.services.visual_evidence_graph")
    q1 = _mapped(
        1,
        [
            _region(
                "q1-page4", page=4, sequence=2, content="अगला चरण: निष्कर्ष लिखिए"
            ),
            _region(
                "q1-page1", page=1, sequence=1, kind="diagram", content="चित्र और लेबल"
            ),
        ],
    )

    result = graph.validate_mapping_payload(
        _mapping(q1), question_numbers=[1], page_count=4
    )

    mapped = result.questions[1]
    assert mapped["attempt_status"] == "attempted"
    assert [r["region_id"] for r in mapped["evidence_regions"]] == [
        "q1-page1",
        "q1-page4",
    ]
    assert {r["page_number"] for r in mapped["evidence_regions"]} == {1, 4}
    assert mapped["evidence_regions"][0]["diagram_components"]
    assert "अगला चरण" in mapped["evidence_regions"][1]["evidence"]
    assert result.errors == []


def test_uncertain_teacher_annotation_cannot_be_used_as_student_evidence():
    graph = _load("pcr.services.visual_evidence_graph")
    result = graph.validate_mapping_payload(
        _mapping(
            _mapped(
                1,
                [
                    _region("q1-tick", page=1, sequence=1, authorship="uncertain", content="teacher tick"),
                ],
            )
        ),
        question_numbers=[1],
        page_count=1,
    )

    mapped = result.questions[1]
    assert mapped["attempt_status"] == "unresolved"
    assert mapped["needs_review"] is True
    assert "authorship" in mapped["review_reason"].lower()


def test_server_derives_deterministic_total_and_requires_owned_evidence_for_marks():
    grading = _load("pcr.services.full_document_grading")
    graph = _load("pcr.services.visual_evidence_graph")
    question = _question(1, "method", "answer")
    mapping = graph.validate_mapping_payload(
        _mapping(
            _mapped(
                1,
                [
                    _region("q1-method", page=1, sequence=1),
                    _region("q1-answer", page=1, sequence=2),
                ],
            )
        ),
        question_numbers=[1],
        page_count=1,
    )
    merged = graph.merge_mapping_and_grading(
        mapping,
        {"questions": [_grade(1, [_criterion("method", 1, "q1-method"), _criterion("answer", 2, "q1-answer")])]},
    )

    grades, errors, review = grading._validate_ledger(
        merged, questions=[question], page_count=1
    )
    assert errors == []
    assert review.required is False
    assert grades[0].total_score == 3

    # A positive criterion score citing another question's region must block,
    # even if the model reports full marks.
    bad = graph.merge_mapping_and_grading(
        mapping,
        {"questions": [_grade(1, [_criterion("method", 2, "q2-owned"), _criterion("answer", 2, "q1-answer")])]},
    )
    bad_grades, _, _ = grading._validate_ledger(bad, questions=[question], page_count=1)
    assert bad_grades[0].attempt_status == "unresolved"
    assert "another question" in bad_grades[0].review_reason


def test_full_marks_cannot_pass_when_any_locked_criterion_is_missing():
    grading = _load("pcr.services.full_document_grading")
    graph = _load("pcr.services.visual_evidence_graph")
    question = _question(1, "method", "answer")
    mapping = graph.validate_mapping_payload(
        _mapping(_mapped(1, [_region("q1-work", page=1, sequence=1)])),
        question_numbers=[1],
        page_count=1,
    )
    partial_payload = graph.merge_mapping_and_grading(
        mapping,
        {"questions": [_grade(1, [_criterion("method", 2, "q1-work")])]},
    )
    grades, _, _ = grading._validate_ledger(
        partial_payload, questions=[question], page_count=1
    )
    assert grades[0].attempt_status == "unresolved"
    assert "marking plan" in grades[0].review_reason


def test_normalized_regions_retain_explicit_original_page_frame_metadata():
    """Coordinates must remain in the exact stored page frame used by grading.

    A rotated mapper crop is not interchangeable with an original stored page.
    The frame identity/geometry therefore travels with the region; silently
    converting coordinates (or dropping the frame) is an invalid materialization.
    """
    grading = _load("pcr.services.full_document_grading")
    question = _question(1, "c1")
    frame = {
        "id": "SUB-IDEMPOTENT:page-1:original",
        "kind": "original_page",
        "width_px": 1200,
        "height_px": 1600,
        "rotation_degrees": 90,
    }
    region = _region("q1-frame", page=1, sequence=1)
    region["coordinate_space"] = "normalized_1000"
    region["coordinate_frame"] = frame
    item = _grade(1, [_criterion("c1", 2, "q1-frame")])
    item["source_pages"] = [region]

    grade = grading._validate_question_grade(
        item, question=question, question_number=1, page_count=1
    )
    assert grade.attempt_status == "attempted"
    assert grade.source_pages[0]["coordinate_space"] == "normalized_1000"
    assert grade.source_pages[0]["coordinate_frame"] == frame
    assert (grade.source_pages[0]["x_start"], grade.source_pages[0]["y_start"]) == (
        region["x_start"],
        region["y_start"],
    )


def test_v14_mapping_schema_is_provider_strict_and_frame_identity_is_server_owned():
    """Reject malformed provider schemas locally instead of spending a request."""

    graph = _load("pcr.services.visual_evidence_graph")
    schema = graph.compact_mapping_schema([_question(1, "c1")])
    region_schema = schema["properties"]["questions"]["items"]["properties"][
        "evidence_regions"
    ]["items"]

    assert region_schema["additionalProperties"] is False
    assert set(region_schema["required"]) == set(region_schema["properties"])
    assert "coordinate_frame" not in region_schema["properties"]

    with pytest.raises(ValueError, match="additionalProperties=false"):
        graph.strict_provider_schema(
            {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            }
        )


def test_not_attempted_is_a_deterministic_zero_after_complete_copy_review():
    grading = _load("pcr.services.full_document_grading")
    graph = _load("pcr.services.visual_evidence_graph")
    question = _question(1, "c1")
    mapping = graph.validate_mapping_payload(
        _mapping(_mapped(1, [], status="not_attempted")),
        question_numbers=[1],
        page_count=2,
    )
    merged = graph.merge_mapping_and_grading(
        mapping,
        {"questions": [{"question_number": 1, "attempt_status": "not_attempted", "total_score": 0, "criterion_marks": []}]},
    )
    grades, errors, review = grading._validate_ledger(
        merged, questions=[question], page_count=2
    )
    assert errors == []
    assert review.required is False
    assert grades[0].attempt_status == "not_attempted"
    assert grades[0].total_score == 0
    assert all(row["marks_awarded"] == 0 for row in grades[0].criterion_marks)


class _Gate:
    def __init__(self, response: Any):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def call(self, model_id: str, prompt: str, caller_id: str, **kwargs: Any):
        self.calls.append({"model_id": model_id, "prompt": prompt, "caller_id": caller_id, **kwargs})
        return self.response


class _SequenceGate:
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def call(self, model_id: str, prompt: str, caller_id: str, **kwargs: Any):
        self.calls.append({"model_id": model_id, "prompt": prompt, "caller_id": caller_id, **kwargs})
        if not self.responses:
            raise AssertionError("A completed bounded run made an unexpected provider call")
        return self.responses.pop(0)


def _response_payload(payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        content=json.dumps(payload, ensure_ascii=False),
        completion_status="completed",
        incomplete_reason="",
        usage=SimpleNamespace(model="gpt-5.1-2025-11-13", total_tokens=100),
    )


async def _seed_idempotency_submission(db) -> None:
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "SUB-IDEMPOTENT",
            "exam_id": "EXAM-BOUND",
            "student_id": "STU-BOUND",
            "source": "camera",
            "content_hash": "copy-hash",
        }
    )
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "EXAM-BOUND",
            "exam_type": "pcr",
            "prepared_document_id": "DOC-BOUND",
            "paper_version_id": "PAPER-BOUND",
            "paper_content_hash": "paper-hash",
        }
    )
    await db["exampen_paper_versions"].insert_one(
        {"paper_version_id": "PAPER-BOUND", "document_id": "DOC-BOUND"}
    )
    await db["documents"].insert_one(
        {
            "document_id": "DOC-BOUND",
            "file_path": "paper.pdf",
            "filename": "paper.pdf",
            "answer_sheet_path": "solution.pdf",
            "answer_sheet_filename": "solution.pdf",
        }
    )
    await db["evalpen_questions"].insert_many(
        [_question(1, "c1"), _question(2, "c2")]
    )
    await db["evalpen_answer_pages"].insert_one(
        {
            "page_id": "PAGE-BOUND",
            "submission_id": "SUB-IDEMPOTENT",
            "page_number": 1,
            "raw_image_ref": "page.jpg",
            "content_hash": "page-hash",
        }
    )


@pytest.mark.asyncio
async def test_completed_materialization_is_idempotent_and_does_not_duplicate_work(monkeypatch):
    module = _load("pcr.services.full_document_grading")
    graph = _load("pcr.services.visual_evidence_graph")
    db = AsyncMongoMockClient()["pcr_bounded_idempotency"]
    await _seed_idempotency_submission(db)
    monkeypatch.setattr(module, "_read_canonical_file", lambda *args, **kwargs: _value(b"pdf"))
    monkeypatch.setattr(
        module,
        "_student_copy_content",
        lambda pages, **_kwargs: _value(([{"type": "input_text", "text": "copy"}], 20)),
    )
    q2_region = _region("q2-r1", page=1, sequence=1, continuation="q2")
    # Distinct physical page area: ownership validation must not mistake two
    # compact test regions for one overlapping answer.
    q2_region.update(y_start=600, y_end=800)
    mapping = _mapping(
        _mapped(1, [_region("q1-r1", page=1, sequence=1)]),
        _mapped(2, [q2_region]),
    )
    grading = {
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
        "questions": [
            _grade(1, [_criterion("c1", 2, "q1-r1")]),
            _grade(2, [_criterion("c2", 2, "q2-r1")]),
        ]
    }
    gate = _SequenceGate([_response_payload(mapping), _response_payload(grading)])
    # The production model guard is intentionally retained; the gate is still a
    # local fake and no provider call is made.
    service = module.FullDocumentGradingService(
        db, gate, model_id="gpt-5.1-2025-11-13"
    )

    first = await service.grade_submission("SUB-IDEMPOTENT")
    second = await service.grade_submission("SUB-IDEMPOTENT")

    assert first.status == "completed"
    assert second.status == "completed"
    assert first.run_id == second.run_id
    assert len(gate.calls) == 2
    assert await db["evalpen_evaluations"].count_documents({"student_id": "STU-BOUND"}) == 2
    assert await db["evalpen_detected_responses"].count_documents({"submission_id": "SUB-IDEMPOTENT", "superseded_at": {"$exists": False}}) == 2


async def _value(value: Any) -> Any:
    return value


@pytest.mark.asyncio
async def test_incomplete_output_is_recorded_without_identical_whole_copy_retry():
    """A max-output failure must not be retried with the same oversized request."""
    module = _load("pcr.services.full_document_grading")
    db = AsyncMongoMockClient()["pcr_bounded_incomplete"]
    await db["evalpen_document_grading_runs"].insert_one(
        {
            "run_id": "RUN-INCOMPLETE",
            "submission_id": "SUB-INCOMPLETE",
            "status": "generating",
            "generation_lease_token": "LEASE-INCOMPLETE",
        }
    )
    gate = _Gate(
        SimpleNamespace(
            content="",
            completion_status="incomplete",
            incomplete_reason="max_output_tokens",
            usage=SimpleNamespace(model="test", total_tokens=100),
        )
    )
    with pytest.raises(module.StructuredGradingOutputError):
        await module._run_evidence_first_grading(
            db=db,
            gate=gate,
            existing_run=await db["evalpen_document_grading_runs"].find_one({"run_id": "RUN-INCOMPLETE"}),
            generation_lease_token="LEASE-INCOMPLETE",
            run_id="RUN-INCOMPLETE",
            submission_id="SUB-INCOMPLETE",
            exam_id="EXAM-BOUND",
            questions=[_question(1, "c1")],
            page_count=1,
            paper_bytes=b"paper",
            solution_bytes=b"solution",
            student_content=[{"type": "input_text", "text": "copy"}],
            paper_filename="paper.pdf",
            solution_filename="solution.pdf",
            model_id="test-model",
            reasoning_effort="low",
            temperature=0,
            paper_hash="paper-hash",
            solution_hash="solution-hash",
        )
    assert len(gate.calls) == 1
    assert gate.calls[0]["metadata"]["pcr_stage"] == "student_evidence_mapping"
    assert gate.calls[0]["max_output_tokens"] == 10_000


@pytest.mark.asyncio
async def test_v14_max_output_failure_splits_page_unit_and_checkpoints_each_shard():
    """A bounded mapper retries smaller units, never the same whole-copy input."""
    module = _load("pcr.services.full_document_grading")
    db = AsyncMongoMockClient()["pcr_bounded_shards"]
    await db["evalpen_document_grading_runs"].insert_one(
        {
            "run_id": "RUN-SHARDS",
            "submission_id": "SUB-SHARDS",
            "status": "generating",
            "generation_lease_token": "LEASE-SHARDS",
        }
    )
    q1 = _question(1, "c1")
    oversized = SimpleNamespace(
        content="",
        completion_status="incomplete",
        incomplete_reason="max_output_tokens",
        usage=SimpleNamespace(model="gpt-5.1-2025-11-13", total_tokens=100),
    )
    map_page_1 = _mapping(
        _mapped(1, [_region("q1-page1", page=1, sequence=1)]),
    )
    map_page_2 = _mapping(
        _mapped(1, [_region("q1-page2", page=2, sequence=2)]),
    )
    grade = {
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
        "questions": [
            _grade(1, [_criterion("c1", 2, "q1-page1", "q1-page2")])
        ],
    }
    audit = {
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
        "questions": [{
            "question_number": 1,
            "criterion_marks": [{
                "criterion_id": "c1",
                "marks_supported": 2,
                "evidence_region_ids": ["q1-page1", "q1-page2"],
                "rationale": "Independent visible evidence supports the criterion.",
            }],
        }],
    }
    gate = _SequenceGate(
        [oversized, _response_payload(map_page_1), _response_payload(map_page_2), _response_payload(grade), _response_payload(audit)]
    )
    merged, _raw, usage = await module._run_evidence_first_grading(
        db=db,
        gate=gate,
        existing_run=await db["evalpen_document_grading_runs"].find_one({"run_id": "RUN-SHARDS"}),
        generation_lease_token="LEASE-SHARDS",
        run_id="RUN-SHARDS",
        submission_id="SUB-SHARDS",
        exam_id="EXAM-BOUND",
        questions=[q1],
        page_count=2,
        paper_bytes=b"paper",
        solution_bytes=b"solution",
        student_content=[
            {"type": "input_text", "text": "Student answer-copy page 1, original"},
            {"type": "input_text", "text": "page one"},
            {"type": "input_text", "text": "Student answer-copy page 2, original"},
            {"type": "input_text", "text": "page two"},
        ],
        paper_filename="paper.pdf",
        solution_filename="solution.pdf",
        model_id="gpt-5.1-2025-11-13",
        reasoning_effort="low",
        temperature=0,
        paper_hash="paper-hash",
        solution_hash="solution-hash",
        pipeline_version=module._V14_PROMPT_VERSION,
    )

    assert merged["questions"][0]["question_number"] == 1
    assert usage["stage_count"] == 4
    assert len(gate.calls) == 5
    assert gate.calls[0]["metadata"]["unit_pages"] == [1, 2]
    assert gate.calls[0]["max_output_tokens"] == 4_000
    assert [call["metadata"]["unit_pages"] for call in gate.calls[1:3]] == [[1], [2]]
    # No retry has the same unit ID or the same page set as the failed request.
    assert gate.calls[0]["metadata"]["unit_id"] not in {
        gate.calls[1]["metadata"]["unit_id"],
        gate.calls[2]["metadata"]["unit_id"],
    }
    saved = await db["evalpen_document_grading_runs"].find_one({"run_id": "RUN-SHARDS"})
    assert {row["unit_id"] for row in saved["evidence_mapping_units"]} == {
        "pages-1-a",
        "pages-1-b",
    }


@pytest.mark.asyncio
async def test_v14_restart_consumes_mapping_split_children_without_parent_retry():
    """A persisted split manifest is the restart checkpoint for its parent unit."""
    module = _load("pcr.services.full_document_grading")
    db = AsyncMongoMockClient()["pcr_bounded_mapping_restart"]
    q1 = _question(1, "c1")
    map_page_1 = _mapping(_mapped(1, [_region("restart-q1-p1", page=1, sequence=1)]))
    map_page_2 = _mapping(_mapped(1, [_region("restart-q1-p2", page=2, sequence=2)]))
    compact_1 = module._compact_payload_from_response(map_page_1)
    compact_2 = module._compact_payload_from_response(map_page_2)
    usage = {"model": "gpt-5.1-2025-11-13", "total_tokens": 1}
    await db["evalpen_document_grading_runs"].insert_one(
        {
            "run_id": "RUN-MAP-RESTART",
            "submission_id": "SUB-MAP-RESTART",
            "status": "generating",
            "generation_lease_token": "LEASE-MAP-RESTART",
            "evidence_mapping_units": [
                {"unit_id": "pages-1-a", "pages": [1], "question_numbers": [1], "payload": compact_1, "usage": usage},
                {"unit_id": "pages-1-b", "pages": [2], "question_numbers": [1], "payload": compact_2, "usage": usage},
            ],
            "evidence_mapping_split_manifests": [
                {
                    "unit_id": "pages-1",
                    "status": "split",
                    "pages": [1, 2],
                    "children": [
                        {"unit_id": "pages-1-a", "pages": [1], "question_numbers": [1]},
                        {"unit_id": "pages-1-b", "pages": [2], "question_numbers": [1]},
                    ],
                }
            ],
            "evidence_grading_batches": [{
                "batch_id": "batch-1", "payload": {
                    "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
                    "questions": [_grade(1, [_criterion("c1", 2, "restart-q1-p1", "restart-q1-p2")])],
                }, "usage": usage,
            }],
            "evidence_verification_batches": [{
                "batch_id": "audit-1", "payload": {
                    "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
                    "questions": [{
                        "question_number": 1,
                        "criterion_marks": [{
                            "criterion_id": "c1", "marks_supported": 2,
                            "evidence_region_ids": ["restart-q1-p1", "restart-q1-p2"],
                            "rationale": "Independent visible evidence supports the criterion.",
                        }],
                    }],
                }, "usage": usage,
            }],
        }
    )
    gate = _SequenceGate([])
    merged, _raw, _usage = await module._run_evidence_first_grading(
        db=db,
        gate=gate,
        existing_run=await db["evalpen_document_grading_runs"].find_one({"run_id": "RUN-MAP-RESTART"}),
        generation_lease_token="LEASE-MAP-RESTART",
        run_id="RUN-MAP-RESTART",
        submission_id="SUB-MAP-RESTART",
        exam_id="EXAM-BOUND",
        questions=[q1],
        page_count=2,
        paper_bytes=b"paper",
        solution_bytes=b"solution",
        student_content=[{"type": "input_text", "text": "copy"}],
        paper_filename="paper.pdf",
        solution_filename="solution.pdf",
        model_id="gpt-5.1-2025-11-13",
        reasoning_effort="low",
        temperature=0,
        paper_hash="paper-hash",
        solution_hash="solution-hash",
        pipeline_version=module._V14_PROMPT_VERSION,
    )
    assert merged["questions"][0]["question_number"] == 1
    assert len(gate.calls) == 0


@pytest.mark.asyncio
async def test_v14_restart_consumes_grading_split_children_without_parent_retry():
    """Completed grading children must make the original batch request unnecessary."""
    module = _load("pcr.services.full_document_grading")
    db = AsyncMongoMockClient()["pcr_bounded_grading_restart"]
    q1, q2 = _question(1, "c1"), _question(2, "c2")
    r1 = _region("restart-g-q1", page=1, sequence=1)
    r2 = _region("restart-g-q2", page=1, sequence=1, continuation="q2")
    r2.update(y_start=600, y_end=800)
    compact = module._compact_payload_from_response(
        _mapping(_mapped(1, [r1]), _mapped(2, [r2]))
    )
    payload_1 = {
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
        "questions": [_grade(1, [_criterion("c1", 2, "restart-g-q1")])],
    }
    payload_2 = {
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
        "questions": [_grade(2, [_criterion("c2", 2, "restart-g-q2")])],
    }
    usage = {"model": "gpt-5.1-2025-11-13", "total_tokens": 1}
    await db["evalpen_document_grading_runs"].insert_one(
        {
            "run_id": "RUN-GRADE-RESTART",
            "submission_id": "SUB-GRADE-RESTART",
            "status": "generating",
            "generation_lease_token": "LEASE-GRADE-RESTART",
            "evidence_mapping_units": [
                {"unit_id": "pages-1", "pages": [1, 2], "question_numbers": [1, 2], "payload": compact, "usage": usage}
            ],
            "evidence_grading_batches": [
                {"batch_id": "batch-1-a", "question_numbers": [1], "payload": payload_1, "usage": usage},
                {"batch_id": "batch-1-b", "question_numbers": [2], "payload": payload_2, "usage": usage},
            ],
            "evidence_grading_split_manifests": [
                {
                    "batch_id": "batch-1",
                    "status": "split",
                    "question_numbers": [1, 2],
                    "children": [
                        {"batch_id": "batch-1-a", "question_numbers": [1]},
                        {"batch_id": "batch-1-b", "question_numbers": [2]},
                    ],
                }
            ],
            "evidence_verification_batches": [{
                "batch_id": "audit-1", "payload": {
                    "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
                    "questions": [
                        {"question_number": 1, "criterion_marks": [{
                            "criterion_id": "c1", "marks_supported": 2,
                            "evidence_region_ids": ["restart-g-q1"],
                            "rationale": "Independent visible evidence supports the criterion.",
                        }]},
                        {"question_number": 2, "criterion_marks": [{
                            "criterion_id": "c2", "marks_supported": 2,
                            "evidence_region_ids": ["restart-g-q2"],
                            "rationale": "Independent visible evidence supports the criterion.",
                        }]},
                    ],
                }, "usage": usage,
            }],
        }
    )
    gate = _SequenceGate([])
    merged, _raw, _usage = await module._run_evidence_first_grading(
        db=db,
        gate=gate,
        existing_run=await db["evalpen_document_grading_runs"].find_one({"run_id": "RUN-GRADE-RESTART"}),
        generation_lease_token="LEASE-GRADE-RESTART",
        run_id="RUN-GRADE-RESTART",
        submission_id="SUB-GRADE-RESTART",
        exam_id="EXAM-BOUND",
        questions=[q1, q2],
        page_count=2,
        paper_bytes=b"paper",
        solution_bytes=b"solution",
        student_content=[{"type": "input_text", "text": "copy"}],
        paper_filename="paper.pdf",
        solution_filename="solution.pdf",
        model_id="gpt-5.1-2025-11-13",
        reasoning_effort="low",
        temperature=0,
        paper_hash="paper-hash",
        solution_hash="solution-hash",
        pipeline_version=module._V14_PROMPT_VERSION,
    )
    assert gate.calls == []
    assert {item["question_number"] for item in merged["questions"]} == {1, 2}


def test_compact_mapping_unit_rejects_evidence_from_another_page():
    module = _load("pcr.services.full_document_grading")
    payload = {
        "questions": [
            {"question_number": 1, "evidence_regions": [{"page_number": 2}]}
        ]
    }
    with pytest.raises(module.StructuredGradingOutputError, match="outside its supplied pages"):
        module._validate_compact_unit_pages(payload, [1])


def test_detailed_single_question_gets_criterion_aware_output_budget():
    """Reasoning plus five required criterion rows must not be forced into 4k."""

    module = _load("pcr.services.full_document_grading")
    detailed = _question(1, "c1", "c2", "c3", "c4", "c5")

    limit = module._bounded_grading_output_limit(
        [detailed],
        reasoning_effort="medium",
    )

    assert limit == 9_000
    assert limit > module._V14_MAX_OUTPUT_TOKENS
    assert limit <= module._BOUNDED_GRADE_MAX_OUTPUT_TOKENS


def test_compact_mapping_normalization_returns_one_terminal_state_per_catalog_question():
    """A sparse unit response cannot leave catalog questions as implicit blanks."""
    graph = _load("pcr.services.visual_evidence_graph")
    payload = {
        "mapping_version": "pcr-compact-evidence-map-v1",
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
        "all_student_work_accounted": False,
        "questions": [
            {
                "question_number": 1,
                "attempt_status": "attempted",
                "content_type": "TEXT_ONLY",
                "evidence_regions": [_region("q1-terminal", page=1, sequence=1)],
            }
        ],
        "unassigned_student_regions": [],
    }

    result = graph.normalize_compact_mapping_payload(
        payload, question_numbers=[1, 2, 3], page_count=1
    )

    assert set(result.questions) == {1, 2, 3}
    assert {
        item["attempt_status"] for item in result.questions.values()
    } <= {"attempted", "not_attempted", "unresolved"}
    assert result.questions[1]["attempt_status"] == "attempted"
    assert result.questions[2]["attempt_status"] == "unresolved"
    assert result.questions[3]["attempt_status"] == "unresolved"
    assert result.document_review["all_student_work_accounted"] is False
    assert result.errors


def test_missing_mapper_row_with_visible_unassigned_work_is_never_blank_or_zero():
    """Unassigned evidence forces review; it cannot become an omitted zero."""
    graph = _load("pcr.services.visual_evidence_graph")
    unassigned = _region(
        "visible-unassigned",
        page=1,
        sequence=1,
        content="Visible student work with no safe question owner",
    )
    payload = {
        "mapping_version": "pcr-compact-evidence-map-v1",
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
        "all_student_work_accounted": False,
        "questions": [],
        "unassigned_student_regions": [unassigned],
    }

    result = graph.normalize_compact_mapping_payload(
        payload, question_numbers=[1], page_count=1
    )

    assert result.questions[1]["attempt_status"] == "unresolved"
    assert result.unassigned_regions[0]["region_id"] == "visible-unassigned"
    assert result.document_review["all_student_work_accounted"] is False
    assert result.errors
    assert "unassigned" in " ".join(result.document_review["warnings"]).lower()


def test_complete_mapping_without_unassigned_work_materializes_omitted_questions_as_not_attempted():
    """Only a complete, globally accounted map can prove a question absent."""
    graph = _load("pcr.services.visual_evidence_graph")
    payload = {
        "mapping_version": "pcr-compact-evidence-map-v1",
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
        "all_student_work_accounted": True,
        "questions": [
            {
                "question_number": 1,
                "attempt_status": "attempted",
                "content_type": "TEXT_ONLY",
                "evidence_regions": [_region("q1-present", page=1, sequence=1)],
            }
        ],
        "unassigned_student_regions": [],
    }

    result = graph.merge_compact_mapping_payloads(
        [payload], question_numbers=[1, 2, 3], page_count=1
    )

    assert result.questions[1]["attempt_status"] == "attempted"
    assert result.questions[2]["attempt_status"] == "not_attempted"
    assert result.questions[3]["attempt_status"] == "not_attempted"
    assert all(not result.questions[number]["evidence_regions"] for number in (2, 3))
    assert result.document_review["all_student_work_accounted"] is True
    assert result.errors == []


def test_v15_orientation_manifest_and_region_conversion_use_original_normalized_frame():
    """A mapper region from a rotated view is invertible in the stored-page frame."""
    from io import BytesIO

    from PIL import Image

    orientation = _load("pcr.services.orientation_views")
    output = BytesIO()
    Image.new("RGB", (120, 240), "white").save(output, format="PNG")
    views = orientation.build_orientation_views(
        output.getvalue(),
        physical_page_number=7,
        width_px=120,
        height_px=240,
        detector=lambda *_args, **_kwargs: (True, {"method": "test", "sideways": True}),
    )

    assert [view.rotation_degrees_clockwise for view in views] == [0, 90, 270]
    assert all(view.coordinate_frame["kind"] == "original_stored_page" for view in views)
    assert all(view.coordinate_frame["coordinate_space"] == "normalized_1000" for view in views)
    assert views[1].coordinate_frame["original_width_px"] == 120
    assert views[1].coordinate_frame["original_height_px"] == 240

    original = orientation.view_region_to_original(
        {
            "region_id": "rotated-region",
            "x_start": 100,
            "y_start": 200,
            "x_end": 400,
            "y_end": 600,
        },
        rotation_degrees_clockwise=90,
    )
    assert (original["x_start"], original["y_start"], original["x_end"], original["y_end"]) == (
        200.0,
        600.0,
        600.0,
        900.0,
    )
    assert original["coordinate_transform"] == {
        "type": "normalized_1000_view_to_original",
        "rotation_degrees_clockwise": 90,
        "invertible": True,
    }

    graph = _load("pcr.services.visual_evidence_graph")
    schema = graph.compact_mapping_schema(
        [{"question_number": 1}],
        prompt_version=graph.V15_PROMPT_VERSION,
    )
    region_properties = schema["properties"]["questions"]["items"]["properties"][
        "evidence_regions"
    ]["items"]["properties"]
    assert "source_rotation_degrees_clockwise" in region_properties
    assert "mapping_confidence" in region_properties
    recovery_schema = graph.compact_mapping_schema(
        [{"question_number": 1}],
        prompt_version=graph.V15_PROMPT_VERSION,
        recovery_pass=True,
    )
    recovery_region = recovery_schema["properties"]["questions"]["items"][
        "properties"
    ]["evidence_regions"]["items"]
    assert "supersedes_region_ids" in recovery_region["properties"]
    assert set(recovery_region["required"]) == set(recovery_region["properties"])

    grading = _load("pcr.services.full_document_grading")
    provider_region = _region("provider-rotated", page=1, sequence=1)
    provider_region.update(
        x_start=100,
        y_start=200,
        x_end=400,
        y_end=600,
        source_rotation_degrees_clockwise=90,
    )
    compact = grading._compact_payload_from_response(
        _mapping(_mapped(1, [provider_region])),
        prompt_version=grading._V15_PROMPT_VERSION,
        student_content=[{
            "type": "input_text",
            "text": (
                "Student answer-copy page 1, physical-page orientation view; "
                "source_rotation_degrees_clockwise=90; "
                "view_id=physical-page-1-rotation-90; "
                "alternate_of=physical-page-1-original; "
                "original_width_px=120; original_height_px=240."
            ),
        }],
    )
    stored = compact["questions"][0]["evidence_regions"][0]
    assert (stored["x_start"], stored["y_start"], stored["x_end"], stored["y_end"]) == (
        200.0,
        600.0,
        600.0,
        900.0,
    )
    assert stored["coordinate_frame"] == {
        "id": "physical-page-1-original-frame",
        "kind": "original_stored_page",
        "coordinate_space": "normalized_1000",
        "width_px": 120,
        "height_px": 240,
        "source_rotation_degrees_clockwise": 90,
        "invertible": True,
    }


def test_one_targeted_mapping_recovery_assigns_unassigned_work_without_regrading_completed_question():
    """Recovery coverage is narrow: it fills the orphan and preserves Q1 once."""
    graph = _load("pcr.services.visual_evidence_graph")
    q1_region = _region("completed-q1", page=1, sequence=1, content="already graded Q1")
    orphan = _region(
        "orphan-q2",
        page=2,
        sequence=1,
        continuation="orphan",
        content="visible Q2 work awaiting targeted recovery",
    )
    initial = {
        "mapping_version": "pcr-compact-evidence-map-v1",
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
        "all_student_work_accounted": False,
        "questions": [
            {
                "question_number": 1,
                "attempt_status": "attempted",
                "content_type": "TEXT_ONLY",
                "evidence_regions": [q1_region],
            }
        ],
        "unassigned_student_regions": [orphan],
    }
    recovery = {
        "mapping_version": "pcr-compact-evidence-map-v1",
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
        "all_student_work_accounted": True,
        "questions": [
            {
                "question_number": 2,
                "attempt_status": "attempted",
                "content_type": "TEXT_ONLY",
                "evidence_regions": [dict(orphan)],
            }
        ],
        "unassigned_student_regions": [],
    }

    result = graph.reconcile_compact_mapping_recovery(
        [initial], [recovery], question_numbers=[1, 2], page_count=2
    )

    assert result.unassigned_regions == []
    assert result.questions[1]["attempt_status"] == "attempted"
    assert result.questions[2]["attempt_status"] == "attempted"
    assert [r["region_id"] for r in result.questions[1]["evidence_regions"]] == ["completed-q1"]
    assert [r["region_id"] for r in result.questions[2]["evidence_regions"]] == ["orphan-q2"]
    assert len([r for q in result.questions.values() for r in q["evidence_regions"] if r["region_id"] == "completed-q1"]) == 1


def test_v15_recovery_can_traceably_correct_collapsed_question_ownership():
    """A second pass can replace a wrong owner, but only by exact ID and overlap."""

    graph = _load("pcr.services.visual_evidence_graph")
    wrong = _region("wrong-q9-owner", page=1, sequence=1, continuation="q9")
    initial = {
        "mapping_version": "pcr-compact-evidence-map-v2",
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
        "all_student_work_accounted": True,
        "questions": [{
            "question_number": 9,
            "attempt_status": "attempted",
            "content_type": "TEXT_ONLY",
            "evidence_regions": [wrong],
        }],
        "unassigned_student_regions": [],
    }
    corrected = dict(wrong)
    corrected.update(
        region_id="correct-q1-owner",
        continuation_group="q1",
        supersedes_region_ids=["wrong-q9-owner"],
    )
    recovery = {
        "mapping_version": "pcr-compact-evidence-map-v2",
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
        "all_student_work_accounted": True,
        "questions": [{
            "question_number": 1,
            "attempt_status": "attempted",
            "content_type": "TEXT_ONLY",
            "evidence_regions": [corrected],
        }],
        "unassigned_student_regions": [],
    }

    result = graph.reconcile_compact_mapping_recovery(
        [initial],
        [recovery],
        question_numbers=[1, 9],
        page_count=1,
        recovered_page_numbers=[1],
    )

    assert result.questions[1]["attempt_status"] == "attempted"
    assert result.questions[9]["attempt_status"] == "not_attempted"
    assert [
        region["region_id"] for region in result.questions[1]["evidence_regions"]
    ] == ["correct-q1-owner"]
    assert result.unassigned_regions == []
    assert result.errors == []


@pytest.mark.asyncio
async def test_v15_worker_runs_one_checkpointed_recovery_before_one_grading_pass():
    """Visible orphan work is reassociated once, then every attempted answer is graded once."""

    module = _load("pcr.services.full_document_grading")
    db = AsyncMongoMockClient()["pcr_v15_recovery_worker"]
    await db["evalpen_document_grading_runs"].insert_one({
        "run_id": "RUN-V15-RECOVERY",
        "submission_id": "SUB-V15-RECOVERY",
        "status": "generating",
        "generation_lease_token": "LEASE-V15-RECOVERY",
    })
    q1, q2 = _question(1, "c1"), _question(2, "c2")
    q1_region = _region("v15-q1", page=1, sequence=1)
    q1_region["source_rotation_degrees_clockwise"] = 0
    orphan = _region(
        "v15-q2-orphan",
        page=1,
        sequence=1,
        continuation="q2",
    )
    orphan.update(y_start=600, y_end=800, source_rotation_degrees_clockwise=0)
    initial = _mapping(_mapped(1, [q1_region]))
    initial["mapping_version"] = "pcr-compact-evidence-map-v2"
    # The first mapper incorrectly treats Q2 as absent; no unassigned-region
    # signal is available. V15 must still run the bounded absence verifier.
    initial["unassigned_student_regions"] = []
    recovered = _mapping(_mapped(2, [dict(orphan)]))
    recovered["mapping_version"] = "pcr-compact-evidence-map-v2"
    grading_payload = {
        "evidence_graph_version": "pcr-multimodal-evidence-graph-v3",
        "questions": [
            _grade(1, [_criterion("c1", 1, "v15-q1")]),
            _grade(2, [_criterion("c2", 1, "v15-q2-orphan")]),
        ],
    }
    gate = _SequenceGate([
        _response_payload(initial),
        _response_payload(recovered),
        _response_payload(grading_payload),
    ])
    kwargs = dict(
        db=db,
        generation_lease_token="LEASE-V15-RECOVERY",
        run_id="RUN-V15-RECOVERY",
        submission_id="SUB-V15-RECOVERY",
        exam_id="EXAM-BOUND",
        questions=[q1, q2],
        page_count=1,
        paper_bytes=b"paper",
        solution_bytes=b"solution",
        student_content=[
            {"type": "input_text", "text": "Student answer-copy page 1, original"},
            {"type": "input_text", "text": "page one"},
        ],
        paper_filename="paper.pdf",
        solution_filename="solution.pdf",
        model_id="gpt-5.1-2025-11-13",
        reasoning_effort="low",
        temperature=0,
        paper_hash="paper-hash",
        solution_hash="solution-hash",
        pipeline_version=module._V15_PROMPT_VERSION,
    )

    merged, _raw, usage = await module._run_evidence_first_grading(
        gate=gate,
        existing_run=await db["evalpen_document_grading_runs"].find_one(
            {"run_id": "RUN-V15-RECOVERY"}
        ),
        **kwargs,
    )

    assert [call["metadata"]["pcr_stage"] for call in gate.calls] == [
        "bounded_student_evidence_mapping",
        "bounded_student_evidence_mapping_recovery",
        "bounded_mapped_evidence_grading",
    ]
    assert gate.calls[0]["reasoning_effort"] == "medium"
    assert gate.calls[1]["reasoning_effort"] == "medium"
    assert usage["stage_count"] == 3
    assert {row["question_number"] for row in merged["questions"]} == {1, 2}
    assert all(row["attempt_status"] == "attempted" for row in merged["questions"])
    assert {
        sum(item["marks_awarded"] for item in row["criterion_marks"])
        for row in merged["questions"]
    } == {1.0}
    saved = await db["evalpen_document_grading_runs"].find_one(
        {"run_id": "RUN-V15-RECOVERY"}
    )
    assert len(saved["evidence_mapping_recovery_units"]) == 1

    resumed_gate = _SequenceGate([])
    resumed, _raw, _usage = await module._run_evidence_first_grading(
        gate=resumed_gate,
        existing_run=saved,
        **kwargs,
    )
    assert resumed_gate.calls == []
    assert resumed == merged
