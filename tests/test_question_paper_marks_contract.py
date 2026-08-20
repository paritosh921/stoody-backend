import json
from types import SimpleNamespace

import pytest

from services.exampen_paper_service import validate_pcr_questions
from services.question_paper_marks_contract import (
    effective_question_marks,
    extracted_marks_metadata,
    project_question_marks_for_authoring,
    summarize_question_marks,
    teacher_confirmed_marks_metadata,
    validate_visual_marks_evidence,
)


def _visual_evidence(value, text, confidence=0.99):
    return {
        "value": value,
        "printed_text": text,
        "page": 0,
        "bbox": {"x0": 800, "y0": 100, "x1": 980, "y1": 150},
        "confidence": confidence,
    }


def test_legacy_implicit_four_projects_provisional_one_without_mutating_stored_row():
    stored = {
        "id": "q2",
        "points": 4.0,
        "metadata": {"max_marks_extracted": False},
    }

    projected = project_question_marks_for_authoring(stored)

    assert stored["points"] == 4.0
    assert effective_question_marks(stored) == 1.0
    assert projected["points"] == 1.0
    assert projected["metadata"]["marks_status"] == "provisional_default"
    assert projected["metadata"]["marks_source"] == "system_default"
    assert projected["metadata"]["marks_review_required"] is True


def test_legacy_owned_four_remains_compatible():
    question = {"id": "manual-q", "points": 4.0, "metadata": {}}
    assert effective_question_marks(question) == 4.0
    assert project_question_marks_for_authoring(question)["points"] == 4.0


@pytest.mark.parametrize(
    ("value", "printed", "verified", "reason_fragment"),
    [
        (5, "(5x1=5)", True, None),
        (6, "(5x1=5)", False, "disagrees"),
        (6, "(5x1=6)", False, "arithmetically"),
        (2, "[2]", True, None),
        (4, "unclear", False, "not present"),
    ],
)
def test_visual_marks_require_printed_and_arithmetically_consistent_evidence(
    value, printed, verified, reason_fragment
):
    result = validate_visual_marks_evidence(_visual_evidence(value, printed))
    assert result["verified"] is verified
    if reason_fragment:
        assert reason_fragment in result["reason"]


def test_low_confidence_visual_mark_is_not_converted_to_a_score():
    result = extracted_marks_metadata(
        {"marks_evidence": _visual_evidence(4, "4 marks", confidence=0.52)},
        visual_source=True,
    )
    assert result["points"] == 1.0
    assert result["marks_status"] == "provisional_default"
    assert result["marks_source"] == "system_default"
    assert result["marks_review_required"] is True


def test_hindi_paper_marks_reconcile_to_twenty_without_a_default():
    printed = ["5x1=5", "2x0.5=1", "2x1=2", "2x1=2", "1", "2", "1", "1", "5"]
    values = [5, 1, 2, 2, 1, 2, 1, 1, 5]
    questions = []
    for number, (value, text) in enumerate(zip(values, printed), start=1):
        contract = extracted_marks_metadata(
            {"marks_evidence": _visual_evidence(value, text)},
            visual_source=True,
        )
        points = contract.pop("points")
        questions.append({"id": f"q{number}", "points": points, "metadata": contract})

    summary = summarize_question_marks(questions, expected_total=20)

    assert summary == {
        "question_count": 9,
        "resolved_count": 9,
        "authoritative_count": 9,
        "unresolved_count": 0,
        "unresolved_question_ids": [],
        "provisional_count": 0,
        "provisional_question_ids": [],
        "review_required_count": 0,
        "calculated_total": 20.0,
        "expected_total": 20.0,
        "reconciled": True,
    }


def test_teacher_confirmation_owns_a_previously_unresolved_mark():
    question = {
        "id": "q4",
        "points": 2.0,
        "metadata": teacher_confirmed_marks_metadata(
            {"max_marks_extracted": False},
            actor_id="teacher-1",
            confirmed_at="now",
        ),
    }
    assert effective_question_marks(question) == 2.0
    assert question["metadata"]["marks_status"] == "teacher_confirmed"


def test_pcr_finalization_allows_provisional_one_but_rejects_authoritative_total_mismatch():
    provisional_errors = validate_pcr_questions(
        [
            {
                "id": "q2",
                "text": "Question",
                "points": 4,
                "question_type": "subjective",
                "reference_solution": "Answer",
                "metadata": {"max_marks_extracted": False},
            }
        ],
        marking_policy={"mode": "legacy"},
    )
    assert not any("verify the printed marks" in error for error in provisional_errors)

    mismatch_errors = validate_pcr_questions(
        [
            {
                "id": "q1",
                "text": "Question",
                "points": 5,
                "question_type": "subjective",
                "reference_solution": "Answer",
                "metadata": {
                    "marks_status": "verified",
                    "paper_marks_reconciled": False,
                    "paper_marks_summary": {"expected_total": 6},
                },
            }
        ],
        marking_policy={"mode": "legacy"},
    )
    assert any("paper total" in error for error in mismatch_errors)

    corrected_errors = validate_pcr_questions(
        [
            {
                "id": "q1",
                "text": "Question",
                "points": 5,
                "question_type": "subjective",
                "reference_solution": "Answer",
                "metadata": {
                    "marks_status": "teacher_confirmed",
                    "marks_source": "teacher",
                    "paper_marks_reconciled": False,
                    "paper_marks_summary": {"expected_total": 5},
                },
            }
        ],
        marking_policy={"mode": "legacy"},
    )
    assert not any("paper total" in error for error in corrected_errors)


@pytest.mark.asyncio
async def test_test_series_visual_requirement_fails_closed_without_openai(monkeypatch):
    from api.v1 import pdf_async

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Original question-paper page images"):
        await pdf_async.extract_questions_with_gpt(
            {"pages": [{"index": 0, "markdown": "1. Question [4]", "images": []}]},
            "Hindi",
            "medium",
            pdf_bytes=b"fake-pdf",
            require_visual_source=True,
        )


@pytest.mark.asyncio
async def test_full_paper_structuring_sends_original_pages_and_defaults_missing_marks_to_reviewable_one(monkeypatch):
    from api.v1 import pdf_async
    import openai

    captured = {}
    payload = {
        "paper_total_marks": _visual_evidence(6, "M.M. 6"),
        "questions": [
            {
                "number": "1",
                "text": "Question one",
                "options": [],
                "page": 0,
                "has_figure": True,
                "max_marks": 5,
                "marks_evidence": _visual_evidence(5, "5x1=5"),
                "diagram_regions": [
                    {"page": 0, "bbox": {"x0": 100, "y0": 200, "x1": 600, "y1": 700}}
                ],
            },
            {
                "number": "2",
                "text": "Question two",
                "options": [],
                "page": 0,
                "has_figure": False,
                "max_marks": None,
                "marks_evidence": None,
            },
        ],
    }

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
            )

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeClient)
    monkeypatch.setattr(
        pdf_async,
        "_render_question_paper_pages_for_structuring",
        lambda _pdf: [
            {
                "index": 0,
                "label": "Original question-paper page 1",
                "data_uri": "data:image/jpeg;base64,ZmFrZQ==",
                "byte_size": 4,
            }
        ],
    )

    questions = await pdf_async.extract_questions_with_gpt(
        {"pages": [{"index": 0, "markdown": "OCR hint", "images": []}]},
        "Hindi",
        "medium",
        pdf_bytes=b"fake-pdf",
    )

    content = captured["messages"][0]["content"]
    assert any(part.get("type") == "image_url" for part in content)
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert questions[0].points == 5
    assert questions[0].metadata["marks_source"] == "visual_printed_evidence"
    assert questions[0].metadata["diagram_regions"]
    assert questions[1].points == 1.0
    assert questions[1].metadata["marks_status"] == "provisional_default"
    assert questions[1].metadata["marks_review_required"] is True
    assert all(question.points != 4 for question in questions)


def test_question_extraction_contract_normalises_compound_language_option_objects():
    from services.question_extraction_contract import (
        QUESTION_EXTRACTION_CONTRACT_VERSION,
        normalize_question_extraction_payload,
    )

    result = normalize_question_extraction_payload({
        "paper_total_marks": None,
        "questions": [{
            "number": "4",
            "text": "Read the extract and answer all parts.",
            "options": [
                {"label": "A", "text": "the river was rising"},
                {"B": "the bridge remained firm"},
                "personification",
            ],
            "page": 0,
            "has_figure": False,
            "max_marks": 8,
            "marks_evidence": None,
        }],
    })

    assert result["contract_version"] == QUESTION_EXTRACTION_CONTRACT_VERSION
    assert result["questions"][0]["options"] == [
        "the river was rising",
        "the bridge remained firm",
        "personification",
    ]
    assert result["questions"][0]["continuation_pages"] == []
    assert result["questions"][0]["diagram_regions"] == []


def test_question_extraction_contract_rejects_missing_question_text():
    from services.question_extraction_contract import (
        QuestionExtractionContractError,
        normalize_question_extraction_payload,
    )

    with pytest.raises(QuestionExtractionContractError) as captured:
        normalize_question_extraction_payload({
            "paper_total_marks": None,
            "questions": [{"number": "1", "text": "", "options": []}],
        })

    assert captured.value.code == "incomplete_model_output"


def test_document_ocr_failure_fields_are_durable_bounded_and_stage_specific():
    from api.v1.pdf_async import _document_ocr_failure_fields
    from services.question_extraction_contract import QuestionExtractionContractError

    result = _document_ocr_failure_fields(
        QuestionExtractionContractError(
            "invalid_model_output",
            "Question extraction returned an invalid response shape.",
        ),
        stage="question_extraction",
        job_id="job-123",
    )

    assert result["ocr_status"] == "error"
    assert result["ocr_job_id"] == "job-123"
    assert result["ocr_error_code"] == "invalid_model_output"
    assert result["ocr_error_stage"] == "question_extraction"
    assert result["ocr_error"] == "Question extraction returned an invalid response shape."
    assert result["ocr_failed_at"] is not None


def _weighted_unit(label: str) -> dict:
    suffix = label.strip("()")
    return {
        "unit_id": f"unit_{suffix}",
        "label": label,
        "prompt": f"Answer part {label}",
        "relative_weight": 1,
        "scoring_model": "point_based",
        "reference_solution": f"Solution {label}",
        "marking_criteria": [
            {
                "criterion_id": f"criterion_{suffix}",
                "description": f"Correctly answers {label}",
                "relative_weight": 1,
                "acceptable_evidence": f"Solution {label} or an equivalent answer",
            }
        ],
    }


def test_one_mark_multipart_draft_compiles_to_one_inclusive_unit():
    from api.v1.pdf_async import _parse_pcr_marking_plan_draft

    draft = _parse_pcr_marking_plan_draft(
        json.dumps({"assessment_units": [_weighted_unit("(a)"), _weighted_unit("(b)"), _weighted_unit("(c)")]}),
        question_marks=1,
        question_text="Answer (a), (b), and (c).",
    )

    assert len(draft["assessment_units"]) == 1
    assert draft["assessment_units"][0]["max_marks"] == 1.0
    assert draft["assessment_units"][0]["label"] == "Whole question"
    assert len(draft["marking_criteria"]) == 1
    assert draft["marking_criteria"][0]["max_marks"] == 1.0
    assert all(part in draft["reference_solution"] for part in ("(a)", "(b)", "(c)"))


def test_known_question_total_compiles_relative_weights_to_exact_step_marks():
    from api.v1.pdf_async import _parse_pcr_marking_plan_draft

    draft = _parse_pcr_marking_plan_draft(
        json.dumps({"assessment_units": [_weighted_unit("(a)"), _weighted_unit("(b)"), _weighted_unit("(c)")]}),
        question_marks=5,
        question_text="Answer (a), (b), and (c). [5]",
    )

    unit_marks = [unit["max_marks"] for unit in draft["assessment_units"]]
    assert len(unit_marks) == 3
    assert sum(unit_marks) == 5.0
    assert all(mark > 0 for mark in unit_marks)
    assert sum(item["max_marks"] for item in draft["marking_criteria"]) == 5.0


def test_printed_fractional_subpart_ledger_is_preserved_for_one_mark_question():
    from api.v1.pdf_async import _parse_pcr_marking_plan_draft

    draft = _parse_pcr_marking_plan_draft(
        json.dumps({"assessment_units": [_weighted_unit("(a)"), _weighted_unit("(b)")]}),
        question_marks=1,
        question_text="Answer both parts. (2 x 0.5 = 1)",
    )

    assert [unit["max_marks"] for unit in draft["assessment_units"]] == [0.5, 0.5]
