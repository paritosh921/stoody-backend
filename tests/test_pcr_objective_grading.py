import pytest

from api.v1._exampen_imports import load_exampen
from services.exampen_paper_service import (
    resolve_question_layout_for_finalization,
    validate_pcr_questions,
)
from services.objective_answer_ledger_contract import (
    OBJECTIVE_LEDGER_VERSION,
    OBJECTIVE_PAPER_CONTEXT_VERSION,
    merge_objective_page_ledgers,
    objective_extraction_catalog,
    objective_page_observation_schema,
    objective_reader_instructions,
)
from services.objective_scoring_service import (
    ObjectiveScoringContractError,
    score_objective_response,
)


def _objective_question(**overrides):
    question = {
        "id": "q-1",
        "question_id": "q-1",
        "question_number": 1,
        "question_text": "Choose the correct value.",
        "question_type": "mcq",
        "grading_mode": "objective",
        "points": 4,
        "max_marks": 4,
        "penalty": 1,
        "options": [
            {"label": "A", "text": "One"},
            {"label": "B", "text": "Two"},
            {"label": "C", "text": "Three"},
            {"label": "D", "text": "Four"},
        ],
        "correct_answer": "B",
    }
    question.update(overrides)
    return question


def test_objective_pcr_finalization_uses_options_and_key_not_subjective_criteria():
    assert validate_pcr_questions(
        [_objective_question()],
        marking_policy={"mode": "criterion_rubric_v1"},
    ) == []


def test_objective_paper_finalization_selects_answer_ledger_contract():
    resolved = resolve_question_layout_for_finalization(
        {
            "document_id": "DOC-OBJECTIVE",
            "file_path": "s3://private/papers/objective.pdf",
            "pages_count": 1,
        },
        [_objective_question()],
        None,
    )

    assert resolved["ready"] is True
    assert (
        resolved["paper_context"]["version"]
        == OBJECTIVE_PAPER_CONTEXT_VERSION
    )
    assert resolved["paper_context"]["mode"] == "objective_answer_ledger"


def test_objective_extraction_catalog_never_contains_correct_answer():
    catalog = objective_extraction_catalog([_objective_question()])

    assert catalog == [
        {
            "question_number": 1,
            "answer_format": "option_label",
            "allowed_option_labels": ["A", "B", "C", "D"],
        }
    ]
    assert "correct_answer" not in str(catalog)
    assert "One" not in str(catalog)


def test_objective_schema_only_allows_conducted_question_numbers():
    schema = objective_page_observation_schema([1, 2, 75])

    number_schema = schema["properties"]["observations"]["items"][
        "properties"
    ]["question_number"]
    assert number_schema["enum"] == [1, 2, 75]
    assert "bbox" not in schema["properties"]["observations"]["items"][
        "properties"
    ]


def test_reader_contract_supports_omr_written_lists_and_nonstandard_marks():
    instructions = objective_reader_instructions()

    assert "OMR grid" in instructions
    assert "handwritten list" in instructions
    assert "crossed out" in instructions
    assert "clean filled bubble is the final selected answer" in instructions
    assert "two or more non-cancelled options" in instructions


def test_selected_objective_model_has_nonzero_usage_pricing():
    provider = load_exampen("llm_gate.provider")

    assert provider.estimate_cost("gpt-5.6-sol", 1_000_000, 1_000_000) == 35


def test_objective_page_ledger_merges_selected_blank_and_ambiguous_states():
    questions = [
        _objective_question(question_number=1, question_id="q-1"),
        _objective_question(question_number=2, question_id="q-2"),
        _objective_question(question_number=3, question_id="q-3"),
    ]
    payload, errors = merge_objective_page_ledgers(
        [
            {
                "ledger_version": OBJECTIVE_LEDGER_VERSION,
                "page_number": 1,
                "sheet_format": "omr_grid",
                "page_fully_reviewed": True,
                "observations": [
                    {
                        "question_number": 1,
                        "state": "selected",
                        "selected_answer": "Option C",
                        "alternative_answers": [],
                        "confidence": 0.98,
                    },
                    {
                        "question_number": 2,
                        "state": "blank",
                        "selected_answer": "",
                        "alternative_answers": [],
                        "confidence": 0.96,
                    },
                    {
                        "question_number": 3,
                        "state": "multiple_selected",
                        "selected_answer": "",
                        "alternative_answers": ["A", "D"],
                        "confidence": 0.91,
                    },
                ],
            }
        ],
        questions=questions,
        page_count=1,
    )

    assert errors == []
    by_number = {
        item["question_number"]: item for item in payload["questions"]
    }
    assert by_number[1]["attempt_status"] == "attempted"
    assert by_number[1]["student_answer"] == "C"
    assert by_number[2]["attempt_status"] == "not_attempted"
    assert by_number[3]["attempt_status"] == "unresolved"
    assert "More than one option" in by_number[3]["review_reason"]


def test_numbered_answer_list_treats_missing_catalog_entries_as_not_attempted():
    payload, errors = merge_objective_page_ledgers(
        [
            {
                "ledger_version": OBJECTIVE_LEDGER_VERSION,
                "page_number": 1,
                "sheet_format": "numbered_answer_list",
                "page_fully_reviewed": True,
                "observations": [
                    {
                        "question_number": 1,
                        "state": "selected",
                        "selected_answer": "B",
                        "alternative_answers": [],
                        "confidence": 0.98,
                    },
                ],
            }
        ],
        questions=[
            _objective_question(),
            _objective_question(question_number=2, question_id="q-2"),
        ],
        page_count=1,
    )

    assert errors == []
    assert payload["document_review"]["all_student_work_accounted"] is True
    assert payload["questions"][0]["student_answer"] == "B"
    assert payload["questions"][1]["attempt_status"] == "not_attempted"


def test_objective_pcr_finalization_rejects_incomplete_objective_contract():
    errors = validate_pcr_questions(
        [_objective_question(options=[], correct_answer="")],
        marking_policy={"mode": "criterion_rubric_v1"},
    )

    assert "add at least two objective answer options" in errors[0]
    assert any("select the correct objective answer" in error for error in errors)


def test_online_and_camera_objective_scoring_share_negative_marking_contract():
    question = _objective_question()

    assert score_objective_response(question, "Option B")["points_earned"] == 4
    wrong = score_objective_response(question, "(C)")
    assert wrong["selected_answer"] == "C"
    assert wrong["points_earned"] == -1
    assert score_objective_response(question, "")["points_earned"] == 0


def test_objective_scorer_rejects_a_label_outside_the_frozen_options():
    with pytest.raises(
        ObjectiveScoringContractError,
        match="frozen objective options",
    ):
        score_objective_response(_objective_question(), "Option E")


def test_camera_grader_transcribes_option_then_server_scores_it():
    module = load_exampen("pcr.services.full_document_grading")
    question = _objective_question()
    item = {
        "question_number": 1,
        "attempt_status": "attempted",
        "confidence": 0.96,
        "student_answer": "C",
        "content_type": "TEXT_ONLY",
        "evidence_regions": [
            {
                "region_id": "q1-option",
                "page_number": 1,
                "x_start": 100,
                "y_start": 100,
                "x_end": 300,
                "y_end": 200,
                "evidence_kind": "handwriting",
                "continuation_group": "",
                "evidence": "Q1 C",
                "mapping_confidence": 0.96,
            }
        ],
        "interpretation_hypotheses": [
            {
                "interpretation_id": "q1-reading",
                "value": "C",
                "confidence": 0.96,
                "evidence_region_ids": ["q1-option"],
                "ambiguity_notes": "",
            }
        ],
        "visual_semantics": {
            "summary": "",
            "elements": [],
            "relationships": [],
            "confidence": 0,
        },
        "method_analysis": module._not_applicable_method_analysis(),
        "criterion_marks": [],
        "total_score": 0,
        "overall_feedback": "",
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

    assert grade.student_answer == "C"
    assert grade.total_score == -1
    assert grade.criterion_marks == []
    assert grade.manual_review_required is False
