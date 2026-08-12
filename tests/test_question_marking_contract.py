import pytest

from services.question_marking_contract import (
    normalize_question_penalty,
    parse_question_penalty,
    supports_negative_marking,
)


def test_subjective_and_unclassified_questions_cannot_have_negative_marks():
    assert normalize_question_penalty(1, question_type="subjective") == 0
    assert normalize_question_penalty(1, question_type="unclassified") == 0
    assert not supports_negative_marking("subjective")


def test_objective_questions_keep_explicit_and_historical_default_penalties():
    assert normalize_question_penalty(0.5, question_type="mcq") == 0.5
    assert normalize_question_penalty(None, question_type="integer") == 1
    assert supports_negative_marking("objective")


def test_document_type_is_used_only_when_the_question_has_no_explicit_type():
    assert normalize_question_penalty(
        None,
        question_type=None,
        document_question_type="mcq",
    ) == 1
    assert normalize_question_penalty(
        1,
        question_type=None,
        document_question_type="subjective",
    ) == 0


@pytest.mark.parametrize(
    "value",
    [True, -0.25, 50.01, float("nan"), float("inf"), "bad"],
)
def test_strict_penalty_parser_rejects_values_that_cannot_be_frozen(value):
    with pytest.raises(ValueError):
        parse_question_penalty(value, allow_missing=False)


def test_strict_penalty_parser_accepts_bounded_fractional_marks():
    assert parse_question_penalty("0.25", allow_missing=False) == 0.25


def test_pcr_metadata_adapter_drops_legacy_subjective_penalty():
    from api.v1._exampen_imports import load_exampen

    adapter = load_exampen("pcr.metadata_adapter")
    adapted = adapter.adapt_question_to_pcr(
        {
            "id": "q1",
            "question_type": "subjective",
            "text": "Explain the water cycle.",
            "points": 4,
            "penalty": 1,
        },
        exam_id="exam1",
    )

    assert adapted["grading_mode"] == "subjective"
    assert adapted["penalty_marks"] == 0
