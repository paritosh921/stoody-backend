from services.answer_mapping_contract import (
    build_answer_key_mapping,
    select_effective_answer_mapping,
)
from services.answer_solution_coverage_service import AnswerSolutionCoverageService


def _document(**overrides):
    value = {
        "document_id": "doc-key-only",
        "exam_mode": "pcr",
        "answer_solution_mode": "upload",
        "answer_sheet_path": "uploads/answer-key.pdf",
        "answer_sheet_ocr_status": "completed",
        "answer_key_candidates": [
            {
                "question_id": "q1",
                "correct_answer": "B",
                "confidence": 0.92,
                "needs_review": False,
                "evidence": "1. B",
            }
        ],
    }
    value.update(overrides)
    return value


def _question(**overrides):
    value = {
        "id": "q1",
        "text": "Choose the correct value.",
        "options": ["150 m", "420 m", "2000 m", "800 m"],
        "correct_answer": "B",
        "points": 4,
    }
    value.update(overrides)
    return value


def test_accepted_answer_key_is_exposed_as_read_only_mapped_answer():
    mapping = build_answer_key_mapping(_document(), _question())

    assert mapping is not None
    assert mapping["mapping_id"] == "doc-key-only:q1:answer-key"
    assert mapping["answer_text"] == "B. 420 m"
    assert mapping["final_answer_text"] == "B. 420 m"
    assert mapping["mapped_question_text"] == "Choose the correct value."
    assert mapping["answer_kind"] == "answer_key"
    assert mapping["review_status"] == "accepted"
    assert mapping["manual_review_required"] is False
    assert mapping["virtual"] is True
    assert mapping["editable"] is False


def test_reviewed_worked_solution_takes_priority_over_answer_key():
    mapping = select_effective_answer_mapping(
        _document(),
        _question(),
        [
            {
                "mapping_id": "worked-1",
                "question_id": "q1",
                "answer_text": "Convert the speed and use horizontal projectile motion.",
                "source": "answer_sheet_full_ocr",
                "review_status": "accepted",
                "manual_review_required": False,
                "confidence": 0.95,
            }
        ],
    )

    assert mapping is not None
    assert mapping["mapping_id"] == "worked-1"


def test_accepted_answer_key_beats_an_unreviewed_ocr_guess():
    mapping = select_effective_answer_mapping(
        _document(),
        _question(),
        [
            {
                "mapping_id": "unreviewed-1",
                "question_id": "q1",
                "answer_text": "uncertain OCR fragment",
                "source": "answer_sheet_full_ocr",
                "review_status": "needs_review",
                "manual_review_required": True,
                "confidence": 0.99,
            }
        ],
    )

    assert mapping is not None
    assert mapping["answer_kind"] == "answer_key"
    assert mapping["answer_text"] == "B. 420 m"


def test_conflicting_uploaded_key_is_not_silently_trusted():
    document = _document(
        answer_key_candidates=[
            {
                "question_id": "q1",
                "correct_answer": "C",
                "confidence": 0.92,
                "needs_review": True,
            }
        ]
    )
    mapping = build_answer_key_mapping(document, _question(correct_answer="B"))

    assert mapping is not None
    assert mapping["correct_answer_candidate"] == "B"
    assert mapping["review_status"] == "needs_review"
    assert mapping["manual_review_required"] is True
    assert "answer_key_conflicts_with_saved_answer" in mapping["mapping_reasons"]


def test_key_only_uploaded_answer_sheet_has_complete_pcr_coverage():
    document = _document(
        answer_key_candidates=[
            {
                "question_id": f"q{index}",
                "correct_answer": label,
                "confidence": 0.92,
                "needs_review": False,
            }
            for index, label in enumerate(["B", "C"], start=1)
        ]
    )
    questions = [
        {
            "id": "q1",
            "question_type": "subjective",
            "options": ["A1", "B1", "C1", "D1"],
            "correct_answer": "B",
        },
        {
            "id": "q2",
            "question_type": "subjective",
            "options": ["A2", "B2", "C2", "D2"],
            "correct_answer": "C",
        },
    ]

    result = AnswerSolutionCoverageService().compute(
        document=document,
        questions=questions,
        mappings=[],
    )

    assert result["answer_solution_coverage_status"] == "ready"
    assert result["answer_solution_coverage_score"] == 1.0
    assert result["answer_solution_coverage_summary"]["mapped_answer_count"] == 2
    assert result["answer_solution_coverage_summary"]["answer_key_mapped_count"] == 2
    assert result["answer_solution_coverage_summary"]["worked_solution_mapped_count"] == 0
