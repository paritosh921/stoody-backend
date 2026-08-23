from services.answer_mapping_contract import (
    build_answer_key_mapping,
    effective_answer_mappings,
    rebind_uploaded_answer_mappings,
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


def test_reocr_stale_uploaded_mapping_rebinds_by_unique_answer_number():
    questions = [
        {"id": "new-q1", "question_number": 1, "text": "Q1. Read the passage."},
        {"id": "new-q2", "question_number": 2, "text": "Q2. Edit the sentence."},
    ]
    mappings = [
        {
            "mapping_id": "doc:old-q1:a1",
            "question_id": "old-q1",
            "answer_number": "1",
            "answer_text": "Teacher answer one",
            "source": "answer_sheet_full_ocr",
            "review_status": "accepted",
        },
        {
            "mapping_id": "doc:old-q2:a2",
            "question_id": "old-q2",
            "answer_number": "2",
            "answer_text": "Teacher answer two",
            "source": "answer_sheet_full_ocr",
            "review_status": "needs_review",
            "manual_review_required": True,
        },
    ]

    rebound = rebind_uploaded_answer_mappings(questions, mappings)
    effective = effective_answer_mappings(_document(), questions, mappings)

    assert [mapping["question_id"] for mapping in rebound] == ["new-q1", "new-q2"]
    assert rebound[0]["source_question_id"] == "old-q1"
    assert rebound[0]["mapping_id"] == "doc:old-q1:a1"
    assert all(mapping["mapping_rebound_to_current_catalog"] for mapping in rebound)
    assert [mapping["question_id"] for mapping in effective] == ["new-q1", "new-q2"]


def test_uploaded_answer_path_overrides_stale_auto_mode_without_migration():
    result = AnswerSolutionCoverageService().compute(
        document={
            "document_id": "doc-race",
            "exam_mode": "pcr",
            "answer_solution_mode": "auto",
            "answer_sheet_path": "s3://bucket/teacher-answer.pdf",
            "answer_sheet_ocr_status": "completed",
            "answer_mapping_status": "needs_review",
        },
        questions=[{"id": "new-q1", "question_number": 1}],
        mappings=[{
            "question_id": "old-q1",
            "answer_number": "1",
            "answer_text": "Teacher supplied answer",
            "source": "answer_sheet_full_ocr",
            "review_status": "accepted",
            "manual_review_required": False,
        }],
    )

    assert result["answer_solution_coverage_status"] == "ready"
    assert result["answer_solution_coverage_summary"]["answer_source"] == "upload"
    assert result["answer_solution_coverage_summary"]["mapped_answer_count"] == 1
    assert result["answer_solution_coverage_summary"]["rebound_mapping_count"] == 1
