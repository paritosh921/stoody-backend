from api.v1.pdf_async import (
    _build_test_series_activation_errors,
    _missing_correct_answer_question_numbers,
)


def test_mixed_content_activation_requires_question_categories_and_objective_answers_only():
    document = {
        "document_type": "Test Series",
        "question_type": "mixed",
        "total_minutes": 60,
    }
    questions = [
        {"question_number": 1, "question_type": "mcq", "correct_answer": ""},
        {"question_number": 2, "question_type": "subjective", "correct_answer": ""},
        {"question_number": 3, "question_type": "unclassified", "correct_answer": ""},
        {"question_number": 4, "correct_answer": ""},
    ]

    assert _missing_correct_answer_question_numbers(questions, document) == [1]

    errors = _build_test_series_activation_errors(
        document=document,
        questions=questions,
    )

    assert "1" in errors
    assert "Question category is not selected for: 3, 4" in errors
