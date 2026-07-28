from api.v1.pdf_async import _build_test_series_activation_errors


def _test_series_document():
    return {
        "document_type": "Test Series",
        "question_type": "mcq",
        "total_minutes": 30,
        "answer_sheet_path": "uploads/answer.pdf",
    }


def _questions(count=2):
    return [
        {
            "id": f"q-{index}",
            "question_number": index,
            "question_type": "mcq",
            "options": ["First", "Second", "Third", "Fourth"],
            "correct_answer": "A",
        }
        for index in range(1, count + 1)
    ]


def test_objective_activation_ignores_worked_solution_review_status():
    errors = _build_test_series_activation_errors(
        document=_test_series_document(),
        questions=_questions(),
        answer_coverage={
            "answer_solution_coverage_status": "needs_review",
            "answer_solution_coverage_summary": {
                "question_count": 2,
                "mapped_answer_count": 2,
                "manual_review_count": 1,
            },
        },
    )

    assert errors == []


def test_objective_activation_uses_answer_key_without_worked_solution_mapping():
    errors = _build_test_series_activation_errors(
        document=_test_series_document(),
        questions=_questions(),
        answer_coverage={
            "answer_solution_coverage_status": "not_ready",
            "answer_solution_coverage_summary": {
                "question_count": 2,
                "mapped_answer_count": 1,
                "manual_review_count": 0,
            },
        },
    )

    assert errors == []


def test_subjective_activation_requires_uploaded_worked_solution_mapping():
    errors = _build_test_series_activation_errors(
        document={
            "document_type": "Test Series",
            "question_type": "subjective",
            "total_minutes": 30,
            "answer_sheet_path": "uploads/answer.pdf",
            "answer_solution_mode": "upload",
        },
        questions=[
            {"id": "q-1", "question_number": 1, "question_type": "subjective"},
            {"id": "q-2", "question_number": 2, "question_type": "subjective"},
        ],
        answer_coverage={
            "answer_solution_coverage_status": "not_ready",
            "answer_solution_coverage_summary": {
                "question_count": 2,
                "mapped_answer_count": 1,
                "manual_review_count": 0,
            },
        },
    )

    assert errors == [
        "Uploaded answer sheet is not fully mapped for subjective questions. "
        "1/2 subjective question(s) have mapped solutions."
    ]
