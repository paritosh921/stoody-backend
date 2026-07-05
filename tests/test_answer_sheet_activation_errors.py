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
            "correct_answer": "A",
        }
        for index in range(1, count + 1)
    ]


def test_activation_reports_review_pending_without_incomplete_mapping_message():
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

    assert errors == ["1 mapped answer(s) still need manual review before activation."]


def test_activation_reports_incomplete_mapping_only_when_questions_are_unmapped():
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

    assert errors == [
        "Uploaded answer sheet is not fully mapped. 1/2 question(s) have mapped solutions."
    ]
