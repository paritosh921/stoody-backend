from api.v1._exampen_imports import load_exampen
from services.exampen_paper_service import validate_pcr_questions
from services.objective_scoring_service import score_objective_response


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
