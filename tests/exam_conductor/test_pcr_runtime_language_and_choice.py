from __future__ import annotations

from api.v1._exampen_imports import load_exampen


policy = load_exampen("pcr.marking_policy")
language = load_exampen("pcr.language_assessment")
grading = load_exampen("pcr.services.full_document_grading")
whole_copy = load_exampen("pcr.services.whole_copy_grading")


def _optional_units(count: int = 5) -> list[dict]:
    return [
        {
            "unit_id": f"part_{index}",
            "label": f"({chr(96 + index)})",
            "prompt": f"Literature response {index}",
            "max_marks": 1 if index > 3 else 2,
            "scoring_model": "holistic_banded",
            "reference_solution": f"A valid response to part {index}.",
            "marking_criteria": [
                {
                    "criterion_id": "complete_response",
                    "description": f"Answers optional part {index}",
                    "max_marks": 1 if index > 3 else 2,
                    "acceptable_evidence": f"Relevant evidence for part {index}",
                }
            ],
        }
        for index in range(1, count + 1)
    ]


def _optional_question() -> dict:
    return {
        "question_id": "Q10",
        "question_number": 10,
        "question_text": "Answer ANY FOUR of the following five questions in 40-50 words each.",
        "subject": "Mathematics",  # intentionally stale imported metadata
        "question_type": "subjective",
        "grading_mode": "subjective",
        "max_marks": 8,
        "assessment_units": _optional_units(),
        "marking_criteria": [],
        "reference_solution": "Alternative literature answers.",
    }


def test_attempt_any_compiles_each_alternative_without_inflating_parent_budget():
    question = _optional_question()
    compiled = policy.compile_assessment_units_to_budget(
        question["assessment_units"],
        question["max_marks"],
        question_text=question["question_text"],
    )
    selection = policy.derive_response_selection(
        question["question_text"], compiled, question["max_marks"]
    )

    assert selection == {
        "version": "response-selection-v1",
        "mode": "attempt_any",
        "required_count": 4,
        "available_unit_ids": [f"part_{index}" for index in range(1, 6)],
        "per_unit_marks": 2.0,
    }
    assert [unit["max_marks"] for unit in compiled] == [2, 2, 2, 2, 2]
    assert policy.validate_assessment_units(
        compiled,
        8,
        question_text=question["question_text"],
        response_selection=selection,
        require_reference_solution=True,
    ) == []


def test_v16_runtime_repairs_legacy_choice_and_wrong_language_subject_without_writes():
    questions = [
        _optional_question(),
        {
            **_optional_question(),
            "question_id": "Q11",
            "question_number": 11,
            "question_text": "Write a formal letter to the editor about water conservation.",
            "max_marks": 5,
            "assessment_units": [],
            "marking_criteria": [{
                "criterion_id": "letter",
                "description": "Writes a relevant formal letter",
                "max_marks": 5,
                "acceptable_evidence": "Appropriate content and format",
            }],
        },
        {
            **_optional_question(),
            "question_id": "Q12",
            "question_number": 12,
            "question_text": "Read the passage and answer the comprehension questions.",
            "max_marks": 5,
            "assessment_units": [],
            "marking_criteria": [{
                "criterion_id": "comprehension",
                "description": "Answers the passage questions",
                "max_marks": 5,
                "acceptable_evidence": "Relevant passage-based answers",
            }],
        },
    ]
    original_marks = [unit["max_marks"] for unit in questions[0]["assessment_units"]]

    prepared = grading._prepare_runtime_question_catalog(questions)

    assert original_marks == [2, 2, 2, 1, 1]
    assert [unit["max_marks"] for unit in prepared[0]["assessment_units"]] == [2] * 5
    assert prepared[0]["response_selection"]["required_count"] == 4
    assert all(item["language_subject_inferred"] is True for item in prepared)
    assert language.language_feedback_profile(prepared[1])["enabled"] is True


def test_stem_catalogue_does_not_get_language_feedback_from_english_prose():
    questions = [
        {
            "question_id": f"Q{index}",
            "question_number": index,
            "question_text": text,
            "subject": "Physics",
            "question_type": "subjective",
        }
        for index, text in enumerate(
            (
                "Calculate the acceleration of the block.",
                "Derive the lens formula.",
                "Solve the equation for velocity.",
            ),
            start=1,
        )
    ]

    assert language.infer_language_paper(questions)["enabled"] is False
    assert all(
        language.language_feedback_profile(question)["enabled"] is False
        for question in questions
    )


def test_choice_schema_and_validator_score_only_visible_selected_units():
    prepared = grading._prepare_runtime_question_catalog([_optional_question()])[0]
    catalog = grading._catalog_question(prepared)
    schema = whole_copy.whole_copy_schema([catalog])
    item = schema["properties"]["questions"]["items"]
    assert item["properties"]["attempted_unit_ids"]["items"]["enum"] == [
        f"part_{index}" for index in range(1, 6)
    ]

    selected_ids = ["part_1", "part_3", "part_4"]
    by_unit = {
        criterion["assessment_unit_id"]: criterion
        for criterion in policy.flatten_assessment_unit_criteria(
            prepared["assessment_units"]
        )
    }
    criterion_marks = [
        {
            "criterion_id": by_unit[unit_id]["criterion_id"],
            "marks_awarded": 2,
            "confidence": 0.95,
            "rationale": "The visible response meets this optional part.",
            "evidence": "Relevant visible literature response.",
            "evidence_region_ids": ["q10-legacy-page-1"],
            "credit_basis": "direct_evidence",
        }
        for unit_id in selected_ids
    ]
    normalized_item = whole_copy.normalize_payload({
        "all_student_work_accounted": True,
        "questions": [{
            "question_number": 10,
            "attempt_status": "attempted",
            "confidence": 0.95,
            "student_answer": "The student answered parts (a), (c), and (d).",
            "content_type": "TEXT_ONLY",
            "source_pages": [1],
            "attempted_unit_ids": selected_ids,
            "criterion_marks": criterion_marks,
            "total_score": 8,
            "overall_feedback": "Three of four permitted responses were attempted.",
            "needs_review": False,
            "review_reason": "",
        }],
    })["questions"][0]
    result = grading._validate_question_grade(
        normalized_item,
        question=prepared,
        question_number=10,
        page_count=1,
    )

    assert result.attempt_status == "attempted"
    assert result.total_score == 6
    assert len(result.criterion_marks) == 3


def test_instruction_only_catalogue_row_is_unresolved_not_silently_scored():
    question = {
        "question_id": "Q1",
        "question_number": 1,
        "question_text": (
            "This question paper is divided into three sections. All questions are "
            "compulsory. Marks are indicated against each question."
        ),
        "subject": "English",
        "question_type": "subjective",
        "max_marks": 10,
        "marking_criteria": [{
            "criterion_id": "instructions",
            "description": "Follows paper instructions",
            "max_marks": 10,
            "acceptable_evidence": "Instructions",
        }],
    }
    prepared = grading._prepare_runtime_question_catalog([question])[0]
    result = grading._validate_question_grade(
        {
            "question_number": 1,
            "attempt_status": "attempted",
            "confidence": 0.9,
            "student_answer": "Student work",
            "content_type": "TEXT_ONLY",
            "source_pages": [],
            "criterion_marks": [],
            "total_score": 10,
            "overall_feedback": "",
            "needs_review": False,
            "review_reason": "",
        },
        question=prepared,
        question_number=1,
        page_count=1,
    )

    assert result.attempt_status == "unresolved"
    assert result.total_score is None
    assert "paper instructions" in result.review_reason


def test_new_paper_readiness_rejects_instruction_row_before_finalization():
    from services.exampen_paper_service import validate_pcr_questions

    errors = validate_pcr_questions(
        [{
            "id": "Q1",
            "question_text": (
                "This question paper is divided into three sections. All questions are "
                "compulsory. Marks are indicated against each question."
            ),
            "points": 1,
            "question_type": "subjective",
            "reference_solution": "Not an assessable question.",
        }],
        marking_policy={"mode": "legacy"},
    )

    assert any("paper instructions" in error for error in errors)
