from __future__ import annotations

from copy import deepcopy

from api.v1._exampen_imports import load_exampen


language = load_exampen("pcr.language_assessment")
grading = load_exampen("pcr.services.full_document_grading")
whole_copy = load_exampen("pcr.services.whole_copy_grading")


def _question(*, subject: str = "Hindi", text: str = "पत्र लिखिए।") -> dict:
    return {
        "question_id": "Q1",
        "question_number": 1,
        "question_text": text,
        "subject": subject,
        "question_type": "subjective",
        "grading_mode": "subjective",
        "max_marks": 5,
        "reference_solution": "A relevant, complete response in the requested format.",
        "marking_criteria": [
            {
                "criterion_id": "response_quality",
                "description": "Fulfils the requested language task",
                "max_marks": 5,
                "acceptable_evidence": "Relevant content in the requested form",
            }
        ],
    }


def _feedback(profile: dict) -> dict:
    return {
        "version": language.LANGUAGE_FEEDBACK_VERSION,
        "response_family": profile["response_family"],
        "feedback_language": "Hindi",
        "summary": "उत्तर प्रासंगिक है और अभिव्यक्ति सामान्यतः स्पष्ट है।",
        "priority_actions": ["समापन को पूरा करें।"],
        "example_revision": "भवदीय,\nसिया",
        "dimensions": [
            {
                "dimension_id": item["dimension_id"],
                "applicability": (
                    "applicable" if item["applicable"] else "not_applicable"
                ),
                "level": "secure" if item["applicable"] else "not_applicable",
                "evidence": "Visible supporting phrase" if item["applicable"] else "",
                "feedback": "One specific next step" if item["applicable"] else "",
            }
            for item in profile["dimensions"]
        ],
    }


def test_hindi_functional_writing_enables_all_seven_dimensions():
    profile = language.language_feedback_profile(
        _question(text="प्रधानाचार्य को दो दिन के अवकाश के लिए पत्र लिखिए।")
    )

    assert profile["enabled"] is True
    assert profile["response_family"] == "functional_writing"
    assert len(profile["dimensions"]) == 7
    assert all(item["applicable"] for item in profile["dimensions"])


def test_language_task_type_controls_dimension_applicability():
    profile = language.language_feedback_profile(
        _question(text="दिए गए शब्दों के विलोम और पर्यायवाची लिखिए।")
    )
    applicability = {
        item["dimension_id"]: item["applicable"]
        for item in profile["dimensions"]
    }

    assert profile["response_family"] == "grammar_vocabulary"
    assert applicability["language_grammar"] is True
    assert applicability["conciseness_precision"] is True
    assert applicability["structure_organization"] is False
    assert applicability["tone_style"] is False


def test_non_language_subject_does_not_receive_writing_profile():
    profile = language.language_feedback_profile(
        _question(subject="Physics", text="Derive the lens formula.")
    )

    assert profile == {
        "enabled": False,
        "version": language.LANGUAGE_FEEDBACK_VERSION,
    }

    assert language.language_feedback_profile(
        _question(subject="English Medium Physics", text="Explain Newton's law.")
    )["enabled"] is False


def test_objective_question_cannot_enable_language_profile_by_override():
    question = _question()
    question["grading_mode"] = "objective"
    question["question_type"] = "mcq"
    question["language_feedback_profile"] = {
        "enabled": True,
        "response_family": "grammar_vocabulary",
    }

    assert language.language_feedback_profile(question) == {
        "enabled": False,
        "version": language.LANGUAGE_FEEDBACK_VERSION,
    }


def test_native_script_language_subject_is_detected_without_database_changes():
    profile = language.language_feedback_profile(
        _question(subject="हिंदी भाषा", text="अनुच्छेद लिखिए।")
    )

    assert profile["enabled"] is True
    assert profile["response_family"] == "creative_writing"


def test_whole_copy_schema_adds_feedback_only_to_language_questions():
    language_catalog = grading._catalog_question(_question())
    language_schema = whole_copy.whole_copy_schema([language_catalog])
    item_schema = language_schema["properties"]["questions"]["items"]

    assert "language_feedback" in item_schema["properties"]
    assert "language_feedback" in item_schema["required"]
    dimension_schema = item_schema["properties"]["language_feedback"]["properties"][
        "dimensions"
    ]
    assert len(dimension_schema["required"]) == 7
    assert set(dimension_schema["properties"]) == set(dimension_schema["required"])

    maths_catalog = grading._catalog_question(
        _question(subject="Mathematics", text="Solve x + 1 = 2.")
    )
    maths_schema = whole_copy.whole_copy_schema([maths_catalog])
    maths_item = maths_schema["properties"]["questions"]["items"]
    assert "language_feedback" not in maths_item["properties"]


def test_language_feedback_is_diagnostic_and_cannot_change_marks():
    question = _question()
    profile = language.language_feedback_profile(question)
    payload = {
        "question_number": 1,
        "attempt_status": "attempted",
        "confidence": 0.9,
        "student_answer": "आदरणीय प्रधानाचार्य जी, मुझे दो दिन का अवकाश चाहिए।",
        "content_type": "TEXT_ONLY",
        "source_pages": [1],
        "criterion_marks": [
            {
                "criterion_id": "response_quality",
                "marks_awarded": 3,
                "confidence": 0.9,
                "rationale": "The purpose is clear but the closing is incomplete.",
                "evidence": "दो दिन का अवकाश चाहिए",
                "evidence_region_ids": ["q1-legacy-page-1"],
                "credit_basis": "direct_evidence",
            }
        ],
        "total_score": 5,
        "overall_feedback": "Complete the closing.",
        "needs_review": False,
        "review_reason": "",
        "language_feedback": _feedback(profile),
    }

    result = grading._validate_question_grade(
        payload,
        question=question,
        question_number=1,
        page_count=1,
    )

    assert result.total_score == 3
    assert result.manual_review_required is False
    assert "**Understanding (Secure):**" in result.overall_feedback
    assert "**Language & Grammar (Secure):**" in result.overall_feedback


def test_malformed_diagnostic_feedback_never_blocks_a_valid_score():
    question = _question()
    profile = language.language_feedback_profile(question)
    payload = {
        "question_number": 1,
        "attempt_status": "attempted",
        "confidence": 0.9,
        "student_answer": "उत्तर लिखा गया है।",
        "content_type": "TEXT_ONLY",
        "source_pages": [1],
        "criterion_marks": [
            {
                "criterion_id": "response_quality",
                "marks_awarded": 4,
                "confidence": 0.9,
                "rationale": "Most requirements are met.",
                "evidence": "उत्तर लिखा गया है",
                "evidence_region_ids": ["q1-legacy-page-1"],
                "credit_basis": "direct_evidence",
            }
        ],
        "total_score": 0,
        "overall_feedback": "Good response.",
        "needs_review": False,
        "review_reason": "",
        "language_feedback": _feedback(profile),
    }
    payload["language_feedback"]["dimensions"] = payload["language_feedback"][
        "dimensions"
    ][:-1]

    result = grading._validate_question_grade(
        deepcopy(payload),
        question=question,
        question_number=1,
        page_count=1,
    )

    assert result.total_score == 4
    assert result.attempt_status == "attempted"
    assert result.manual_review_required is False
    assert result.overall_feedback == "Good response."


def test_unclear_language_dimension_can_be_preserved_as_not_assessed():
    profile = language.language_feedback_profile(_question())
    raw = _feedback(profile)
    raw["dimensions"][0]["level"] = "not_assessed"
    raw["dimensions"][0]["evidence"] = ""
    raw["dimensions"][0]["feedback"] = ""

    normalized = language.normalize_language_feedback(
        raw,
        profile=profile,
        attempted=True,
    )

    assert normalized is not None
    assert normalized["dimensions"][0]["level"] == "not_assessed"


def test_validated_dimensions_are_curated_into_existing_feedback_text():
    profile = language.language_feedback_profile(_question())
    normalized = language.normalize_language_feedback(
        _feedback(profile), profile=profile, attempted=True
    )

    text = language.format_language_feedback(normalized)

    assert "उत्तर प्रासंगिक है" in text
    assert "**Understanding (Secure):**" in text
    assert "**Priority improvements:**" in text
    assert "**Example improvement:**" in text
