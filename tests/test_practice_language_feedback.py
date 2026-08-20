from __future__ import annotations

import json

from api.v1 import practice_async as practice


def _question(*, subject: str, text: str) -> dict:
    return {
        "_id": "practice-q1",
        "text": text,
        "subject": subject,
        "question_type": "subjective",
        "correctAnswer": "A relevant answer",
    }


def _raw_feedback(profile: dict) -> dict:
    return {
        "version": "language-feedback-v1",
        "response_family": profile["response_family"],
        "feedback_language": "Hindi",
        "summary": "उत्तर प्रासंगिक है और अभिव्यक्ति स्पष्ट है।",
        "priority_actions": ["वाक्य रचना को अधिक सटीक बनाएं।"],
        "example_revision": "मैं दो दिन के अवकाश के लिए आवेदन कर रही हूँ।",
        "dimensions": {
            item["dimension_id"]: {
                "applicability": (
                    "applicable" if item["applicable"] else "not_applicable"
                ),
                "level": "secure" if item["applicable"] else "not_applicable",
                "evidence": "उत्तर में स्पष्ट कथन है।" if item["applicable"] else "",
                "feedback": "इसी स्पष्टता को बनाए रखें।" if item["applicable"] else "",
            }
            for item in profile["dimensions"]
        },
    }


def test_hindi_medium_physics_remains_non_language_subject():
    profile = practice._practice_language_feedback_profile(
        _question(subject="Physics", text="हिंदी में न्यूटन का दूसरा नियम समझाइए।"),
        is_mcq=False,
    )

    assert profile["enabled"] is False

    english_medium_profile = practice._practice_language_feedback_profile(
        _question(subject="English Medium Physics", text="Explain Newton's law."),
        is_mcq=False,
    )
    assert english_medium_profile["enabled"] is False


def test_subjective_hindi_letter_enables_seven_dimension_feedback():
    profile = practice._practice_language_feedback_profile(
        _question(subject="Hindi", text="प्रधानाचार्य को अवकाश के लिए पत्र लिखिए।"),
        is_mcq=False,
    )

    assert profile["enabled"] is True
    assert profile["response_family"] == "functional_writing"
    assert len(profile["dimensions"]) == 7


def test_non_language_prompt_has_hard_grammar_boundary_and_no_feedback_field():
    profile = practice._practice_language_feedback_profile(
        _question(subject="Mathematics", text="Solve x + 1 = 2."),
        is_mcq=False,
    )
    prompt = practice._build_evaluation_prompt(
        question_text="Solve x + 1 = 2.",
        options_text="",
        correct_answer="x = 1",
        correct_answer_value="x = 1",
        is_option_letter=False,
        is_mcq=False,
        answer_text="x=1",
        uploaded_doc_text="",
        num_student_images=0,
        num_question_figures=0,
        num_option_images=0,
        language_feedback_profile=profile,
    )

    assert "NON-LANGUAGE SUBJECT BOUNDARY" in prompt
    output = prompt.split("OUTPUT — strict JSON only", 1)[1]
    assert '"language_feedback"' not in output
    assert "Do not lower the score or criticize spelling" in prompt


def test_language_prompt_requests_all_dimensions_in_the_same_model_call():
    profile = practice._practice_language_feedback_profile(
        _question(subject="English", text="Write a letter to the editor."),
        is_mcq=False,
    )
    prompt = practice._build_evaluation_prompt(
        question_text="Write a letter to the editor.",
        options_text="",
        correct_answer="A relevant formal letter",
        correct_answer_value="A relevant formal letter",
        is_option_letter=False,
        is_mcq=False,
        answer_text="Dear Editor, ...",
        uploaded_doc_text="",
        num_student_images=0,
        num_question_figures=0,
        num_option_images=0,
        language_feedback_profile=profile,
    )

    assert "LANGUAGE-WRITING DIAGNOSTIC" in prompt
    for dimension in (
        "understanding",
        "content",
        "structure_organization",
        "language_grammar",
        "clarity_expression",
        "tone_style",
        "conciseness_precision",
    ):
        assert f'"{dimension}"' in prompt


def test_language_diagnostic_cannot_modify_practice_score():
    profile = practice._practice_language_feedback_profile(
        _question(subject="Hindi", text="एक अनुच्छेद लिखिए।"),
        is_mcq=False,
    )
    raw = {
        "is_correct": False,
        "score": 0.4,
        "extracted_answer": "विद्यार्थी का उत्तर",
        "work_shown": "एक छोटा अनुच्छेद लिखा।",
        "what_went_wrong": "मुख्य विचार अधूरा है।",
        "language_feedback": _raw_feedback(profile),
    }

    parsed = practice._parse_evaluation_response(
        raw_response=json.dumps(raw, ensure_ascii=False),
        correct_answer_display="आदर्श उत्तर",
        has_correct_answer=True,
        answer_text="",
        language_feedback_profile=profile,
    )

    assert parsed["correct"] is False
    assert parsed["score"] == 0.4
    assert "**Understanding (Secure):**" in parsed["whatWentWrong"]
    assert "**Language & Grammar (Secure):**" in parsed["whatWentWrong"]
    assert "languageFeedback" not in parsed


def test_malformed_language_diagnostic_is_dropped_without_losing_verdict():
    profile = practice._practice_language_feedback_profile(
        _question(subject="English", text="Write a paragraph."),
        is_mcq=False,
    )
    raw = {
        "is_correct": True,
        "score": 0.85,
        "extracted_answer": "A paragraph",
        "work_shown": "The student wrote a paragraph.",
        "what_went_wrong": "",
        "language_feedback": {"dimensions": {}},
    }

    parsed = practice._parse_evaluation_response(
        raw_response=json.dumps(raw),
        correct_answer_display="Reference paragraph",
        has_correct_answer=True,
        answer_text="",
        language_feedback_profile=profile,
    )

    assert parsed["correct"] is True
    assert parsed["score"] == 0.85
    assert "languageFeedback" not in parsed
