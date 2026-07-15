"""Vision evaluation evidence selection for diagram-heavy student answers."""

from __future__ import annotations

from api.v1._exampen_imports import load_exampen


def test_needs_vision_for_mixed_and_diagram_content():
    vision = load_exampen("pcr.services.evidence_vision")
    assert vision.needs_vision_evaluation(
        content_type="MIXED",
        detected_text="6) Factors",
        question_text="Draw a Venn diagram",
        has_page_images=True,
    )
    assert vision.needs_vision_evaluation(
        content_type="DIAGRAM_HEAVY",
        detected_text="",
        question_text="Show the factors",
        has_page_images=True,
    )


def test_needs_vision_for_short_ocr_on_diagram_questions():
    vision = load_exampen("pcr.services.evidence_vision")
    assert vision.needs_vision_evaluation(
        content_type="TEXT_ONLY",
        detected_text="6) Factors 18",
        question_text="Draw a Venn diagram of factors of 18 and 30",
        has_page_images=True,
    )


def test_needs_vision_for_near_empty_ocr():
    vision = load_exampen("pcr.services.evidence_vision")
    assert vision.needs_vision_evaluation(
        content_type="TEXT_ONLY",
        detected_text="6)",
        question_text="Factors of 18 and 30",
        has_page_images=True,
    )


def test_text_only_long_answer_skips_vision():
    vision = load_exampen("pcr.services.evidence_vision")
    long_text = " ".join(["step"] * 40)
    assert (
        vision.needs_vision_evaluation(
            content_type="TEXT_ONLY",
            detected_text=long_text,
            question_text="Explain the water cycle",
            has_page_images=True,
        )
        is False
    )


def test_normal_short_text_answer_skips_vision():
    vision = load_exampen("pcr.services.evidence_vision")
    assert (
        vision.needs_vision_evaluation(
            content_type="TEXT_ONLY",
            detected_text="Server-side OCR text only",
            question_text="Define osmosis carefully",
            has_page_images=True,
        )
        is False
    )


def test_no_images_skips_vision():
    vision = load_exampen("pcr.services.evidence_vision")
    assert (
        vision.needs_vision_evaluation(
            content_type="MIXED",
            detected_text="diagram",
            question_text="Draw",
            has_page_images=False,
        )
        is False
    )
