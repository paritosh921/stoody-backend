"""Vision evaluation evidence selection for diagram-heavy student answers."""

from __future__ import annotations

import pytest

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


def test_low_confidence_handwriting_requires_original_image_verification():
    vision = load_exampen("pcr.services.evidence_vision")
    assert vision.needs_vision_evaluation(
        content_type="TEXT_ONLY",
        detected_text="-13 and -23",
        question_text="Find the two integers",
        has_page_images=True,
        ocr_confidence=0.78,
        segmentation_confidence=0.93,
        question_assignment={"method": "verified_paper_page_order"},
    )


def test_high_confidence_plain_text_can_stay_on_text_path():
    vision = load_exampen("pcr.services.evidence_vision")
    assert not vision.needs_vision_evaluation(
        content_type="TEXT_ONLY",
        detected_text="The two integers are -12 and 3 because their sum is -9",
        question_text="Find the two integers",
        has_page_images=True,
        ocr_confidence=0.98,
        segmentation_confidence=0.98,
        question_assignment={"method": "verified_paper_page_order"},
    )


@pytest.mark.asyncio
async def test_uncertain_png_transcription_attaches_full_page_and_jpeg_crop(monkeypatch):
    vision = load_exampen("pcr.services.evidence_vision")
    ocr = load_exampen("pcr.services.ocr_service")

    async def _image_loader(_reference: str, **_kwargs):
        return "ZnVsbC1wYWdl"

    monkeypatch.setattr(ocr, "_resolve_image_base64", _image_loader)
    monkeypatch.setattr(
        vision,
        "_maybe_crop_page_image_b64",
        lambda *_args, **_kwargs: "Y3JvcC1qcGVn",
    )

    messages = await vision.build_vision_eval_messages(
        prompt="grade Q1",
        response_doc={
            "question_number": 1,
            "ocr_confidence": 0.78,
            "segmentation_confidence": 0.93,
            "question_assignment": {"method": "verified_paper_page_order"},
            "source_pages": [{"page_number": 1, "y_start": 70, "y_end": 140}],
        },
        answer_pages=[
            {"page_number": 1, "raw_image_ref": "s3://private/page-1.png"}
        ],
        question_text="Find the two integers",
    )

    assert messages is not None
    image_urls = [
        part["image_url"]["url"]
        for part in messages[0]["content"]
        if part.get("type") == "image_url"
    ]
    assert image_urls[0].startswith("data:image/png;base64,")
    assert image_urls[1].startswith("data:image/jpeg;base64,")
    assert "OCR transcription is untrusted" in messages[0]["content"][0]["text"]
