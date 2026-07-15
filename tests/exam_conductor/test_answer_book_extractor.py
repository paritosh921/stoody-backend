"""Tests for content-section-style numbered answer extraction from OCR."""

from __future__ import annotations

from api.v1._exampen_imports import load_exampen


def _models():
    return load_exampen("pcr.domain.response_models")


def _extractor():
    return load_exampen("pcr.services.answer_book_extractor")


def _page(page_number: int, blocks: list[tuple[str, float, float]]):
    models = _models()
    return models.PageOCR(
        page_number=page_number,
        page_width_mm=210.0,
        page_height_mm=297.0,
        text_blocks=[
            models.TextBlock(
                text=text,
                bbox=models.BoundingBox(
                    x_min=12.0,
                    y_min=y0,
                    x_max=196.0,
                    y_max=y1,
                ),
                confidence=0.95,
                source="camera",
            )
            for text, y0, y1 in blocks
        ],
        source="camera",
        mean_ocr_confidence=0.95,
    )


def test_extracts_numbered_student_answer_book_ignoring_header():
    extractor = _extractor()
    pages = [
        _page(
            1,
            [
                ("Prayaan Answer Book Date Page", 2.0, 14.0),
                ("1) 30.067", 40.0, 55.0),
                ("2) 3630", 70.0, 85.0),
                ("3) Small diagram", 100.0, 130.0),
                ("4) Partition 5609.32 = 5000 + 600", 150.0, 190.0),
                ("5) Rohit goes to park on June (4,8,12,16,20,24,28)", 210.0, 250.0),
            ],
        )
    ]
    questions = [(n, {"question_text": f"Q{n}"}) for n in range(1, 10)]
    result = extractor.try_extract_answer_book_responses(pages, questions)
    assert result is not None
    responses, assignment = result
    by_q = {r.question_number: r.detected_text for r in responses}
    assert by_q[1] == "30.067" or "30.067" in by_q[1]
    assert "3630" in by_q[2]
    assert "Partition" in by_q[4]
    assert "Rohit" in by_q[5]
    assert "Prayaan" not in " ".join(by_q.values())
    assert all(
        assignment[r.response_id]["method"] == "answer_book_numbered_extract"
        for r in responses
    )


def test_extracts_inline_multi_answer_ocr_blob():
    extractor = _extractor()
    pages = [
        _page(
            1,
            [
                (
                    "1) 30.067 2) 3630 3) Small diagram 4) Partition 5609.32",
                    40.0,
                    120.0,
                ),
            ],
        )
    ]
    questions = [(n, {"question_text": f"Q{n}"}) for n in range(1, 5)]
    result = extractor.try_extract_answer_book_responses(pages, questions)
    assert result is not None
    responses, _assignment = result
    numbers = sorted(r.question_number for r in responses)
    assert numbers == [1, 2, 3, 4]
    assert "3630" in responses[1].detected_text


def test_returns_none_when_no_numbered_labels():
    extractor = _extractor()
    pages = [
        _page(1, [("Projectile motion: range is 38.4 m", 40.0, 80.0)]),
    ]
    questions = [(1, {"question_text": "Q1"}), (2, {"question_text": "Q2"})]
    assert extractor.try_extract_answer_book_responses(pages, questions) is None
