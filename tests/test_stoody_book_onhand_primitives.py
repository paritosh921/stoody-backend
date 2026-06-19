import pytest
from fastapi import HTTPException

from api.v1 import stoody_book_async as stoody_book


def test_pdf_pages_prefers_stored_page_texts():
    pages = stoody_book._pdf_pages({
        "page_texts": [
            {"page": 1, "text": " Intro  text "},
            {"page": 2, "text": ""},
            {"page": 3, "text": "Worked example"},
        ],
        "text": "Page 1: stale",
    })

    assert pages == [
        {"page": 1, "text": "Intro text"},
        {"page": 3, "text": "Worked example"},
    ]


def test_pdf_pages_falls_back_to_legacy_page_markers():
    pages = stoody_book._pdf_pages({
        "text": "Page 1: Photosynthesis uses light.\n\nPage 2: Chlorophyll captures energy.",
    })

    assert pages == [
        {"page": 1, "text": "Photosynthesis uses light."},
        {"page": 2, "text": "Chlorophyll captures energy."},
    ]


def test_search_pdf_pages_returns_ranked_page_snippets():
    results = stoody_book._search_pdf_pages(
        [
            {"page": 1, "text": "Cell division starts with interphase and then mitosis."},
            {"page": 2, "text": "Photosynthesis converts light into stored chemical energy."},
        ],
        "light energy",
        limit=5,
    )

    assert results[0]["page"] == 2
    assert "light" in results[0]["snippet"].lower()
    assert "energy" in results[0]["snippet"].lower()
    assert results[0]["score"] > 0


def test_search_pdf_pages_rejects_blank_query():
    with pytest.raises(HTTPException) as exc:
        stoody_book._search_pdf_pages([{"page": 1, "text": "text"}], "   ")

    assert exc.value.status_code == 400


def test_suggest_citations_uses_answer_and_question_overlap():
    citations = stoody_book._suggest_citations(
        question="What does chlorophyll do?",
        answer="Chlorophyll captures light energy for photosynthesis.",
        pages=[
            {"page": 1, "text": "Mitochondria release energy from glucose."},
            {"page": 2, "text": "Chlorophyll captures light energy inside chloroplasts."},
        ],
    )

    assert citations[0]["page"] == 2
    assert "Chlorophyll" in citations[0]["quote"]


def test_validate_annotation_payload_rejects_cross_document_quote():
    with pytest.raises(HTTPException) as exc:
        stoody_book._validate_annotation_payload(
            page_text="This page discusses acids and bases.",
            page=4,
            quote="mitosis and meiosis",
            note="important",
        )

    assert exc.value.status_code == 400
    assert "quote" in exc.value.detail.lower()
