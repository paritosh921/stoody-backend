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


def test_build_view_state_keeps_safe_pdf_page_and_zoom():
    pdf_id = "64f000000000000000000001"
    state = stoody_book._build_view_state({
        "pdf_id": pdf_id,
        "page": 3,
        "zoom": 5,
        "focused_quote": {"page": 3, "quote": "  important idea  "},
    })

    assert state == {
        "pdf_id": pdf_id,
        "page": 3,
        "zoom": 2.0,
        "focused_quote": {"page": 3, "quote": "important idea"},
    }


def test_record_learning_event_schedules_due_review():
    now = stoody_book._now()
    state = stoody_book._record_learning_event(
        {},
        {
            "concept": "Photosynthesis",
            "outcome": "correct",
            "page": 2,
            "quote": "Plants convert light energy.",
            "prompt": "What does photosynthesis convert?",
        },
        now=now,
    )

    concept = state["concepts"][0]
    assert concept["label"] == "Photosynthesis"
    assert concept["box"] == 2
    assert concept["due_at"] > now.isoformat()
    assert concept["checks"][0]["outcome"] == "correct"


def test_compute_due_reviews_returns_due_concepts_first():
    state = {
        "concepts": [
            {
                "label": "Late concept",
                "normalized": "late concept",
                "page": 4,
                "quote": "Late concept quote",
                "due_at": "2999-01-01T00:00:00",
            },
            {
                "label": "Due concept",
                "normalized": "due concept",
                "page": 1,
                "quote": "Due concept quote",
                "due_at": "2000-01-01T00:00:00",
            },
        ],
    }

    reviews = stoody_book._compute_due_reviews(state, now=stoody_book._now())

    assert reviews == [
        {
            "label": "Due concept",
            "page": 1,
            "quote": "Due concept quote",
            "due_at": "2000-01-01T00:00:00",
        }
    ]


def test_build_study_check_anchors_to_matching_page():
    check = stoody_book._build_study_check(
        [
            {"page": 1, "text": "Mitochondria release energy."},
            {"page": 2, "text": "Photosynthesis converts light into chemical energy."},
        ],
        "photosynthesis energy",
    )

    assert check["page"] == 2
    assert check["concept"] == "photosynthesis energy"
    assert "Before I explain" in check["prompt"]
