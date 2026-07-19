from __future__ import annotations

import pytest


def _questions():
    return [
        {"id": "q-a", "text": "First", "marks": 1, "rubric": "One mark"},
        {"id": "q-b", "text": "Second", "marks": 2, "rubric": "Two marks"},
    ]


def _regions():
    return {
        "regions": [
            {
                "id": "q-b",
                "pageNumber": 2,
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 40,
                "order": 2,
                "label": "Q2",
                "ocrStatus": "completed",
            },
            {
                "id": "q-a",
                "pageNumber": 1,
                "x": 0,
                "y": 10,
                "width": 100,
                "height": 40,
                "order": 1,
                "label": "Q1",
                "ocrStatus": "completed",
            },
        ],
        "excluded_pages": [],
    }


def test_question_layout_is_complete_and_uses_printed_order():
    from services.exampen_paper_service import build_question_layout

    layout, errors = build_question_layout(
        {"document_id": "DOC-1", "pages_count": 2},
        _questions(),
        _regions(),
    )

    assert errors == []
    assert [item["source_question_id"] for item in layout] == ["q-a", "q-b"]
    assert layout[1]["page_number"] == 2
    assert layout[0]["bbox_percent"]["y"] == 10.0


def test_question_layout_rejects_missing_extra_and_overlapping_regions():
    from services.exampen_paper_service import build_question_layout

    regions = _regions()
    regions["regions"][0]["id"] = "unknown"
    regions["regions"][0]["pageNumber"] = 1
    regions["regions"][0]["y"] = 20

    _layout, errors = build_question_layout(
        {"document_id": "DOC-1", "pages_count": 2},
        _questions(),
        regions,
    )

    assert any("Question q-b" in error and "no saved" in error for error in errors)
    assert any("Region unknown" in error and "no reviewed" in error for error in errors)


def test_question_layout_uses_complete_reviewed_ocr_anchors_when_manual_boxes_are_not_required():
    from services.exampen_paper_service import build_question_layout

    document = {
        "document_id": "DOC-OCR",
        "pages_count": 2,
        "ocr_manual_segmentation_recommended": False,
        "ocr_layout_summary": {
            "pages": [
                {
                    "page": 1,
                    "page_height": 800,
                    "question_anchors": [
                        {"number": "1", "x": 50, "y": 100},
                        {"number": "2", "x": 50, "y": 500},
                    ],
                },
                {
                    "page": 2,
                    "page_height": 800,
                    "question_anchors": [{"number": "3", "x": 50, "y": 100}],
                },
            ]
        },
    }
    questions = [
        {"id": "q-a", "text": "First", "marks": 1, "question_number": 1, "page_number": 1},
        {"id": "q-b", "text": "Second", "marks": 1, "question_number": 2, "page_number": 1},
        {"id": "q-c", "text": "Third", "marks": 1, "question_number": 3, "page_number": 2},
    ]

    layout, errors = build_question_layout(document, questions, None)

    assert errors == []
    assert [item["source_question_id"] for item in layout] == ["q-a", "q-b", "q-c"]
    assert [item["page_number"] for item in layout] == [1, 1, 2]
    assert all(item["layout_source"] == "reviewed_ocr_anchor" for item in layout)
    assert layout[0]["bbox_percent"]["y"] == 0.0
    assert layout[0]["bbox_percent"]["height"] == 37.5
    assert layout[1]["bbox_percent"]["y"] == 37.5
    assert layout[2]["bbox_percent"]["height"] == 100.0


def test_question_layout_does_not_bypass_incomplete_or_untrusted_ocr_anchors():
    from services.exampen_paper_service import build_question_layout

    document = {
        "document_id": "DOC-OCR",
        "pages_count": 1,
        "ocr_manual_segmentation_recommended": False,
        "ocr_layout_summary": {
            "pages": [{"page": 1, "question_anchors": [{"number": "1", "y": 100}]}]
        },
    }
    questions = [
        {"id": "q-a", "text": "First", "marks": 1, "question_number": 1, "page_number": 1},
        {"id": "q-b", "text": "Second", "marks": 1, "question_number": 2, "page_number": 1},
    ]

    layout, errors = build_question_layout(document, questions, None)
    assert layout == []
    assert errors == ["Question 2: no reviewed OCR page anchor"]

    document["ocr_manual_segmentation_recommended"] = True
    layout, errors = build_question_layout(document, questions[:1], None)
    assert layout == []
    assert "Segment every printed question" in errors[0]


@pytest.mark.asyncio
async def test_paper_snapshot_hash_and_rows_include_layout():
    from mongomock_motor import AsyncMongoMockClient

    from services.exampen_paper_service import (
        build_question_layout,
        create_paper_snapshot,
    )

    db = AsyncMongoMockClient()["paper_layout_test"]
    document = {
        "document_id": "DOC-1",
        "exam_mode": "pcr",
        "title": "Layout paper",
        "pages_count": 2,
    }
    layout, errors = build_question_layout(document, _questions(), _regions())
    assert errors == []

    version = await create_paper_snapshot(
        db,
        document,
        _questions(),
        question_layout=layout,
    )

    assert version["layout_status"] == "verified"
    assert version["question_layout"] == layout
    rows = await db["exampen_paper_questions"].find(
        {"paper_version_id": version["paper_version_id"]}
    ).sort("position", 1).to_list(length=10)
    assert [row["layout"]["source_region_id"] for row in rows] == ["q-a", "q-b"]

    moved_layout = [dict(item) for item in layout]
    moved_layout[0] = {
        **moved_layout[0],
        "bbox_percent": {**moved_layout[0]["bbox_percent"], "y": 11.0},
    }
    moved = await create_paper_snapshot(
        db,
        document,
        _questions(),
        question_layout=moved_layout,
    )
    assert moved["content_hash"] != version["content_hash"]
